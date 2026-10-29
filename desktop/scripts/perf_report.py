#!/usr/bin/env python3
"""Performance report from one or more `undertext-cli soak` logs.

    python3 scripts/perf_report.py soak.jsonl                      # one run
    python3 scripts/perf_report.py normal.jsonl saving.jsonl \\
        --labels "Normal" "Power saving" --bucket 1 --out perf/

Writes to --out (default: perf/):
  summary.md       time per call of each stage (mean, p50, p90, p99, max), how busy each
                   stage keeps its processor, subtitle latency, and start-vs-end trends
  over_time.png    line charts per time bucket: stage times, backlog behind the live
                   audio, subtitle latency and the macOS thermal state
  buckets.csv      the numbers behind the charts

Needs matplotlib (pip install matplotlib). The first bucket is left out of the trends
because it includes loading the models.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

STAGES = [
    # key in the stats record, name in the report, processor
    ("asr", "Speech recognition (Parakeet-TDT)", "CPU"),
    ("draft", "Draft translation (Opus-MT)", "CPU"),
    ("llm", "Translation (Qwen3.5-2B, llama-server)", "GPU"),
    ("speaker", "Speaker voice print (TitaNet)", "CPU"),
]
THERMAL = ["nominal", "fair", "serious", "critical"]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def sentences(records):
    """Per sentence: when the English appeared / was final, when Chinese appeared / was final."""
    s = {}
    for r in records:
        sid = r.get("sid")
        if sid is None or r.get("kind") not in ("source", "source_final", "draft", "commit", "final"):
            continue
        t = r["t"]
        e = s.setdefault(sid, {"en_first": t, "zh_first": None, "en_final": None, "zh_final": None})
        if e["zh_first"] is None and (r.get("zh") or r.get("zh_unstable")):
            e["zh_first"] = t
        if r["kind"] == "source_final":
            e["en_final"] = t
        if r["kind"] == "final":
            e["zh_final"] = t
    out = []
    for e in s.values():
        first = e["zh_first"] - e["en_first"] if e["zh_first"] is not None else None
        final = e["zh_final"] - e["en_final"] if e["zh_final"] is not None and e["en_final"] is not None else None
        out.append((e["en_first"], first, final))
    return out


def analyse(records, bucket_s):
    stats = [r for r in records if r.get("kind") == "stats"]
    if not stats:
        sys.exit("no stats records: run `undertext-cli soak` to make the log")
    t_end = stats[-1]["t"]
    config = next((r for r in records if r.get("kind") == "soak_config"), {})

    # per call samples (logs from before samples were recorded: mean per stats window)
    calls = defaultdict(list)          # stage -> [(t, ms, audio_s)]
    prev = None
    for r in stats:
        for key, _, _ in STAGES:
            m = r.get(key, {})
            if "samples_ms" in m:
                units = m.get("samples_units") or [0] * len(m["samples_ms"])
                calls[key] += [(r["t"], ms, u / 16000) for ms, u in zip(m["samples_ms"], units)]
            elif prev is not None:
                n = m.get("calls", 0) - prev.get(key, {}).get("calls", 0)
                busy = m.get("busy_s", 0) - prev.get(key, {}).get("busy_s", 0)
                audio = (m.get("units", 0) - prev.get(key, {}).get("units", 0)) / 16000
                calls[key] += [(r["t"], busy * 1000 / n, audio / n)] * max(n, 0) if n > 0 else []
        prev = r

    # time buckets
    nb = int(t_end // bucket_s) + 1
    b = [defaultdict(list) for _ in range(nb)]
    for key, rows in calls.items():
        for t, ms, audio in rows:
            i = min(int(t // bucket_s), nb - 1)
            b[i][key + "_ms"].append(ms)
            if audio > 0:
                b[i][key + "_audio"].append(audio)
    for r in stats:
        i = min(int(r["t"] // bucket_s), nb - 1)
        b[i]["lag"].append(r.get("max_lag_s", r.get("lag_s", 0)))
    for r in records:
        if r.get("kind") == "thermal" and r.get("state") in THERMAL:
            b[min(int(r["t"] // bucket_s), nb - 1)]["thermal"].append(THERMAL.index(r["state"]))
    sent = sentences(records)
    for t, first, final in sent:
        i = min(int(t // bucket_s), nb - 1)
        if first is not None:
            b[i]["first_zh"].append(first)
        if final is not None:
            b[i]["final_zh"].append(final)

    rows = []
    for i, x in enumerate(b):
        row = {"minute": (i + 0.5) * bucket_s / 60}
        for key, _, _ in STAGES:
            ms = x.get(key + "_ms", [])
            row[key + "_mean_ms"] = statistics.fmean(ms) if ms else None
            row[key + "_p90_ms"] = pct(ms, 0.9)
        audio = sum(x.get("asr_audio", []))
        row["asr_ms_per_audio_s"] = sum(x["asr_ms"]) / audio if audio > 0 else None
        row["max_lag_s"] = max(x["lag"]) if x.get("lag") else None
        row["first_zh_p50_s"] = statistics.median(x["first_zh"]) if x.get("first_zh") else None
        row["final_zh_p50_s"] = statistics.median(x["final_zh"]) if x.get("final_zh") else None
        row["thermal"] = max(x["thermal"]) if x.get("thermal") else None
        row["sentences"] = len(x.get("first_zh", []))
        rows.append(row)
    # the last bucket is usually cut short by the end of the audio: drop it if under half full
    if len(rows) > 1 and t_end - (nb - 1) * bucket_s < bucket_s / 2:
        rows.pop()
    return {"config": config, "calls": calls, "rows": rows, "sentences": sent, "t_end": t_end, "stats": stats}


def fmt(v, d=0):
    return "-" if v is None else f"{v:.{d}f}"


def trend(rows, key):
    """Start vs end (mean of the first and last quarter, warm-up bucket left out) and the
    least-squares slope per hour."""
    pts = [(r["minute"], r[key]) for r in rows[1:] if r[key] is not None]
    if len(pts) < 4:
        return None
    q = max(1, len(pts) // 4)
    start, end = statistics.fmean(p[1] for p in pts[:q]), statistics.fmean(p[1] for p in pts[-q:])
    mx, my = statistics.fmean(p[0] for p in pts), statistics.fmean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx * 60 if sxx else 0.0
    return start, end, slope


def summary(label, a):
    c, wall = a["config"], a["stats"][-1].get("elapsed_s", a["t_end"])
    audio_s = a["stats"][-1].get("audio_s", 0)
    out = [f"## {label}", ""]
    out.append(
        f"{audio_s / 60:.1f} min of audio in {wall / 60:.1f} min. Power saving: {c.get('power_saving', '?')}, "
        f"speaker labels: {c.get('diarize', '?')}, translation model: {c.get('commit_model', '?')}."
    )
    out += ["", "### Time per call", "",
            "| Stage | Runs on | Calls | Mean ms | p50 ms | p90 ms | p99 ms | Max ms | Busy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, name, unit in STAGES:
        ms = [x[1] for x in a["calls"].get(key, [])]
        if not ms:
            continue
        busy = sum(ms) / 1000 / wall * 100
        out.append(f"| {name} | {unit} | {len(ms)} | {statistics.fmean(ms):.0f} | {pct(ms, .5):.0f} | "
                   f"{pct(ms, .9):.0f} | {pct(ms, .99):.0f} | {max(ms):.0f} | {busy:.0f}% |")
    asr = a["calls"].get("asr", [])
    audio = sum(x[2] for x in asr)
    if audio > 0:
        ms = sum(x[1] for x in asr)
        out += ["", f"Speech recognition decodes {audio / 60:.1f} min of audio (the open segment is decoded "
                    f"again every 0.5 s) at {ms / audio:.0f} ms per second of audio, a real-time factor of "
                    f"{ms / audio / 1000:.3f}."]
    first = [s[1] for s in a["sentences"] if s[1] is not None]
    final = [s[2] for s in a["sentences"] if s[2] is not None]
    lags = [r.get("max_lag_s", 0) for r in a["stats"][1:]]
    out += ["", "### Subtitle latency", "",
            "| Measure | n | p50 s | p90 s | Max s |", "|---|---:|---:|---:|---:|"]
    for name, v in [("First Chinese after the English appears", first),
                    ("Final Chinese after the English sentence ends", final)]:
        if v:
            out.append(f"| {name} | {len(v)} | {pct(v, .5):.2f} | {pct(v, .9):.2f} | {max(v):.2f} |")
    if lags:
        out.append(f"| Backlog behind the live audio | {len(lags)} | {pct(lags, .5):.2f} | {pct(lags, .9):.2f} | {max(lags):.2f} |")
    out += ["", "### Start vs end", ""]
    if len([r for r in a["rows"][1:]]) < 4:
        out.append("The run is too short for a start-vs-end comparison (needs at least 5 buckets).")
        return "\n".join(out) + "\n"
    out += ["Mean of the first and last quarter of the run (first bucket left out), and the trend per hour.", "",
            "| Measure | Start | End | Change | Trend per hour |", "|---|---:|---:|---:|---:|"]
    for key, name, d in [("asr_ms_per_audio_s", "Speech recognition, ms per audio second", 0),
                         ("draft_mean_ms", "Opus-MT, ms per call", 0), ("llm_mean_ms", "Qwen, ms per call", 0),
                         ("speaker_mean_ms", "TitaNet, ms per call", 0), ("max_lag_s", "Backlog, s", 2),
                         ("first_zh_p50_s", "First Chinese, s", 2), ("final_zh_p50_s", "Final Chinese, s", 2)]:
        t = trend(a["rows"], key)
        if t:
            s, e, slope = t
            ch = f"{(e / s - 1) * 100:+.0f}%" if s else "-"
            out.append(f"| {name} | {s:.{d}f} | {e:.{d}f} | {ch} | {slope:+.{max(d, 1)}f} |")
    th = [r["thermal"] for r in a["rows"] if r["thermal"] is not None]
    if th:
        out += ["", f"Worst thermal state: {THERMAL[max(th)]}."]
    return "\n".join(out) + "\n"


def plot(runs, path, bucket_min):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    colors = {"asr": "#2563eb", "draft": "#16a34a", "llm": "#dc2626", "speaker": "#9333ea"}
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2, 1]})
    for n, (label, a) in enumerate(runs):
        ls = styles[n % len(styles)]
        suffix = f" ({label})" if len(runs) > 1 else ""
        x = [r["minute"] for r in a["rows"]]
        y = lambda k: [r[k] if r[k] is not None else float("nan") for r in a["rows"]]
        ax[0].plot(x, y("asr_ms_per_audio_s"), ls, color=colors["asr"], label="Parakeet, ms per audio s" + suffix)
        for key, name in [("draft", "Opus-MT"), ("llm", "Qwen"), ("speaker", "TitaNet")]:
            if any(r[key + "_mean_ms"] is not None for r in a["rows"]):
                ax[0].plot(x, y(key + "_mean_ms"), ls, color=colors[key], label=f"{name}, ms per call" + suffix)
        ax[1].plot(x, y("first_zh_p50_s"), ls, color="#ea580c", label="First Chinese (median)" + suffix)
        ax[1].plot(x, y("final_zh_p50_s"), ls, color="#0891b2", label="Final Chinese (median)" + suffix)
        ax[2].plot(x, y("max_lag_s"), ls, color="#475569", label="Worst backlog" + suffix)
        ax[3].step(x, y("thermal"), ls, where="mid", color="#b45309", label=label if len(runs) > 1 else None)
    # Qwen takes hundreds of ms, the CPU stages tens: a log scale keeps all of them readable
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    low = min((v for l in ax[0].get_lines() for v in l.get_ydata() if v == v and v > 0), default=10)
    ax[0].set_ylim(bottom=min(10, low * 0.8))
    ax[0].set_ylabel("ms (log scale)")
    ax[0].set_title("Time per stage", loc="left", fontsize=11)
    ax[1].set_ylabel("seconds")
    ax[1].set_title("Subtitle latency after the English appears / ends", loc="left", fontsize=11)
    ax[2].set_ylabel("seconds")
    ax[2].set_title("Backlog behind the live audio", loc="left", fontsize=11)
    ax[3].set_yticks(range(4), THERMAL)
    ax[3].set_ylim(-0.5, 3.5)
    ax[3].set_title("macOS thermal state", loc="left", fontsize=11)
    ax[3].set_xlabel(f"minutes into the run ({bucket_min:g}-minute buckets)")
    for a_ in ax:
        a_.grid(alpha=0.25)
        a_.spines[["top", "right"]].set_visible(False)
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax[1].set_ylim(bottom=0)
    ax[2].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--bucket", type=float, default=1.0, help="bucket length in minutes (default 1)")
    p.add_argument("--out", default="perf")
    args = p.parse_args()
    labels = args.labels or [os.path.splitext(os.path.basename(x))[0] for x in args.logs]
    if len(labels) != len(args.logs):
        sys.exit("give one label per log")
    os.makedirs(args.out, exist_ok=True)
    runs = [(label, analyse(load(path), args.bucket * 60)) for label, path in zip(labels, args.logs)]

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("# UnderText performance\n\n")
        for label, a in runs:
            f.write(summary(label, a) + "\n")
    with open(os.path.join(args.out, "buckets.csv"), "w", newline="") as f:
        cols = list(runs[0][1]["rows"][0].keys())
        w = csv.writer(f)
        w.writerow(["run"] + cols)
        for label, a in runs:
            for r in a["rows"]:
                w.writerow([label] + [("" if r[c] is None else round(r[c], 3)) for c in cols])
    plot(runs, os.path.join(args.out, "over_time.png"), args.bucket)
    print(open(os.path.join(args.out, "summary.md")).read())
    print(f"wrote {args.out}/summary.md, {args.out}/over_time.png, {args.out}/buckets.csv")


if __name__ == "__main__":
    main()
