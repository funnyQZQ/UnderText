//! Long-run (soak) analysis of an event log: does any stage get slower over time?
//!
//! The engine writes cumulative `stats` records every few seconds (see `metrics`), the
//! soak runner adds `thermal` records, and sentences carry their own timings. This
//! module cuts the run into time buckets and reports, per bucket:
//! * speech recognition cost normalized by audio length (ms per second of audio decoded),
//!   so a rise means slower hardware (thermal throttling), not longer sentences;
//! * Opus-MT / LLM / speaker time per call, LLM time per prompt token;
//! * how far processing is behind the live audio (backlog);
//! * end-to-end subtitle latency (first Chinese / final Chinese after the English).

use std::collections::BTreeMap;
use std::fmt::Write;

use serde_json::Value;

use crate::report::latency_report;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bucket {
    /// Start / end in log seconds.
    pub t0: f64,
    pub t1: f64,
    pub asr_ms_per_audio_s: Option<f64>,
    pub asr_max_ms: f64,
    pub draft_avg_ms: Option<f64>,
    pub llm_avg_ms: Option<f64>,
    pub llm_max_ms: f64,
    pub llm_ms_per_prompt_token: Option<f64>,
    pub llm_calls_per_min: f64,
    pub speaker_avg_ms: Option<f64>,
    /// Backlog behind live audio at the end of the bucket / worst inside it.
    pub lag_s: f64,
    pub max_lag_s: f64,
    pub first_zh_lag: Option<f64>,
    pub final_lag: Option<f64>,
    pub sentences: usize,
    /// Worst macOS thermal state seen (nominal / fair / serious / critical).
    pub thermal: Option<String>,
}

fn f(v: &Value, path: &[&str]) -> f64 {
    let mut x = v;
    for p in path {
        x = &x[*p];
    }
    x.as_f64().unwrap_or(0.0)
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

const THERMAL: [&str; 4] = ["nominal", "fair", "serious", "critical"];

/// Split a soak log into buckets of `bucket_s` seconds.
pub fn buckets(records: &[Value], bucket_s: f64) -> Vec<Bucket> {
    let stats: Vec<&Value> = records.iter().filter(|r| r["kind"] == "stats").collect();
    if stats.is_empty() {
        return Vec::new();
    }
    let key = |t: f64| (t / bucket_s).floor() as i64;
    let mut groups: BTreeMap<i64, Vec<&Value>> = BTreeMap::new();
    for s in &stats {
        groups.entry(key(f(s, &["t"]))).or_default().push(s);
    }
    let rows = latency_report(records);
    let mut thermal: BTreeMap<i64, usize> = BTreeMap::new();
    for r in records.iter().filter(|r| r["kind"] == "thermal") {
        if let Some(i) = r["state"].as_str().and_then(|s| THERMAL.iter().position(|x| *x == s)) {
            let e = thermal.entry(key(f(r, &["t"]))).or_insert(0);
            *e = (*e).max(i);
        }
    }

    let mut out = Vec::new();
    let mut prev: Option<&Value> = None;
    for (k, group) in &groups {
        let last = *group.last().unwrap();
        let d = |path: &[&str]| f(last, path) - prev.map(|p| f(p, path)).unwrap_or(0.0);
        let per_call = |m: &str| {
            let calls = d(&[m, "calls"]);
            (calls > 0.0).then(|| d(&[m, "busy_s"]) * 1000.0 / calls)
        };
        let wall = (d(&["elapsed_s"])).max(1e-6);
        let asr_audio_s = d(&["asr", "units"]) / 16000.0;
        let prompt = d(&["llm", "prompt_tokens"]);
        let (t0, t1) = (*k as f64 * bucket_s, (*k + 1) as f64 * bucket_s);
        let in_bucket: Vec<_> = rows.iter().filter(|r| r.en_first >= t0 && r.en_first < t1).collect();
        out.push(Bucket {
            t0,
            t1,
            asr_ms_per_audio_s: (asr_audio_s > 0.0).then(|| d(&["asr", "busy_s"]) * 1000.0 / asr_audio_s),
            asr_max_ms: group.iter().map(|g| f(g, &["asr", "max_ms"])).fold(0.0, f64::max),
            draft_avg_ms: per_call("draft"),
            llm_avg_ms: per_call("llm"),
            llm_max_ms: group.iter().map(|g| f(g, &["llm", "max_ms"])).fold(0.0, f64::max),
            llm_ms_per_prompt_token: (prompt > 0.0).then(|| d(&["llm", "busy_s"]) * 1000.0 / prompt),
            llm_calls_per_min: d(&["llm", "calls"]) / wall * 60.0,
            speaker_avg_ms: per_call("speaker"),
            lag_s: f(last, &["lag_s"]),
            max_lag_s: group.iter().map(|g| f(g, &["max_lag_s"])).fold(0.0, f64::max),
            first_zh_lag: median(in_bucket.iter().filter_map(|r| r.first_zh_lag()).collect()),
            final_lag: median(in_bucket.iter().filter_map(|r| r.final_lag()).collect()),
            sentences: in_bucket.len(),
            thermal: thermal.get(k).map(|i| THERMAL[*i].to_string()),
        });
        prev = Some(last);
    }
    out
}

fn cell(v: Option<f64>, digits: usize) -> String {
    v.map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "-".into())
}

pub fn render_table(bs: &[Bucket]) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:>11} {:>9} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7} {:>7} {:>7} {:>8} {:>8} {:>5}  thermal",
        "time(min)", "ASR ms/s", "ASR max", "draft", "LLM ms", "LLM max", "ms/token", "LLM/min", "lag s", "maxlag", "1st zh", "final zh", "sent"
    );
    for b in bs {
        let _ = writeln!(
            s,
            "{:>5.1}-{:<5.1} {:>9} {:>8.0} {:>8} {:>8} {:>8.0} {:>9} {:>7.1} {:>7.2} {:>7.2} {:>8} {:>8} {:>5}  {}",
            b.t0 / 60.0,
            b.t1 / 60.0,
            cell(b.asr_ms_per_audio_s, 1),
            b.asr_max_ms,
            cell(b.draft_avg_ms, 0),
            cell(b.llm_avg_ms, 0),
            b.llm_max_ms,
            cell(b.llm_ms_per_prompt_token, 2),
            b.llm_calls_per_min,
            b.lag_s,
            b.max_lag_s,
            cell(b.first_zh_lag, 2),
            cell(b.final_lag, 2),
            b.sentences,
            b.thermal.as_deref().unwrap_or("-")
        );
    }
    s
}

pub fn to_csv(bs: &[Bucket]) -> String {
    let mut s = String::from("t0_min,t1_min,asr_ms_per_audio_s,asr_max_ms,draft_avg_ms,llm_avg_ms,llm_max_ms,llm_ms_per_prompt_token,llm_calls_per_min,speaker_avg_ms,lag_s,max_lag_s,first_zh_lag_s,final_lag_s,sentences,thermal\n");
    let o = |v: Option<f64>| v.map(|x| format!("{x:.3}")).unwrap_or_default();
    for b in bs {
        let _ = writeln!(
            s,
            "{:.2},{:.2},{},{:.1},{},{},{:.1},{},{:.2},{},{:.3},{:.3},{},{},{},{}",
            b.t0 / 60.0,
            b.t1 / 60.0,
            o(b.asr_ms_per_audio_s),
            b.asr_max_ms,
            o(b.draft_avg_ms),
            o(b.llm_avg_ms),
            b.llm_max_ms,
            o(b.llm_ms_per_prompt_token),
            b.llm_calls_per_min,
            o(b.speaker_avg_ms),
            b.lag_s,
            b.max_lag_s,
            o(b.first_zh_lag),
            o(b.final_lag),
            b.sentences,
            b.thermal.as_deref().unwrap_or("")
        );
    }
    s
}

/// Compare the start of the run with its end. `n` buckets on each side (the very first
/// bucket is skipped: model warm-up). Returns one line per stage; lines starting with
/// "!" mark a slowdown of more than 25 %.
pub fn verdict(bs: &[Bucket], n: usize) -> Vec<String> {
    let body: Vec<&Bucket> = bs.iter().skip(1).collect();
    if body.len() < 2 * n.max(1) {
        return vec!["run too short for a start-vs-end comparison".into()];
    }
    let (head, tail) = (&body[..n], &body[body.len() - n..]);
    let mean = |xs: &[&Bucket], g: &dyn Fn(&Bucket) -> Option<f64>| {
        let v: Vec<f64> = xs.iter().filter_map(|b| g(b)).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let checks: [(&str, &dyn Fn(&Bucket) -> Option<f64>, f64); 7] = [
        ("speech recognition, ms per audio second", &|b| b.asr_ms_per_audio_s, 2.0),
        ("Opus-MT draft, ms per call", &|b| b.draft_avg_ms, 5.0),
        ("LLM, ms per call", &|b| b.llm_avg_ms, 20.0),
        ("LLM, ms per prompt token", &|b| b.llm_ms_per_prompt_token, 0.05),
        ("speaker voice prints, ms per call", &|b| b.speaker_avg_ms, 5.0),
        ("first Chinese after English, s", &|b| b.first_zh_lag, 0.2),
        ("final Chinese after English, s", &|b| b.final_lag, 0.2),
    ];
    let mut out = Vec::new();
    for (name, g, min_abs) in checks {
        let (Some(a), Some(b)) = (mean(head, g), mean(tail, g)) else { continue };
        let worse = b > a * 1.25 && b - a > min_abs;
        let pct = if a > 0.0 { (b / a - 1.0) * 100.0 } else { 0.0 };
        out.push(format!("{} {name}: {a:.2} -> {b:.2} ({pct:+.0}%)", if worse { "!" } else { " " }));
    }
    let lag_end = tail.iter().map(|b| b.max_lag_s).fold(0.0, f64::max);
    let lag_start = head.iter().map(|b| b.max_lag_s).fold(0.0, f64::max);
    out.push(format!(
        "{} backlog behind live audio (worst): {lag_start:.2} s -> {lag_end:.2} s",
        if lag_end > 1.0 && lag_end > lag_start + 0.5 { "!" } else { " " }
    ));
    out
}
