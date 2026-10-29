//! Side-by-side comparison of two runs on the same audio (e.g. two different
//! translation models): sentences are aligned by their English text, the final Chinese of
//! both runs is shown next to each other with the differences marked, together with
//! each run's compute cost and latency.

use std::fmt::Write;

use serde_json::Value;

use crate::report::{latency_report, SentenceLatency};

pub struct Run {
    pub name: String,
    pub config: Value,
    pub sentences: Vec<SentenceLatency>,
    /// Last cumulative `stats` record.
    pub stats: Option<Value>,
}

impl Run {
    pub fn from_records(name: &str, records: &[Value]) -> Self {
        let mut sentences: Vec<SentenceLatency> = latency_report(records).into_iter().filter(|s| !s.en.trim().is_empty()).collect();
        sentences.sort_by(|a, b| a.en_first.total_cmp(&b.en_first));
        Run {
            name: name.to_string(),
            config: records.iter().find(|r| r["kind"] == "soak_config").cloned().unwrap_or(Value::Null),
            sentences,
            stats: records.iter().rev().find(|r| r["kind"] == "stats").cloned(),
        }
    }

    fn model(&self) -> String {
        self.config["commit_model"].as_str().unwrap_or("?").to_string()
    }
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect()
}

/// Share of words two English sentences have in common (0..1).
fn similarity(a: &str, b: &str) -> f64 {
    let (wa, wb) = (words(a), words(b));
    if wa.is_empty() || wb.is_empty() {
        return 0.0;
    }
    let mut rest = wb.clone();
    let mut common = 0;
    for w in &wa {
        if let Some(i) = rest.iter().position(|x| x == w) {
            rest.swap_remove(i);
            common += 1;
        }
    }
    2.0 * common as f64 / (wa.len() + wb.len()) as f64
}

/// Global alignment of the two sentence lists (Needleman–Wunsch on English similarity);
/// sentences that were split differently end up as one-sided rows.
pub fn align(a: &[&str], b: &[&str]) -> Vec<(Option<usize>, Option<usize>)> {
    let (n, m) = (a.len(), b.len());
    const GAP: f64 = -0.2;
    let mut score = vec![vec![0.0f64; m + 1]; n + 1];
    for i in 1..=n {
        score[i][0] = i as f64 * GAP;
    }
    for j in 1..=m {
        score[0][j] = j as f64 * GAP;
    }
    let pair = |i: usize, j: usize| {
        let s = similarity(a[i], b[j]);
        if s >= 0.3 { s } else { -1.0 }
    };
    for i in 1..=n {
        for j in 1..=m {
            score[i][j] = (score[i - 1][j - 1] + pair(i - 1, j - 1)).max(score[i - 1][j] + GAP).max(score[i][j - 1] + GAP);
        }
    }
    let (mut i, mut j) = (n, m);
    let mut out = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && (score[i][j] - (score[i - 1][j - 1] + pair(i - 1, j - 1))).abs() < 1e-9 {
            out.push((Some(i - 1), Some(j - 1)));
            i -= 1;
            j -= 1;
        } else if i > 0 && (j == 0 || (score[i][j] - (score[i - 1][j] + GAP)).abs() < 1e-9) {
            out.push((Some(i - 1), None));
            i -= 1;
        } else {
            out.push((None, Some(j - 1)));
            j -= 1;
        }
    }
    out.reverse();
    out
}

/// Character diff: (text, changed) runs of `b` relative to `a` (longest common subsequence).
pub fn char_diff(a: &str, b: &str) -> Vec<(String, bool)> {
    let (x, y): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let (n, m) = (x.len(), y.len());
    let mut l = vec![vec![0u16; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            l[i][j] = if x[i] == y[j] { l[i + 1][j + 1] + 1 } else { l[i + 1][j].max(l[i][j + 1]) };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out: Vec<(String, bool)> = Vec::new();
    let mut push = |c: char, changed: bool| match out.last_mut() {
        Some((s, ch)) if *ch == changed => s.push(c),
        _ => out.push((c.to_string(), changed)),
    };
    while j < m {
        if i < n && x[i] == y[j] {
            push(y[j], false);
            i += 1;
            j += 1;
        } else if i < n && l[i + 1][j] >= l[i][j + 1] {
            i += 1;
        } else {
            push(y[j], true);
            j += 1;
        }
    }
    out
}

/// Share of `b`'s characters that also appear (in order) in `a`, and vice versa.
fn char_agreement(a: &str, b: &str) -> f64 {
    let same: usize = char_diff(a, b).iter().filter(|(_, c)| !c).map(|(s, _)| s.chars().count()).sum();
    let total = a.chars().count() + b.chars().count();
    if total == 0 { 1.0 } else { 2.0 * same as f64 / total as f64 }
}

/// Warnings for one translation. Length is judged against the English (Chinese chars per
/// English word is normally about 1.2 to 2.8) and the other run, so a run that invents
/// extra text is flagged without blaming the other one for being "short".
pub fn flags(en: &str, zh: &str, other: &str) -> Vec<&'static str> {
    let mut f = Vec::new();
    let n = zh.chars().filter(|c| !c.is_ascii_punctuation() && !"，。！？；：、“”（）".contains(*c)).count() as f64;
    let o = other.chars().filter(|c| !c.is_ascii_punctuation() && !"，。！？；：、“”（）".contains(*c)).count() as f64;
    let w = words(en).len().max(1) as f64;
    if zh.trim().is_empty() && !en.trim().is_empty() {
        f.push("no translation");
    }
    if n / w > 3.5 && n > o * 1.5 && n >= 8.0 {
        f.push("much longer than expected: invented or repeated text?");
    }
    if n > 0.0 && n / w < 0.8 && n < o * 0.6 && w >= 4.0 {
        f.push("much shorter than expected: something left out?");
    }
    let latin = zh.chars().filter(|c| c.is_ascii_alphabetic()).count() as f64;
    if n > 0.0 && latin / n > 0.3 {
        f.push("much English left in");
    }
    f
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

struct Summary {
    gpu_pct: f64,
    llm_ms: f64,
    llm_per_min: f64,
    prompt_tokens: f64,
    asr_pct: f64,
    first_zh: Option<f64>,
    final_zh: Option<f64>,
}

fn summary(r: &Run) -> Summary {
    let st = r.stats.clone().unwrap_or(Value::Null);
    let el = st["elapsed_s"].as_f64().unwrap_or(1.0).max(1e-6);
    let g = |m: &str, k: &str| st[m][k].as_f64().unwrap_or(0.0);
    let calls = g("llm", "calls").max(1.0);
    Summary {
        gpu_pct: g("llm", "busy_s") / el * 100.0,
        llm_ms: g("llm", "busy_s") * 1000.0 / calls,
        llm_per_min: g("llm", "calls") / el * 60.0,
        prompt_tokens: g("llm", "prompt_tokens") / calls,
        asr_pct: g("asr", "busy_s") / el * 100.0,
        first_zh: median(r.sentences.iter().filter_map(|s| s.first_zh_lag()).collect()),
        final_zh: median(r.sentences.iter().filter_map(|s| s.final_lag()).collect()),
    }
}

/// The whole comparison as a self-contained HTML page.
pub fn render_html(a: &Run, b: &Run) -> String {
    let ea: Vec<&str> = a.sentences.iter().map(|s| s.en.as_str()).collect();
    let eb: Vec<&str> = b.sentences.iter().map(|s| s.en.as_str()).collect();
    let pairs = align(&ea, &eb);
    let t0 = a.sentences.first().map(|s| s.en_first).unwrap_or(0.0);

    let (sa, sb) = (summary(a), summary(b));
    let matched: Vec<(&SentenceLatency, &SentenceLatency)> =
        pairs.iter().filter_map(|p| Some((&a.sentences[p.0?], &b.sentences[p.1?]))).collect();
    let identical = matched.iter().filter(|(x, y)| x.zh.trim() == y.zh.trim()).count();
    let agreement = if matched.is_empty() {
        0.0
    } else {
        matched.iter().map(|(x, y)| char_agreement(&x.zh, &y.zh)).sum::<f64>() / matched.len() as f64
    };
    let flagged_b = matched.iter().filter(|(x, y)| !flags(&y.en, &y.zh, &x.zh).is_empty()).count();
    let flagged_a = matched.iter().filter(|(x, y)| !flags(&x.en, &x.zh, &y.zh).is_empty()).count();

    let mut h = String::new();
    let _ = write!(
        h,
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Translation comparison</title><style>
:root{{--bg:#F6F7F9;--card:#FFFFFF;--ink:#1D2330;--muted:#667085;--line:#E3E6EB;--a:#2F6FDE;--b:#C2410C;--mark:#FDE7C8;--warn:#B42318}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151B;--card:#1A1E26;--ink:#E7EAF0;--muted:#98A2B3;--line:#2A303B;--a:#7DA7F5;--b:#F59E66;--mark:#5A3B12;--warn:#F97066}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"PingFang SC","Segoe UI",sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--muted);margin:0 0 20px}}
table{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}
th{{font-size:13px;color:var(--muted);font-weight:600;background:var(--card);position:sticky;top:0}}
.sum td{{font-variant-numeric:tabular-nums}} .sum td.n{{text-align:right}}
.ka{{color:var(--a);font-weight:600}} .kb{{color:var(--b);font-weight:600}}
.t{{color:var(--muted);font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums}}
.en{{color:var(--muted);font-size:13.5px}} .en2{{color:var(--muted);font-size:12.5px;font-style:italic;margin-top:4px}}
mark{{background:var(--mark);color:inherit;border-radius:2px}}
.flag{{display:block;color:var(--warn);font-size:12px;margin-top:3px}}
.same{{color:var(--muted);font-size:12px}}
.filters{{margin:18px 0 10px;display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:14px}}
.hide-same tr.same-row{{display:none}}
.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Translation comparison</h1>
<p class="sub">Final translations from two runs on the same audio. <span class="ka">A</span>: {}, <span class="kb">B</span>: {}. Characters in B that differ from A are highlighted.</p>
<div class="wrap"><table class="sum"><tr><th></th><th>Translation model</th><th>Power saving</th><th>GPU busy</th><th>Per call</th><th>Calls/min</th><th>Prompt tokens/call</th><th>ASR CPU</th><th>First Chinese</th><th>Final Chinese</th><th>Flagged sentences</th></tr>"#,
        esc(&a.model()),
        esc(&b.model())
    );
    let opt = |v: Option<f64>| v.map(|x| format!("{x:.2} s")).unwrap_or_else(|| "-".into());
    for (k, r, s, flagged) in [("A", a, &sa, flagged_a), ("B", b, &sb, flagged_b)] {
        let _ = write!(
            h,
            r#"<tr><td class="k{}">{k}</td><td>{}</td><td>{}</td><td class="n">{:.1}%</td><td class="n">{:.0} ms</td><td class="n">{:.1}</td><td class="n">{:.0}</td><td class="n">{:.1}%</td><td class="n">{}</td><td class="n">{}</td><td class="n">{flagged}</td></tr>"#,
            k.to_lowercase(),
            esc(&r.model()),
            r.config["power_saving"].as_bool().map(|p| if p { "on" } else { "off" }).unwrap_or("-"),
            s.gpu_pct,
            s.llm_ms,
            s.llm_per_min,
            s.prompt_tokens,
            s.asr_pct,
            opt(s.first_zh),
            opt(s.final_zh)
        );
    }
    let _ = write!(
        h,
        r#"</table></div>
<p class="sub" style="margin-top:12px">{} sentences aligned; {} translated identically ({:.0}%); {:.0}% of characters agree on average. Latencies are medians. A sentence is flagged when its translation is much longer or shorter than the English suggests, missing, or mostly English.</p>
<div class="filters"><label><input type="checkbox" id="hs"> Only sentences that differ</label><label><input type="checkbox" id="hf"> Only flagged sentences</label></div>
<div class="wrap"><table id="rows"><tr><th>Time</th><th>English</th><th><span class="ka">A</span></th><th><span class="kb">B</span></th></tr>"#,
        matched.len(),
        identical,
        identical as f64 * 100.0 / matched.len().max(1) as f64,
        agreement * 100.0
    );
    for (ia, ib) in &pairs {
        let sa = ia.map(|i| &a.sentences[i]);
        let sb = ib.map(|i| &b.sentences[i]);
        let t = sa.or(sb).map(|s| s.en_first - t0).unwrap_or(0.0).max(0.0);
        let za = sa.map(|s| s.zh.as_str()).unwrap_or("");
        let zb = sb.map(|s| s.zh.as_str()).unwrap_or("");
        let same = sa.is_some() && sb.is_some() && za.trim() == zb.trim();
        let fa = sa.map(|s| flags(&s.en, za, zb)).unwrap_or_default();
        let fb = sb.map(|s| flags(&s.en, zb, za)).unwrap_or_default();
        let mut en = esc(sa.or(sb).map(|s| s.en.as_str()).unwrap_or(""));
        if let (Some(x), Some(y)) = (sa, sb) {
            if words(&x.en) != words(&y.en) {
                let _ = write!(en, r#"<div class="en2">B heard: {}</div>"#, esc(&y.en));
            }
        }
        let cell_b = if sb.is_none() {
            r#"<span class="same">(no matching sentence in B)</span>"#.to_string()
        } else if same {
            format!(r#"{}<div class="same">same as A</div>"#, esc(zb))
        } else {
            char_diff(za, zb).iter().map(|(s, c)| if *c { format!("<mark>{}</mark>", esc(s)) } else { esc(s) }).collect()
        };
        let cell_a = if sa.is_none() { r#"<span class="same">(no matching sentence in A)</span>"#.to_string() } else { esc(za) };
        let flag_html = |f: &[&str]| f.iter().map(|x| format!(r#"<div class="flag">⚠ {x}</div>"#)).collect::<String>();
        let _ = write!(
            h,
            r#"<tr class="{}{}"><td class="t">{:02}:{:02}</td><td class="en">{en}</td><td>{cell_a}{}</td><td>{cell_b}{}</td></tr>"#,
            if same { "same-row" } else { "" },
            if fa.is_empty() && fb.is_empty() { " ok-row" } else { "" },
            (t / 60.0) as u64,
            (t % 60.0) as u64,
            flag_html(&fa),
            flag_html(&fb)
        );
    }
    let _ = write!(
        h,
        r#"</table></div></main><script>
const rows=document.getElementById('rows');
const hs=document.getElementById('hs'),hf=document.getElementById('hf');
function apply(){{rows.classList.toggle('hide-same',hs.checked);for(const r of rows.querySelectorAll('tr.ok-row'))r.style.display=hf.checked?'none':'';}}
hs.onchange=apply;hf.onchange=apply;
</script></body></html>"#
    );
    h
}

/// Plain-text summary for the terminal.
pub fn render_text(a: &Run, b: &Run) -> String {
    let mut s = String::new();
    for (k, r) in [("A", a), ("B", b)] {
        let x = summary(r);
        let _ = writeln!(
            s,
            "{k}: {:28} GPU {:5.1}%  LLM {:4.0} ms/call {:5.1}/min {:4.0} tok  ASR {:4.1}%  1st zh {}  final zh {}",
            r.model(),
            x.gpu_pct,
            x.llm_ms,
            x.llm_per_min,
            x.prompt_tokens,
            x.asr_pct,
            x.first_zh.map(|v| format!("{v:.2}s")).unwrap_or("-".into()),
            x.final_zh.map(|v| format!("{v:.2}s")).unwrap_or("-".into())
        );
    }
    s
}
