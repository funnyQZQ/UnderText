//! Per-sentence latency report from an event log.

use std::collections::BTreeMap;

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct SentenceLatency {
    pub sid: u64,
    pub en: String,
    pub zh: String,
    pub en_first: f64,
    pub zh_first: Option<f64>,
    pub zh_stable_first: Option<f64>,
    pub en_final: Option<f64>,
    pub zh_final: Option<f64>,
    pub speaker: Option<String>,
}

impl SentenceLatency {
    pub fn first_zh_lag(&self) -> Option<f64> {
        self.zh_first.map(|t| t - self.en_first)
    }

    pub fn final_lag(&self) -> Option<f64> {
        Some(self.zh_final? - self.en_final?)
    }
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

pub fn latency_report(records: &[Value]) -> Vec<SentenceLatency> {
    let mut by_sid: BTreeMap<u64, SentenceLatency> = BTreeMap::new();
    for r in records {
        let Some(sid) = r.get("sid").and_then(Value::as_u64) else { continue };
        let t = r.get("t").and_then(Value::as_f64).unwrap_or(0.0);
        let kind = s(r, "kind");
        let e = by_sid.entry(sid).or_insert_with(|| SentenceLatency {
            sid,
            en: String::new(),
            zh: String::new(),
            en_first: t,
            zh_first: None,
            zh_stable_first: None,
            en_final: None,
            zh_final: None,
            speaker: None,
        });
        let en = s(r, "en");
        if !en.is_empty() {
            e.en = en;
        }
        if let Some(sp) = r.get("speaker").and_then(Value::as_str) {
            e.speaker = Some(sp.to_string());
        }
        let (zh, zh_u) = (s(r, "zh"), s(r, "zh_unstable"));
        if (!zh.is_empty() || !zh_u.is_empty()) && e.zh_first.is_none() {
            e.zh_first = Some(t);
        }
        if !zh.is_empty() && e.zh_stable_first.is_none() {
            e.zh_stable_first = Some(t);
        }
        if kind == "source_final" {
            e.en_final = Some(t);
        }
        if kind == "final" {
            e.zh_final = Some(t);
            e.zh = zh;
        }
    }
    by_sid.into_values().collect()
}

pub fn summarize(values: &[f64]) -> String {
    if values.is_empty() {
        return "n/a".into();
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let median = if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
    let p90 = v[((0.9 * (n - 1) as f64).round_ties_even() as usize).min(n - 1)];
    format!("median {median:.2}s  p90 {p90:.2}s  max {:.2}s  (n={n})", v[n - 1])
}
