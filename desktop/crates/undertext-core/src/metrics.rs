//! Process-wide work counters: how much time each model spends computing, so heat /
//! battery problems can be traced to a component (`stats` records in the event log).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{json, Value};

pub struct Meter {
    busy_us: AtomicU64,
    calls: AtomicU64,
    /// Amount of work done (ASR: audio samples decoded), to tell slower hardware
    /// (thermal throttling) from bigger inputs.
    units: AtomicU64,
    /// Longest single call since the last `snapshot`.
    max_us: AtomicU64,
    /// LLM only: prompt tokens evaluated / taken from cache / generated
    prompt_tokens: AtomicU64,
    cached_tokens: AtomicU64,
    predicted_tokens: AtomicU64,
    /// Every call since the last `snapshot`: (ms, units), for percentiles in the analysis.
    samples: Mutex<Vec<(f32, u64)>>,
}

impl Meter {
    const fn new() -> Self {
        Self {
            busy_us: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            units: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            predicted_tokens: AtomicU64::new(0),
            samples: Mutex::new(Vec::new()),
        }
    }

    /// Run `f`, adding its wall time to this meter.
    pub fn time<R>(&self, f: impl FnOnce() -> R) -> R {
        self.time_units(0, f)
    }

    /// Like `time`, also counting `units` of work (e.g. audio samples).
    pub fn time_units<R>(&self, units: u64, f: impl FnOnce() -> R) -> R {
        let t0 = Instant::now();
        let r = f();
        let us = t0.elapsed().as_micros() as u64;
        self.busy_us.fetch_add(us, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.units.fetch_add(units, Ordering::Relaxed);
        self.max_us.fetch_max(us, Ordering::Relaxed);
        if let Ok(mut s) = self.samples.lock() {
            s.push((us as f32 / 1000.0, units));
        }
        r
    }

    pub fn add_tokens(&self, prompt: u64, cached: u64, predicted: u64) {
        self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        self.cached_tokens.fetch_add(cached, Ordering::Relaxed);
        self.predicted_tokens.fetch_add(predicted, Ordering::Relaxed);
    }

    fn json(&self) -> Value {
        let samples = self.samples.lock().map(|mut s| std::mem::take(&mut *s)).unwrap_or_default();
        let ms: Vec<f64> = samples.iter().map(|s| (s.0 as f64 * 10.0).round() / 10.0).collect();
        let mut v = json!({
            "busy_s": self.busy_us.load(Ordering::Relaxed) as f64 / 1e6,
            "calls": self.calls.load(Ordering::Relaxed),
            "units": self.units.load(Ordering::Relaxed),
            "max_ms": self.max_us.swap(0, Ordering::Relaxed) as f64 / 1e3,
            "prompt_tokens": self.prompt_tokens.load(Ordering::Relaxed),
            "cached_tokens": self.cached_tokens.load(Ordering::Relaxed),
            "predicted_tokens": self.predicted_tokens.load(Ordering::Relaxed),
            "samples_ms": ms,
        });
        // ASR: audio length of each call, to tell slow decoding from long segments
        if samples.iter().any(|s| s.1 > 0) {
            v["samples_units"] = json!(samples.iter().map(|s| s.1).collect::<Vec<_>>());
        }
        v
    }

    fn reset(&self) {
        for a in [&self.busy_us, &self.calls, &self.units, &self.max_us, &self.prompt_tokens, &self.cached_tokens, &self.predicted_tokens] {
            a.store(0, Ordering::Relaxed);
        }
        if let Ok(mut s) = self.samples.lock() {
            s.clear();
        }
    }
}

pub static ASR: Meter = Meter::new();
pub static DRAFT: Meter = Meter::new();
pub static LLM: Meter = Meter::new();
pub static SPEAKER: Meter = Meter::new();

/// Totals since the last `reset` (`max_ms` and `samples_ms`: calls since the previous
/// snapshot).
pub fn snapshot() -> Value {
    json!({"asr": ASR.json(), "draft": DRAFT.json(), "llm": LLM.json(), "speaker": SPEAKER.json()})
}

pub fn reset() {
    for m in [&ASR, &DRAFT, &LLM, &SPEAKER] {
        m.reset();
    }
}

/// Human-readable summary of a `snapshot` over `elapsed_s` seconds of wall time.
pub fn summarize(stats: &Value, elapsed_s: f64) -> String {
    let e = elapsed_s.max(1e-6);
    let mut out = format!("work over {:.0} s (share of wall time one component was busy):\n", elapsed_s);
    for (key, name) in [("asr", "speech recognition (CPU)"), ("draft", "Opus-MT draft (CPU)"), ("llm", "LLM translation (GPU)"), ("speaker", "speaker voice prints (CPU)")] {
        let m = &stats[key];
        let busy = m["busy_s"].as_f64().unwrap_or(0.0);
        let calls = m["calls"].as_u64().unwrap_or(0);
        out.push_str(&format!("  {name:28} {:5.1}%  {:6.1} calls/min", busy / e * 100.0, calls as f64 / e * 60.0));
        if key == "llm" && calls > 0 {
            let per = |k: &str| m[k].as_u64().unwrap_or(0) as f64 / calls as f64;
            out.push_str(&format!(
                "  prompt {:.0} tok/call evaluated, {:.0} reused from cache, {:.0} generated",
                per("prompt_tokens"),
                per("cached_tokens"),
                per("predicted_tokens")
            ));
        }
        out.push('\n');
    }
    out
}
