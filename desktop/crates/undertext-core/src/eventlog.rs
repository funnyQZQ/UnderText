//! JSONL event log (read by `undertext-cli report`, `soak-report` and `compare`).

use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::asr::AsrUpdate;
use crate::board::SentenceView;

pub struct EventLog {
    out: Mutex<Box<dyn Write + Send>>,
    t0: Instant,
}

impl EventLog {
    pub fn new(out: Box<dyn Write + Send>) -> Self {
        Self { out: Mutex::new(out), t0: Instant::now() }
    }

    pub fn write(&self, kind: &str, fields: Value) {
        let t = (self.t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
        let mut rec = Map::new();
        rec.insert("t".into(), json!(t));
        rec.insert("kind".into(), json!(kind));
        if let Value::Object(map) = fields {
            rec.extend(map);
        }
        let mut out = self.out.lock().unwrap();
        let _ = writeln!(out, "{}", Value::Object(rec));
        let _ = out.flush();
    }

    pub fn on_board(&self, kind: &str, view: &SentenceView) {
        let (zh, zh_unstable) = view.chinese_parts();
        self.write(
            kind,
            json!({
                "sid": view.sid, "en": view.source(), "zh": zh, "zh_unstable": zh_unstable,
                "seg": view.segment_id, "speaker": view.speaker,
            }),
        );
    }

    pub fn on_asr(&self, u: &AsrUpdate) {
        self.write(
            "asr",
            json!({"seg": u.segment_id, "committed": u.committed, "tentative": u.tentative, "final": u.is_final}),
        );
    }
}
