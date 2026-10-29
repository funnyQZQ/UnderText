//! Audio -> ASR -> sentences -> translation controller -> board, plus diarization.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;

use crate::asr::{AsrBackend, AsrUpdate};
use crate::board::SubtitleBoard;
use crate::controller::TranslationController;
use crate::diarization::Diarizer;
use crate::eventlog::EventLog;
use crate::metrics;
use crate::sentences::SentenceTracker;
use crate::workers::{InlineWorker, Job, JobKey, JobTag, Worker};

pub struct Engine {
    asr: Box<dyn AsrBackend>,
    tracker: SentenceTracker,
    controller: TranslationController,
    log: Option<Arc<EventLog>>,
    diarizer: Option<Arc<Diarizer>>,
    diarize_worker: Arc<dyn Worker>,
    stop: Arc<AtomicBool>,
    started: Instant,
    last_stats: Instant,
    stats_every: Duration,
    /// Real-time bookkeeping: when audio started arriving, how much has been fed, and
    /// the worst backlog since the last stats record.
    first_audio: Option<Instant>,
    audio_samples: u64,
    max_lag_s: f64,
    /// The last few seconds of audio, for provisional speaker labels of the open segment.
    recent: VecDeque<f32>,
    /// (segment, segment time of its last provisional label)
    last_peek: Option<(usize, f64)>,
}

/// Audio kept for provisional speaker labels, and how much each one looks at / how often.
const RECENT: usize = 5 * 16000;
const PEEK_WINDOW_S: f64 = 1.5;
const PEEK_EVERY_S: f64 = 1.5;

impl Engine {
    pub fn new(asr: Box<dyn AsrBackend>, tracker: SentenceTracker, controller: TranslationController, log: Option<Arc<EventLog>>) -> Self {
        Self {
            asr,
            tracker,
            controller,
            log,
            diarizer: None,
            diarize_worker: Arc::new(InlineWorker),
            stop: Arc::new(AtomicBool::new(false)),
            started: { metrics::reset(); Instant::now() },
            last_stats: Instant::now(),
            stats_every: Duration::from_secs(30),
            first_audio: None,
            audio_samples: 0,
            max_lag_s: 0.0,
            recent: VecDeque::new(),
            last_peek: None,
        }
    }

    pub fn with_diarizer(mut self, diarizer: Arc<Diarizer>, worker: Arc<dyn Worker>) -> Self {
        self.diarizer = Some(diarizer);
        self.diarize_worker = worker;
        self
    }

    /// How often a `stats` record goes to the event log (default 30 s).
    pub fn set_stats_every(&mut self, every: Duration) {
        self.stats_every = every;
    }

    pub fn board(&self) -> &Arc<SubtitleBoard> {
        self.controller.board()
    }

    /// Flag that makes `run` return after the current chunk.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    fn dispatch(&mut self, updates: Vec<AsrUpdate>) {
        for u in updates {
            if let Some(log) = &self.log {
                log.on_asr(&u);
            }
            self.board().set_segment_time(u.segment_id, u.start_s, u.end_s);
            for event in self.tracker.update(&u) {
                self.controller.handle(&event);
            }
            if let (true, Some(d), Some(samples)) = (u.is_final, &self.diarizer, &u.samples) {
                let (d, samples, board, log, seg) = (d.clone(), samples.clone(), self.board().clone(), self.log.clone(), u.segment_id);
                self.diarize_worker.submit(Job::new(None, JobTag::Other, move || {
                    let updates = d.label_segment(seg, &samples);
                    if let Some(log) = log {
                        let spans = &updates[0].1;
                        let parts: Vec<_> = spans
                            .iter()
                            .map(|s| json!({"start": s.start, "end": s.end, "speaker": s.label,
                                            "similarity": (s.similarity * 1000.0).round() / 1000.0, "new": s.new}))
                            .collect();
                        let first = &spans[0];
                        log.write(
                            "diarize",
                            json!({"seg": seg, "speaker": first.label, "similarity": (first.similarity * 1000.0).round() / 1000.0,
                                   "new": first.new, "duration": (samples.len() as f64 / 160.0).round() / 100.0, "spans": parts,
                                   "relabeled": updates[1..].iter().map(|u| u.0).collect::<Vec<_>>()}),
                        );
                    }
                    for (seg, spans) in updates {
                        board.set_segment_speakers(seg, spans.into_iter().map(|s| (s.start, s.end, s.label)).collect());
                    }
                }));
            } else if !u.is_final {
                self.peek_speaker(&u);
            }
        }
    }

    /// While a segment is being spoken, label it from its latest audio every
    /// `PEEK_EVERY_S`, so the speaker shows up before the segment ends.
    fn peek_speaker(&mut self, u: &AsrUpdate) {
        let Some(d) = &self.diarizer else { return };
        let elapsed = u.end_s - u.start_s;
        if elapsed < PEEK_WINDOW_S {
            return;
        }
        if let Some((seg, at)) = self.last_peek {
            if seg == u.segment_id && elapsed - at < PEEK_EVERY_S {
                return;
            }
        }
        // u.end_s is in stream time; `recent` ends at `audio_samples`
        let end = (u.end_s * 16000.0) as u64;
        let len = (PEEK_WINDOW_S * 16000.0) as u64;
        let first = self.audio_samples - self.recent.len() as u64;
        if end > self.audio_samples || end < first + len {
            return;
        }
        let from = (end - len - first) as usize;
        let window: Vec<f32> = self.recent.range(from..from + len as usize).copied().collect();
        self.last_peek = Some((u.segment_id, elapsed));
        let (d, board, log, seg) = (d.clone(), self.board().clone(), self.log.clone(), u.segment_id);
        self.diarize_worker.submit(Job::new(Some(JobKey::Peek), JobTag::Other, move || {
            if let Some(spans) = d.peek(seg, elapsed, &window) {
                if let Some(log) = log {
                    let labels: Vec<_> = spans.iter().map(|s| s.label.clone()).collect();
                    log.write("speaker_peek", json!({"seg": seg, "at": (elapsed * 100.0).round() / 100.0, "speakers": labels}));
                }
                board.set_segment_speakers(seg, spans.into_iter().map(|s| (s.start, s.end, s.label)).collect());
            }
        }));
    }

    /// Seconds the processing is behind a live source (wall time since the first chunk
    /// minus audio fed). Stays near 0 when the pipeline keeps up with real time.
    fn lag_s(&self) -> f64 {
        match self.first_audio {
            Some(t) => (t.elapsed().as_secs_f64() - self.audio_samples as f64 / 16000.0).max(0.0),
            None => 0.0,
        }
    }

    pub fn feed(&mut self, samples: &[f32]) {
        self.first_audio.get_or_insert_with(Instant::now);
        self.audio_samples += samples.len() as u64;
        if self.diarizer.is_some() {
            self.recent.extend(samples);
            let extra = self.recent.len().saturating_sub(RECENT);
            self.recent.drain(..extra);
        }
        let updates = self.asr.accept(samples);
        self.dispatch(updates);
        self.max_lag_s = self.max_lag_s.max(self.lag_s());
        if self.last_stats.elapsed() >= self.stats_every {
            self.write_stats();
        }
    }

    /// Cumulative compute time per component since the engine started (event log `stats`).
    fn write_stats(&mut self) {
        self.last_stats = Instant::now();
        let max_lag = self.max_lag_s;
        self.max_lag_s = 0.0;
        if let Some(log) = &self.log {
            let mut v = metrics::snapshot();
            let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
            v["elapsed_s"] = json!(r3(self.started.elapsed().as_secs_f64()));
            v["audio_s"] = json!(r3(self.audio_samples as f64 / 16000.0));
            v["lag_s"] = json!(r3(self.lag_s()));
            v["max_lag_s"] = json!(r3(max_lag));
            log.write("stats", v);
        }
    }

    /// Human-readable compute summary since the engine started.
    pub fn stats_summary(&self) -> String {
        metrics::summarize(&metrics::snapshot(), self.started.elapsed().as_secs_f64())
    }

    pub fn finish(&mut self, timeout: Duration) {
        let updates = self.asr.flush();
        self.dispatch(updates);
        self.diarize_worker.wait_idle(Some(timeout));
        self.controller.wait_idle(Some(timeout));
        self.write_stats();
    }

    pub fn run(&mut self, source: impl IntoIterator<Item = Vec<f32>>) {
        for chunk in source {
            if self.stop.load(Ordering::SeqCst) {
                break;
            }
            self.feed(&chunk);
        }
        self.finish(Duration::from_secs(120));
    }

    pub fn shutdown(&self) {
        self.diarize_worker.stop();
        self.controller.stop();
    }
}
