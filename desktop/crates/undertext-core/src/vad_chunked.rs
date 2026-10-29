//! Pseudo-streaming ASR: VAD + periodic re-decoding with an offline model.
//!
//! While speech is ongoing the growing buffer is re-decoded every `decode_interval_s`;
//! the stabilizer decides what is committed. When the VAD closes the segment it is
//! decoded once more and emitted as final.

use std::collections::VecDeque;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::asr::{AsrBackend, AsrUpdate, OfflineRecognizer};
use crate::stabilizer::HypothesisStabilizer;
use crate::vad::Vad;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VadChunkedConfig {
    pub decode_interval_s: f64,
    pub max_segment_s: f64,
    pub agreement_n: usize,
    /// Audio kept from before the VAD fires (the VAD reacts ~0.2 s late).
    pub preroll_s: f64,
    /// First words of a segment stay tentative until it has this much audio.
    pub min_commit_s: f64,
    /// Re-decode long segments less often (every max(decode_interval, length / 8)), so
    /// the work grows linearly instead of quadratically with the segment length.
    pub adaptive_interval: bool,
}

impl Default for VadChunkedConfig {
    fn default() -> Self {
        Self { decode_interval_s: 0.5, max_segment_s: 20.0, agreement_n: 2, preroll_s: 0.4, min_commit_s: 1.0, adaptive_interval: false }
    }
}

pub struct VadChunkedAsr {
    vad: Box<dyn Vad>,
    recognizer: Box<dyn OfflineRecognizer>,
    decode_interval: usize,
    min_commit: usize,
    max_segment: usize,
    adaptive: bool,
    /// Audio (samples) worth 1.5x the wall time the last decode took: the next re-decode
    /// waits at least this long, so decoding never needs more than ~2/3 of real time and
    /// a long segment on a slow or throttled machine can't build up a backlog.
    min_gap_from_cost: usize,
    preroll_windows: usize,
    stabilizer: HypothesisStabilizer,
    leftover: Vec<f32>,
    preroll: VecDeque<Vec<f32>>,
    speech: Vec<f32>,
    since_decode: usize,
    in_speech: bool,
    segment_id: usize,
    /// Samples seen so far (stream clock) and where the current segment's speech began.
    processed: usize,
    seg_start: usize,
}

impl VadChunkedAsr {
    pub fn new(vad: Box<dyn Vad>, recognizer: Box<dyn OfflineRecognizer>, cfg: &VadChunkedConfig) -> Self {
        let sr = 16000.0;
        let window = vad.window_size();
        let preroll_samples = (cfg.preroll_s * sr) as usize;
        Self {
            decode_interval: (cfg.decode_interval_s * sr) as usize,
            min_commit: (cfg.min_commit_s * sr) as usize,
            max_segment: (cfg.max_segment_s * sr) as usize,
            adaptive: cfg.adaptive_interval,
            min_gap_from_cost: 0,
            preroll_windows: preroll_samples.div_ceil(window).max(1),
            stabilizer: HypothesisStabilizer::new(cfg.agreement_n),
            vad,
            recognizer,
            leftover: Vec::new(),
            preroll: VecDeque::new(),
            speech: Vec::new(),
            since_decode: 0,
            in_speech: false,
            segment_id: 0,
            processed: 0,
            seg_start: 0,
        }
    }

    fn secs(samples: usize) -> f64 {
        samples as f64 / 16000.0
    }

    fn on_window(&mut self, window: &[f32]) -> Vec<AsrUpdate> {
        let mut updates = Vec::new();
        let window_start = self.processed;
        self.processed += window.len();
        self.vad.accept_waveform(window);
        let speaking = self.vad.is_speech();

        if speaking || self.in_speech {
            if !self.in_speech {
                self.in_speech = true;
                self.seg_start = window_start;
                self.speech = self.preroll.iter().flatten().copied().collect();
                self.since_decode = 0;
            }
            self.speech.extend_from_slice(window);
            self.since_decode += window.len();
        } else {
            self.preroll.push_back(window.to_vec());
            if self.preroll.len() > self.preroll_windows {
                self.preroll.pop_front();
            }
        }

        let segments = self.vad.pop_segments();
        if !segments.is_empty() {
            for seg in segments {
                updates.push(self.finalize(seg, true));
            }
            return updates;
        }

        if self.in_speech && self.speech.len() >= self.max_segment {
            // bound re-decoding cost: close the segment ourselves and restart the VAD
            let seg = std::mem::take(&mut self.speech);
            updates.push(self.finalize(seg, false));
            self.vad.reset();
            return updates;
        }

        let mut interval = if self.adaptive { self.decode_interval.max(self.speech.len() / 8) } else { self.decode_interval };
        interval = interval.max(self.min_gap_from_cost);
        if self.in_speech && self.since_decode >= interval {
            self.since_decode = 0;
            let t0 = std::time::Instant::now();
            let text = self.recognizer.transcribe(&self.speech);
            self.min_gap_from_cost = (t0.elapsed().as_secs_f64() * 1.5 * 16000.0) as usize;
            let allow_commit = self.speech.len() >= self.min_commit;
            if let Some((committed, tentative)) = self.stabilizer.update(&text, allow_commit) {
                if !committed.is_empty() || !tentative.is_empty() {
                    let (start, now) = (Self::secs(self.seg_start), Self::secs(self.processed));
                    updates.push(AsrUpdate::partial(self.segment_id, committed, tentative).with_time(start, now));
                }
            }
        }
        updates
    }

    /// `after_silence`: the VAD closed the segment (speech ended `min_silence_s` ago);
    /// false for forced cuts (max length, end of stream).
    fn finalize(&mut self, segment: Vec<f32>, after_silence: bool) -> AsrUpdate {
        self.min_gap_from_cost = 0; // the next segment starts short and cheap
        let text = self.recognizer.transcribe(&segment);
        let final_text = self.stabilizer.finalize(&text);
        let start = Self::secs(self.seg_start);
        let silence = if after_silence { self.vad.min_silence_s() } else { 0.0 };
        let end = (Self::secs(self.processed) - silence).max(start);
        let update = AsrUpdate::final_(self.segment_id, final_text, Some(Arc::new(segment))).with_time(start, end);
        self.segment_id += 1;
        self.in_speech = false;
        self.speech.clear();
        self.since_decode = 0;
        self.preroll.clear();
        update
    }
}

impl AsrBackend for VadChunkedAsr {
    fn accept(&mut self, samples: &[f32]) -> Vec<AsrUpdate> {
        let mut data = std::mem::take(&mut self.leftover);
        data.extend_from_slice(samples);
        let w = self.vad.window_size();
        let n_full = data.len() / w;
        self.leftover = data[n_full * w..].to_vec();
        let mut updates = Vec::new();
        for i in 0..n_full {
            updates.extend(self.on_window(&data[i * w..(i + 1) * w]));
        }
        updates
    }

    fn flush(&mut self) -> Vec<AsrUpdate> {
        let mut updates = Vec::new();
        if !self.leftover.is_empty() {
            let mut window = std::mem::take(&mut self.leftover);
            window.resize(self.vad.window_size(), 0.0);
            updates.extend(self.on_window(&window));
        }
        self.vad.flush();
        let segments = self.vad.pop_segments();
        let had_segments = !segments.is_empty();
        for seg in segments {
            updates.push(self.finalize(seg, false));
        }
        if !had_segments && self.in_speech && !self.speech.is_empty() {
            let seg = std::mem::take(&mut self.speech);
            updates.push(self.finalize(seg, false));
        }
        updates
    }
}
