//! Common ASR types.

use std::sync::Arc;

/// State of the current ASR segment (one VAD-delimited utterance).
///
/// `committed` only ever grows within a segment; `tentative` may change freely.
/// When `is_final` is true the whole segment text is in `committed`.
#[derive(Debug, Clone)]
pub struct AsrUpdate {
    pub segment_id: usize,
    pub committed: String,
    pub tentative: String,
    pub is_final: bool,
    /// Audio of the whole segment, set on final updates (used for diarization).
    pub samples: Option<Arc<Vec<f32>>>,
    /// Stream time (s) where the segment's speech started / ended so far. On a final
    /// update `end_s` is the estimated end of speech (the VAD's closing silence removed).
    /// Used to group sentences into paragraphs; not part of equality.
    pub start_s: f64,
    pub end_s: f64,
}

impl AsrUpdate {
    pub fn partial(segment_id: usize, committed: impl Into<String>, tentative: impl Into<String>) -> Self {
        Self { segment_id, committed: committed.into(), tentative: tentative.into(), is_final: false, samples: None, start_s: 0.0, end_s: 0.0 }
    }

    pub fn final_(segment_id: usize, text: impl Into<String>, samples: Option<Arc<Vec<f32>>>) -> Self {
        Self { segment_id, committed: text.into(), tentative: String::new(), is_final: true, samples, start_s: 0.0, end_s: 0.0 }
    }

    pub fn with_time(mut self, start_s: f64, end_s: f64) -> Self {
        self.start_s = start_s;
        self.end_s = end_s;
        self
    }

    pub fn text(&self) -> String {
        match (self.committed.is_empty(), self.tentative.is_empty()) {
            (false, false) => format!("{} {}", self.committed, self.tentative),
            (false, true) => self.committed.clone(),
            _ => self.tentative.clone(),
        }
    }
}

impl PartialEq for AsrUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.segment_id == other.segment_id
            && self.committed == other.committed
            && self.tentative == other.tentative
            && self.is_final == other.is_final
    }
}

/// Streaming speech recognizer producing [`AsrUpdate`]s from 16 kHz mono audio.
pub trait AsrBackend: Send {
    fn accept(&mut self, samples: &[f32]) -> Vec<AsrUpdate>;
    /// Finish the current segment (end of stream).
    fn flush(&mut self) -> Vec<AsrUpdate>;
}

/// Non-streaming model that transcribes a whole buffer (e.g. Parakeet-TDT).
pub trait OfflineRecognizer: Send {
    fn transcribe(&self, samples: &[f32]) -> String;
}
