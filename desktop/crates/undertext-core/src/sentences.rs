//! Split the ASR segment stream into sentences, the unit of translation.

use serde::Serialize;

use crate::asr::AsrUpdate;
use crate::text::{best_split_point, is_sentence_end_word, join_words, split_words};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SentenceEvent {
    Progress { sid: usize, committed: String, tentative: String, segment_id: usize },
    Final { sid: usize, text: String, segment_id: usize },
}

/// Equality ignores `segment_id`.
impl PartialEq for SentenceEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Progress { sid: a, committed: b, tentative: c, .. },
                Self::Progress { sid: x, committed: y, tentative: z, .. },
            ) => a == x && b == y && c == z,
            (Self::Final { sid: a, text: b, .. }, Self::Final { sid: x, text: y, .. }) => a == x && b == y,
            _ => false,
        }
    }
}

impl SentenceEvent {
    pub fn segment_id(&self) -> usize {
        match self {
            Self::Progress { segment_id, .. } | Self::Final { segment_id, .. } => *segment_id,
        }
    }
}

/// Cuts committed ASR text into sentences.
///
/// A sentence closes when (a) a committed word ends with sentence punctuation and more
/// committed words follow it or the segment ends, (b) it grows past `max_words` (cut at a
/// clause boundary), or (c) the ASR segment ends.
pub struct SentenceTracker {
    max_words: usize,
    next_sid: usize,
    segment_id: Option<usize>,
    consumed: usize,
    last_progress: Option<SentenceEvent>,
}

impl SentenceTracker {
    pub fn new(max_words: usize) -> Self {
        Self { max_words, next_sid: 0, segment_id: None, consumed: 0, last_progress: None }
    }

    pub fn current_sid(&self) -> usize {
        self.next_sid
    }

    pub fn update(&mut self, u: &AsrUpdate) -> Vec<SentenceEvent> {
        let mut events = Vec::new();
        if let Some(seg) = self.segment_id {
            if u.segment_id != seg {
                // previous segment vanished without a final update: drop its open sentence
                self.consumed = 0;
                self.last_progress = None;
            }
        }
        self.segment_id = Some(u.segment_id);

        let committed = split_words(&u.committed);
        let mut pending: &[String] = committed.get(self.consumed..).unwrap_or(&[]);

        while !pending.is_empty() {
            let mut cut = 0;
            let limit = if u.is_final { pending.len() } else { pending.len() - 1 };
            // a sentence end is trusted only when a later committed word confirms it
            for i in 0..limit.min(self.max_words) {
                if is_sentence_end_word(&pending[i]) {
                    cut = i + 1;
                    break;
                }
            }
            if cut == 0 && pending.len() > self.max_words {
                cut = best_split_point(pending, self.max_words);
            }
            if cut == 0 {
                break;
            }
            events.push(self.close(join_words(&pending[..cut]), u.segment_id));
            self.consumed += cut;
            pending = &pending[cut..];
        }

        if u.is_final {
            if !pending.is_empty() {
                events.push(self.close(join_words(pending), u.segment_id));
            }
            self.consumed = 0;
            self.segment_id = None;
            self.last_progress = None;
            return events;
        }

        let progress = SentenceEvent::Progress {
            sid: self.next_sid,
            committed: join_words(pending),
            tentative: u.tentative.clone(),
            segment_id: u.segment_id,
        };
        let has_text = match &progress {
            SentenceEvent::Progress { committed, tentative, .. } => !committed.is_empty() || !tentative.is_empty(),
            _ => false,
        };
        if has_text && self.last_progress.as_ref() != Some(&progress) {
            self.last_progress = Some(progress.clone());
            events.push(progress);
        }
        events
    }

    fn close(&mut self, text: String, segment_id: usize) -> SentenceEvent {
        let event = SentenceEvent::Final { sid: self.next_sid, text, segment_id };
        self.next_sid += 1;
        self.last_progress = None;
        event
    }
}
