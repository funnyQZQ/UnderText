//! Group sentences into paragraphs for display: a new paragraph starts when the speaker
//! changes or after a long enough pause, so short fragments ("And then") flow on with
//! the sentence around them instead of getting a line of their own.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::board::{SentenceDisplay, SentenceView};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParagraphConfig {
    /// Silence between two ASR segments that starts a new paragraph (seconds).
    pub pause_s: f64,
    /// A change of diarization label starts a new paragraph.
    pub split_on_speaker: bool,
}

impl Default for ParagraphConfig {
    fn default() -> Self {
        Self { pause_s: 1.2, split_on_speaker: true }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Paragraph {
    pub speaker: Option<String>,
    pub sentences: Vec<SentenceDisplay>,
    /// Every sentence is final and has its final translation.
    pub done: bool,
}

fn has_content(v: &SentenceView) -> bool {
    !(v.src_committed.is_empty() && v.src_tentative.is_empty())
}

/// `views` in sentence order; `times`: segment_id -> (speech start, speech end).
pub fn group_paragraphs(views: &[&SentenceView], times: &BTreeMap<usize, (f64, f64)>, cfg: &ParagraphConfig) -> Vec<Paragraph> {
    let mut out: Vec<Paragraph> = Vec::new();
    let mut prev: Option<&SentenceView> = None;
    for &v in views {
        if !has_content(v) {
            continue;
        }
        let new_paragraph = match prev {
            None => true,
            Some(p) => {
                let speaker_change = cfg.split_on_speaker
                    && matches!((&p.speaker, &v.speaker), (Some(a), Some(b)) if a != b);
                let long_pause = match (p.segment_id, v.segment_id) {
                    (Some(a), Some(b)) if a != b => match (times.get(&a), times.get(&b)) {
                        (Some(&(_, end)), Some(&(start, _))) => start - end >= cfg.pause_s,
                        _ => false,
                    },
                    _ => false,
                };
                speaker_change || long_pause
            }
        };
        if new_paragraph {
            out.push(Paragraph { speaker: v.speaker.clone(), sentences: Vec::new(), done: true });
        }
        let para = out.last_mut().expect("paragraph exists");
        if para.speaker.is_none() {
            para.speaker = v.speaker.clone();
        }
        para.done &= v.is_final && v.translated;
        para.sentences.push(v.display());
        prev = Some(v);
    }
    out
}
