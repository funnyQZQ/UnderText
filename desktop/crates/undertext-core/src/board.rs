//! Thread-safe subtitle state shared by the pipeline and the UI.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::Serialize;

use crate::paragraphs::{group_paragraphs, Paragraph, ParagraphConfig};

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SentenceView {
    pub sid: usize,
    pub src_committed: String,
    pub src_tentative: String,
    /// Stable Chinese (only grows, until the final translation).
    pub zh_committed: String,
    /// Latest LLM output beyond `zh_committed` (may change).
    pub zh_pending: String,
    /// Fast-model translation of the whole current sentence.
    pub zh_draft: String,
    pub draft_version: i64,
    /// English is complete.
    pub is_final: bool,
    /// Final Chinese is in `zh_committed`.
    pub translated: bool,
    pub segment_id: Option<usize>,
    /// Diarization label, known once the segment ends.
    pub speaker: Option<String>,
}

impl SentenceView {
    pub fn new(sid: usize) -> Self {
        Self { sid, draft_version: -1, ..Default::default() }
    }

    pub fn source(&self) -> String {
        match (self.src_committed.is_empty(), self.src_tentative.is_empty()) {
            (false, false) => format!("{} {}", self.src_committed, self.src_tentative),
            (false, true) => self.src_committed.clone(),
            _ => self.src_tentative.clone(),
        }
    }

    /// (stable, unstable) Chinese to display.
    ///
    /// The draft stays on screen until the LLM has confirmed some Chinese (its first,
    /// unconfirmed outputs cover less of the sentence than the draft, so switching early
    /// makes the line shrink). Without a draft the LLM's pending text is shown at once.
    pub fn chinese_parts(&self) -> (String, String) {
        if self.translated {
            return (self.zh_committed.clone(), String::new());
        }
        if !self.zh_committed.is_empty() {
            return (self.zh_committed.clone(), self.zh_pending.clone());
        }
        if !self.zh_draft.is_empty() {
            return (String::new(), self.zh_draft.clone());
        }
        (String::new(), self.zh_pending.clone())
    }

    /// Compact form sent to the UI.
    pub fn display(&self) -> SentenceDisplay {
        let (zh_stable, zh_unstable) = self.chinese_parts();
        SentenceDisplay {
            sid: self.sid,
            speaker: self.speaker.clone(),
            en_stable: self.src_committed.clone(),
            en_unstable: self.src_tentative.clone(),
            zh_stable,
            zh_unstable,
            is_final: self.is_final,
            translated: self.translated,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SentenceDisplay {
    pub sid: usize,
    pub speaker: Option<String>,
    pub en_stable: String,
    pub en_unstable: String,
    pub zh_stable: String,
    pub zh_unstable: String,
    pub is_final: bool,
    pub translated: bool,
}

/// (start, end, label): a speaker over a fraction of an ASR segment.
pub type SpanLabel = (f32, f32, Option<String>);

/// Called (with the board locked) after every change; must not call back into the board.
pub type Listener = Box<dyn Fn(&str, &SentenceView) + Send + Sync>;

struct Inner {
    sentences: BTreeMap<usize, SentenceView>,
    /// segment_id -> speakers over fractions of the segment: (start, end, label)
    segment_speaker: BTreeMap<usize, Vec<SpanLabel>>,
    /// segment_id -> (speech start, speech end) in stream seconds; kept for the whole run
    segment_time: BTreeMap<usize, (f64, f64)>,
    /// Sentences that no longer fit in `sentences`, kept for `transcript`.
    archive: Vec<SentenceView>,
}

pub struct SubtitleBoard {
    inner: Mutex<Inner>,
    max: usize,
    listener: Option<Listener>,
    version: AtomicU64,
}

impl SubtitleBoard {
    pub fn new(max_sentences: usize, listener: Option<Listener>) -> Self {
        Self {
            inner: Mutex::new(Inner {
                sentences: BTreeMap::new(),
                segment_speaker: BTreeMap::new(),
                segment_time: BTreeMap::new(),
                archive: Vec::new(),
            }),
            max: max_sentences,
            listener,
            version: AtomicU64::new(0),
        }
    }

    /// Bumped on every change; renderers can poll it cheaply.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    fn with_view<R>(&self, sid: usize, f: impl FnOnce(&mut SentenceView, &BTreeMap<usize, Vec<SpanLabel>>) -> Option<(&'static str, R)>) -> Option<R> {
        let mut inner = self.inner.lock().unwrap();
        let Inner { sentences, segment_speaker, archive, .. } = &mut *inner;
        if !sentences.contains_key(&sid) {
            sentences.insert(sid, SentenceView::new(sid));
            while sentences.len() > self.max {
                if let Some((_, v)) = sentences.pop_first() {
                    archive.push(v);
                }
            }
        }
        let view = sentences.get_mut(&sid)?;
        let (kind, r) = f(view, segment_speaker)?;
        self.version.fetch_add(1, Ordering::SeqCst);
        if let Some(listener) = &self.listener {
            listener(kind, view);
        }
        Some(r)
    }

    fn bind_segment(view: &mut SentenceView, segment_id: Option<usize>, speakers: &BTreeMap<usize, Vec<SpanLabel>>) {
        if let Some(seg) = segment_id {
            view.segment_id = Some(seg);
            if let Some(spans) = speakers.get(&seg) {
                // one speaker for the whole segment; several are placed by `set_segment_speakers`
                if let [(_, _, label)] = spans.as_slice() {
                    view.speaker = label.clone();
                }
            }
        }
    }

    pub fn set_source(&self, sid: usize, committed: &str, tentative: &str, segment_id: Option<usize>) {
        self.with_view(sid, |v, sp| {
            if v.is_final {
                return None;
            }
            v.src_committed = committed.into();
            v.src_tentative = tentative.into();
            Self::bind_segment(v, segment_id, sp);
            Some(("source", ()))
        });
    }

    pub fn finalize_source(&self, sid: usize, text: &str, segment_id: Option<usize>) {
        self.with_view(sid, |v, sp| {
            v.src_committed = text.into();
            v.src_tentative.clear();
            v.is_final = true;
            Self::bind_segment(v, segment_id, sp);
            Some(("source_final", ()))
        });
    }

    /// Speakers over parts of a segment (turn changes without a pause). Each sentence of
    /// the segment gets the speaker at its middle, positions estimated from text length.
    pub fn set_segment_speakers(&self, segment_id: usize, spans: Vec<SpanLabel>) {
        let mut inner = self.inner.lock().unwrap();
        let mine: Vec<usize> =
            inner.sentences.values().filter(|v| v.segment_id == Some(segment_id)).map(|v| v.sid).collect();
        let lens: Vec<f32> = mine
            .iter()
            .map(|sid| inner.sentences[sid].source().chars().count().max(1) as f32)
            .collect();
        let total: f32 = lens.iter().sum();
        let mut before = 0.0;
        let mut labels = Vec::new();
        for len in &lens {
            let mid = (before + len / 2.0) / total.max(1.0);
            before += len;
            let span = spans.iter().find(|(a, b, _)| *a <= mid && mid < *b).or(spans.last());
            labels.push(span.and_then(|s| s.2.clone()));
        }
        inner.segment_speaker.insert(segment_id, spans);
        while inner.segment_speaker.len() > self.max {
            inner.segment_speaker.pop_first();
        }
        for (sid, label) in mine.into_iter().zip(labels) {
            let view = inner.sentences.get_mut(&sid).unwrap();
            if view.speaker != label {
                view.speaker = label;
                self.version.fetch_add(1, Ordering::SeqCst);
                if let Some(listener) = &self.listener {
                    listener("speaker", view);
                }
            }
        }
    }

    /// Record where a segment's speech starts / ends (stream seconds); used to group
    /// sentences into paragraphs. Does not notify listeners.
    pub fn set_segment_time(&self, segment_id: usize, start_s: f64, end_s: f64) {
        let mut inner = self.inner.lock().unwrap();
        inner.segment_time.insert(segment_id, (start_s, end_s));
    }

    /// The last `last` paragraphs (sentences grouped by speaker and pauses).
    pub fn paragraphs(&self, cfg: &ParagraphConfig, last: usize) -> Vec<Paragraph> {
        let inner = self.inner.lock().unwrap();
        let views: Vec<&SentenceView> = inner.sentences.values().collect();
        let mut all = group_paragraphs(&views, &inner.segment_time, cfg);
        let n = all.len().saturating_sub(last);
        all.drain(..n);
        all
    }

    pub fn set_draft(&self, sid: usize, text: &str, version: i64) -> bool {
        self.with_view(sid, |v, _| {
            if v.translated || version <= v.draft_version {
                return None;
            }
            v.zh_draft = text.into();
            v.draft_version = version;
            Some(("draft", ()))
        })
        .is_some()
    }

    pub fn set_commit(&self, sid: usize, committed: &str, pending: &str) {
        self.with_view(sid, |v, _| {
            if v.translated {
                return None;
            }
            v.zh_committed = committed.into();
            v.zh_pending = pending.into();
            Some(("commit", ()))
        });
    }

    pub fn set_final_translation(&self, sid: usize, text: &str) {
        self.with_view(sid, |v, _| {
            v.zh_committed = text.into();
            v.zh_pending.clear();
            v.translated = true;
            Some(("final", ()))
        });
    }

    /// Everything said since the run started, as paragraphs with their start time
    /// (stream seconds), for saving to a file.
    pub fn transcript(&self, cfg: &ParagraphConfig) -> Vec<(Option<f64>, Paragraph)> {
        let inner = self.inner.lock().unwrap();
        let views: Vec<&SentenceView> = inner.archive.iter().chain(inner.sentences.values()).collect();
        let segment_of: BTreeMap<usize, Option<usize>> = views.iter().map(|v| (v.sid, v.segment_id)).collect();
        group_paragraphs(&views, &inner.segment_time, cfg)
            .into_iter()
            .map(|p| {
                let start = p
                    .sentences
                    .first()
                    .and_then(|s| segment_of.get(&s.sid).copied().flatten())
                    .and_then(|seg| inner.segment_time.get(&seg).map(|t| t.0));
                (start, p)
            })
            .collect()
    }

    /// Where the speech segment holding sentence `sid` started (stream seconds).
    pub fn sentence_start(&self, sid: usize) -> Option<f64> {
        let inner = self.inner.lock().unwrap();
        let view = inner.sentences.get(&sid).or_else(|| inner.archive.iter().find(|v| v.sid == sid))?;
        inner.segment_time.get(&view.segment_id?).map(|t| t.0)
    }

    pub fn snapshot(&self, last: Option<usize>) -> Vec<SentenceView> {
        let inner = self.inner.lock().unwrap();
        let all: Vec<SentenceView> = inner.sentences.values().cloned().collect();
        match last {
            Some(n) if n < all.len() => all[all.len() - n..].to_vec(),
            _ => all,
        }
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.sentences.clear();
        inner.segment_speaker.clear();
        inner.segment_time.clear();
        inner.archive.clear();
        self.version.fetch_add(1, Ordering::SeqCst);
    }
}

impl Default for SubtitleBoard {
    fn default() -> Self {
        Self::new(200, None)
    }
}
