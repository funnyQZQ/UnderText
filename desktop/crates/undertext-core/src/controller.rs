//! Translation scheduling: draft / incremental commit / final, per sentence.
//!
//! Draft  : fast model, whole current sentence (committed + tentative English), latest wins.
//! Commit : LLM continues the already-committed Chinese for the committed English;
//!          LocalAgreement over Chinese characters decides what becomes stable.
//! Final  : when the English sentence closes, the LLM finishes (or re-translates) it.
//!
//! Commit and final jobs share one worker so they are strictly ordered per sentence.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::board::SubtitleBoard;
use crate::local_agreement::LocalAgreement;
use crate::sentences::SentenceEvent;
use crate::text::{is_clause_end_word, split_words};
use crate::translate::{ContextPair, IncrementalTranslator, Translator};
use crate::workers::{InlineWorker, Job, JobKey, JobTag, Worker};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FinalMode {
    /// Keep the committed Chinese and let the LLM finish the sentence.
    #[default]
    Continue,
    /// Re-translate the whole sentence (better quality, visible Chinese may change).
    Refine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ControllerConfig {
    /// New committed English words before an LLM call.
    pub commit_min_new_words: usize,
    /// Also call the LLM at , ; : boundaries.
    pub commit_on_clause: bool,
    /// LocalAgreement-n over Chinese characters.
    pub commit_agreement: usize,
    pub final_mode: FinalMode,
    /// Previous sentence pairs given to the LLM.
    pub context_sentences: usize,
    /// Stop drafting a sentence once the LLM has committed Chinese for it (the board
    /// shows the LLM text from then on, so further drafts are never seen).
    pub skip_hidden_drafts: bool,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self { commit_min_new_words: 3, commit_on_clause: true, commit_agreement: 2, final_mode: FinalMode::Continue, context_sentences: 3, skip_hidden_drafts: false }
    }
}

struct SentenceState {
    agreement: LocalAgreement<char>,
    zh_committed: String,
    llm_words: usize,
    draft_version: i64,
    last_draft_source: String,
    final_requested: bool,
}

struct Inner {
    board: Arc<SubtitleBoard>,
    draft: Option<Arc<dyn Translator>>,
    llm: Option<Arc<dyn IncrementalTranslator>>,
    cfg: ControllerConfig,
    states: Mutex<HashMap<usize, SentenceState>>,
    history: Mutex<VecDeque<ContextPair>>,
}

impl Inner {
    fn with_state<R>(&self, sid: usize, f: impl FnOnce(&mut SentenceState) -> R) -> R {
        let mut states = self.states.lock().unwrap();
        if !states.contains_key(&sid) {
            states.insert(
                sid,
                SentenceState {
                    agreement: LocalAgreement::new(self.cfg.commit_agreement),
                    zh_committed: String::new(),
                    llm_words: 0,
                    draft_version: 0,
                    last_draft_source: String::new(),
                    final_requested: false,
                },
            );
            // forget states of long-finished sentences
            if sid > 20 {
                states.retain(|k, _| *k >= sid - 20);
            }
        }
        f(states.get_mut(&sid).unwrap())
    }

    fn context(&self) -> Vec<ContextPair> {
        if self.cfg.context_sentences == 0 {
            return Vec::new();
        }
        let h = self.history.lock().unwrap();
        let skip = h.len().saturating_sub(self.cfg.context_sentences);
        h.iter().skip(skip).cloned().collect()
    }

    fn run_draft(&self, sid: usize, source: &str, version: i64) {
        let Some(draft) = &self.draft else { return };
        match draft.translate(source, &[]) {
            Ok(text) => {
                self.board.set_draft(sid, &text, version);
            }
            Err(e) => log::warn!("draft translation failed: {e:#}"),
        }
    }

    fn run_commit(&self, sid: usize, source: &str) {
        let Some(llm) = &self.llm else { return };
        let prefix = match self.with_state(sid, |st| (!st.final_requested).then(|| st.zh_committed.clone())) {
            Some(p) => p,
            None => return,
        };
        let mut output = match llm.continue_translation(source, &prefix, &self.context(), true) {
            Ok(o) => o,
            Err(e) => {
                log::warn!("commit translation failed: {e:#}");
                return;
            }
        };
        if !output.starts_with(&prefix) {
            output = format!("{prefix}{output}"); // defensive: translator broke the contract
        }
        let chars: Vec<char> = output.chars().collect();
        let committed = self.with_state(sid, |st| {
            st.agreement.update(&chars);
            st.zh_committed = st.agreement.committed().iter().collect();
            (!st.final_requested).then(|| st.zh_committed.clone())
        });
        if let Some(committed) = committed {
            let pending: String = chars.iter().skip(committed.chars().count()).collect();
            self.board.set_commit(sid, &committed, &pending);
        }
    }

    fn run_final(&self, sid: usize, text: &str) {
        let prefix = self.with_state(sid, |st| st.zh_committed.clone());
        let context = self.context();
        let result = match (&self.llm, &self.draft) {
            (None, Some(draft)) => draft.translate(text, &context),
            (Some(llm), _) if self.cfg.final_mode == FinalMode::Continue => llm.continue_translation(text, &prefix, &context, false),
            (Some(llm), _) => llm.translate(text, &context),
            (None, None) => return,
        };
        let result = match result {
            Ok(r) => r.trim().to_string(),
            Err(e) => {
                log::warn!("final translation failed: {e:#}");
                return;
            }
        };
        self.with_state(sid, |st| st.zh_committed = result.clone());
        self.board.set_final_translation(sid, &result);
        let mut h = self.history.lock().unwrap();
        h.push_back((text.to_string(), result));
        while h.len() > self.cfg.context_sentences.max(1) {
            h.pop_front();
        }
    }
}

pub struct TranslationController {
    inner: Arc<Inner>,
    draft_worker: Arc<dyn Worker>,
    llm_worker: Arc<dyn Worker>,
}

impl TranslationController {
    pub fn new(
        board: Arc<SubtitleBoard>,
        draft: Option<Arc<dyn Translator>>,
        llm: Option<Arc<dyn IncrementalTranslator>>,
        cfg: ControllerConfig,
        draft_worker: Option<Arc<dyn Worker>>,
        llm_worker: Option<Arc<dyn Worker>>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner { board, draft, llm, cfg, states: Mutex::new(HashMap::new()), history: Mutex::new(VecDeque::new()) }),
            draft_worker: draft_worker.unwrap_or_else(|| Arc::new(InlineWorker)),
            llm_worker: llm_worker.unwrap_or_else(|| Arc::new(InlineWorker)),
        }
    }

    pub fn board(&self) -> &Arc<SubtitleBoard> {
        &self.inner.board
    }

    pub fn handle(&self, event: &SentenceEvent) {
        match event {
            SentenceEvent::Progress { sid, committed, tentative, segment_id } => {
                self.on_progress(*sid, committed, tentative, *segment_id)
            }
            SentenceEvent::Final { sid, text, segment_id } => self.on_final(*sid, text, *segment_id),
        }
    }

    fn on_progress(&self, sid: usize, committed: &str, tentative: &str, segment_id: usize) {
        let inner = &self.inner;
        if inner.with_state(sid, |st| st.final_requested) {
            return;
        }
        inner.board.set_source(sid, committed, tentative, Some(segment_id));

        // draft: whole current sentence, latest wins
        let source = match (committed.is_empty(), tentative.is_empty()) {
            (false, false) => format!("{committed} {tentative}"),
            (false, true) => committed.to_string(),
            _ => tentative.to_string(),
        };
        if inner.draft.is_some() && !source.is_empty() {
            let skip = inner.cfg.skip_hidden_drafts && inner.llm.is_some();
            let version = inner.with_state(sid, |st| {
                (source != st.last_draft_source && !(skip && !st.zh_committed.is_empty())).then(|| {
                    st.last_draft_source = source.clone();
                    st.draft_version += 1;
                    st.draft_version
                })
            });
            if let Some(version) = version {
                let i = inner.clone();
                self.draft_worker.submit(Job::new(Some(JobKey::Draft), JobTag::Draft(sid), move || {
                    i.run_draft(sid, &source, version)
                }));
            }
        }

        // commit: only committed English goes to the LLM
        if inner.llm.is_some() && !committed.is_empty() {
            let words = split_words(committed);
            let cfg = &inner.cfg;
            let go = inner.with_state(sid, |st| {
                let new_words = words.len() as isize - st.llm_words as isize;
                let boundary = cfg.commit_on_clause && new_words > 0 && is_clause_end_word(words.last().unwrap());
                if new_words >= cfg.commit_min_new_words as isize || boundary {
                    st.llm_words = words.len();
                    true
                } else {
                    false
                }
            });
            if go {
                let i = inner.clone();
                let committed = committed.to_string();
                self.llm_worker.submit(Job::new(Some(JobKey::Commit(sid)), JobTag::Commit(sid), move || {
                    i.run_commit(sid, &committed)
                }));
            }
        }
    }

    fn on_final(&self, sid: usize, text: &str, segment_id: usize) {
        let inner = &self.inner;
        inner.with_state(sid, |st| st.final_requested = true);
        inner.board.finalize_source(sid, text, Some(segment_id));
        self.draft_worker.discard(&|t| *t == JobTag::Draft(sid));
        self.llm_worker.discard(&|t| *t == JobTag::Commit(sid));
        if inner.llm.is_none() && inner.draft.is_none() {
            return; // transcription-only mode
        }
        let i = inner.clone();
        let text = text.to_string();
        let job = Job::new(None, JobTag::Final(sid), move || i.run_final(sid, &text));
        if inner.llm.is_none() {
            self.draft_worker.submit(job); // no LLM: the draft model provides the final translation
        } else {
            self.llm_worker.submit(job);
        }
    }

    pub fn wait_idle(&self, timeout: Option<Duration>) -> bool {
        let ok = self.llm_worker.wait_idle(timeout);
        self.draft_worker.wait_idle(timeout) && ok
    }

    pub fn stop(&self) {
        self.draft_worker.stop();
        self.llm_worker.stop();
    }
}
