//! Turn raw (revisable) ASR hypotheses into committed + tentative text.

use std::collections::VecDeque;

use crate::align::align_tail;
use crate::text::{common_prefix_len, join_words, normalize_word, split_words};

/// LocalAgreement-n over hypothesis tails, relative to the committed text.
/// Committed words are append-only within a segment.
pub struct HypothesisStabilizer {
    n: usize,
    committed: Vec<String>,
    tails: VecDeque<Vec<String>>,
    last: Option<(String, String)>,
}

impl HypothesisStabilizer {
    pub fn new(agreement_n: usize) -> Self {
        assert!(agreement_n >= 1, "agreement_n must be >= 1");
        Self { n: agreement_n, committed: Vec::new(), tails: VecDeque::new(), last: None }
    }

    pub fn committed(&self) -> &[String] {
        &self.committed
    }

    /// Feed a hypothesis; return (committed, tentative), or None if nothing changed.
    /// `allow_commit = false` only refreshes the tentative text.
    pub fn update(&mut self, hypothesis: &str, allow_commit: bool) -> Option<(String, String)> {
        let mut tail = align_tail(&self.committed, &split_words(hypothesis));
        self.tails.push_back(tail.clone());
        if self.tails.len() > self.n {
            self.tails.pop_front();
        }
        if allow_commit && self.tails.len() == self.n {
            let newest = self.tails.back().unwrap();
            let mut agreed = newest.len();
            for other in self.tails.iter().take(self.n - 1) {
                agreed = agreed.min(common_prefix_len(other, newest, |w| normalize_word(w)));
            }
            // the newest hypothesis ends here, so this word's punctuation is a guess
            // ("Phoebe." before "turning away" arrives): wait for a following word
            let agreed = if agreed == tail.len() { agreed as isize - 1 } else { agreed as isize };
            if agreed > 0 {
                let agreed = agreed as usize;
                self.committed.extend(tail[..agreed].iter().cloned());
                for t in self.tails.iter_mut() {
                    t.drain(..agreed);
                }
                tail.drain(..agreed);
            }
        }
        let result = (join_words(&self.committed), join_words(&tail));
        if self.last.as_ref() == Some(&result) {
            return None;
        }
        self.last = Some(result.clone());
        Some(result)
    }

    /// Final segment text; always starts with the already-committed words.
    pub fn finalize(&mut self, hypothesis: &str) -> String {
        let mut words = self.committed.clone();
        words.extend(align_tail(&self.committed, &split_words(hypothesis)));
        self.reset();
        join_words(&words)
    }

    pub fn reset(&mut self) {
        self.committed.clear();
        self.tails.clear();
        self.last = None;
    }
}
