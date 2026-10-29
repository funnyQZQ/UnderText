//! LocalAgreement-n policy.
//!
//! Given a stream of hypotheses for a growing input, commit the longest prefix that the
//! last `n` hypotheses agree on. Committed tokens are never revoked.

use std::collections::VecDeque;

use crate::text::common_prefix_len;

pub struct LocalAgreement<T: Clone, K: PartialEq = T> {
    n: usize,
    key: fn(&T) -> K,
    history: VecDeque<Vec<T>>,
    committed: Vec<T>,
}

impl<T: Clone + PartialEq> LocalAgreement<T, T> {
    pub fn new(n: usize) -> Self {
        Self::with_key(n, <T as Clone>::clone)
    }
}

impl<T: Clone, K: PartialEq> LocalAgreement<T, K> {
    pub fn with_key(n: usize, key: fn(&T) -> K) -> Self {
        assert!(n >= 1, "n must be >= 1");
        Self { n, key, history: VecDeque::new(), committed: Vec::new() }
    }

    pub fn committed(&self) -> &[T] {
        &self.committed
    }

    /// Part of `hypothesis` that comes after the committed prefix.
    pub fn tentative(&self, hypothesis: &[T]) -> Vec<T> {
        hypothesis.get(self.committed.len()..).unwrap_or(&[]).to_vec()
    }

    /// Feed a full hypothesis; return the tokens newly committed by this call.
    pub fn update(&mut self, hypothesis: &[T]) -> Vec<T> {
        self.history.push_back(hypothesis.to_vec());
        if self.history.len() > self.n {
            self.history.pop_front();
        }
        if self.history.len() < self.n {
            return Vec::new();
        }
        let key = self.key;
        let first = &self.history[0];
        let mut agreed = first.len();
        for other in self.history.iter().skip(1) {
            agreed = agreed.min(common_prefix_len(first, other, key));
        }
        // every hypothesis must also agree with what is already committed
        if common_prefix_len(&self.committed, hypothesis, key) < self.committed.len() {
            return Vec::new();
        }
        if agreed <= self.committed.len() {
            return Vec::new();
        }
        let new = hypothesis[self.committed.len()..agreed].to_vec();
        self.committed.extend(new.iter().cloned());
        new
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.committed.clear();
    }
}
