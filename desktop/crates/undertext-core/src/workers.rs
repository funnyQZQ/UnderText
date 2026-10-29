//! Job runners: inline, and a background thread that coalesces queued jobs.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// Jobs with the same key replace each other while queued ("latest wins").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKey {
    Draft,
    Commit(usize),
    /// Provisional speaker label of the open segment.
    Peek,
}

/// Free-form label used by `discard`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobTag {
    Draft(usize),
    Commit(usize),
    Final(usize),
    Other,
}

pub struct Job {
    pub key: Option<JobKey>,
    pub tag: JobTag,
    pub run: Box<dyn FnOnce() + Send>,
}

impl Job {
    pub fn new(key: Option<JobKey>, tag: JobTag, run: impl FnOnce() + Send + 'static) -> Self {
        Self { key, tag, run: Box::new(run) }
    }
}

pub trait Worker: Send + Sync {
    fn submit(&self, job: Job);
    fn discard(&self, predicate: &dyn Fn(&JobTag) -> bool) -> usize;
    fn wait_idle(&self, timeout: Option<Duration>) -> bool;
    fn stop(&self);
}

/// Runs every job immediately in the caller's thread (deterministic).
#[derive(Default)]
pub struct InlineWorker;

impl Worker for InlineWorker {
    fn submit(&self, job: Job) {
        (job.run)();
    }
    fn discard(&self, _: &dyn Fn(&JobTag) -> bool) -> usize {
        0
    }
    fn wait_idle(&self, _: Option<Duration>) -> bool {
        true
    }
    fn stop(&self) {}
}

struct State {
    items: VecDeque<Job>,
    busy: bool,
    closed: bool,
}

struct Shared {
    state: Mutex<State>,
    cond: Condvar,
}

/// Single background thread; a queued job is replaced by a newer one with the same key.
pub struct ThreadWorker {
    shared: Arc<Shared>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl ThreadWorker {
    pub fn new(name: &str) -> Self {
        Self::spawn(name, false)
    }

    /// A worker for work that may run slower but cooler (efficiency cores on macOS).
    pub fn background(name: &str) -> Self {
        Self::spawn(name, true)
    }

    fn spawn(name: &str, background: bool) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { items: VecDeque::new(), busy: false, closed: false }),
            cond: Condvar::new(),
        });
        let s = shared.clone();
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                if background {
                    crate::power::prefer_efficiency_cores();
                }
                Self::run(s)
            })
            .expect("spawn worker thread");
        Self { shared, handle: Mutex::new(Some(handle)) }
    }

    fn run(shared: Arc<Shared>) {
        loop {
            let job = {
                let mut st = shared.state.lock().unwrap();
                while st.items.is_empty() && !st.closed {
                    st = shared.cond.wait(st).unwrap();
                }
                match st.items.pop_front() {
                    Some(job) => {
                        st.busy = true;
                        job
                    }
                    None => return, // closed and drained
                }
            };
            // keep the worker alive: one bad job must not kill the subtitles
            if catch_unwind(AssertUnwindSafe(job.run)).is_err() {
                log::error!("worker job panicked");
            }
            let mut st = shared.state.lock().unwrap();
            st.busy = false;
            shared.cond.notify_all();
        }
    }
}

impl Worker for ThreadWorker {
    fn submit(&self, job: Job) {
        let mut st = self.shared.state.lock().unwrap();
        if let Some(key) = job.key {
            if let Some(slot) = st.items.iter_mut().find(|j| j.key == Some(key)) {
                *slot = job;
                self.shared.cond.notify_all();
                return;
            }
        }
        st.items.push_back(job);
        self.shared.cond.notify_all();
    }

    fn discard(&self, predicate: &dyn Fn(&JobTag) -> bool) -> usize {
        let mut st = self.shared.state.lock().unwrap();
        let before = st.items.len();
        st.items.retain(|j| !predicate(&j.tag));
        before - st.items.len()
    }

    fn wait_idle(&self, timeout: Option<Duration>) -> bool {
        let st = self.shared.state.lock().unwrap();
        let idle = |s: &mut State| !s.items.is_empty() || s.busy;
        match timeout {
            Some(t) => !self.shared.cond.wait_timeout_while(st, t, idle).unwrap().1.timed_out(),
            None => {
                let _st = self.shared.cond.wait_while(st, idle).unwrap();
                true
            }
        }
    }

    fn stop(&self) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.closed = true;
            self.shared.cond.notify_all();
        }
        if let Some(h) = self.handle.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

impl Drop for ThreadWorker {
    fn drop(&mut self) {
        self.stop();
    }
}
