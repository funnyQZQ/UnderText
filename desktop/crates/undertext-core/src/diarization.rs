//! Online speaker diarization: speaker labels for ASR segments, final ones when a segment
//! ends and provisional ones while it is still being spoken.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Speaker {
    /// Set once the speaker has `confirm_s` of speech; until then the voice is a candidate
    /// and its speech is shown without a label.
    label: Option<String>,
    /// Duration-weighted sum of unit embeddings.
    total: Vec<f32>,
    seconds: f64,
    /// Speech that matched this voice, counted towards confirmation.
    heard: f64,
}

impl Speaker {
    fn centroid(&self) -> Vec<f32> {
        unit(&self.total)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Assignment {
    /// Internal speaker index (also for unconfirmed candidates).
    pub id: Option<usize>,
    pub label: Option<String>,
    /// To the chosen (or nearest) speaker; 0.0 when none existed.
    pub similarity: f32,
    pub new: bool,
}

/// Settings of [`SpeakerTracker`]. Thresholds are cosine similarities and depend on
/// the embedding model; the defaults are tuned for NeMo TitaNet (small/large).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeakerTrackerConfig {
    /// Below this similarity to every known speaker a (long enough) voice is a new speaker.
    pub new_speaker_threshold: f32,
    /// Only voices at least this similar refine the speaker's voice print.
    pub update_threshold: f32,
    /// Inside one speech segment, a stretch this dissimilar to what came before is a
    /// different speaker (turn change without a pause).
    pub change_threshold: f32,
    /// Shortest stretch of speech that may create a new speaker.
    pub min_new_speaker_s: f64,
    /// Shortest stretch of speech that may refine a voice print.
    pub min_update_s: f64,
    /// Shorter stretches ("um", "yeah") are too little for a voice print and keep the
    /// current speaker.
    pub min_reliable_s: f64,
    /// Prefer the current speaker when it is at most this much less similar than the best.
    pub continuity_margin: f32,
    /// 0 = unlimited.
    pub max_speakers: usize,
    /// Speech a new voice needs before it gets a label. Stray voices (laughter, crosstalk,
    /// a noisy stretch) rarely reach it, so they don't show up as extra speakers.
    pub confirm_s: f64,
}

impl Default for SpeakerTrackerConfig {
    fn default() -> Self {
        Self {
            new_speaker_threshold: 0.45,
            update_threshold: 0.6,
            change_threshold: 0.5,
            min_new_speaker_s: 2.0,
            min_update_s: 1.5,
            min_reliable_s: 0.8,
            continuity_margin: 0.1,
            max_speakers: 0,
            confirm_s: 5.0,
        }
    }
}

/// Online speaker assignment with hysteresis: a separate (low) threshold for creating
/// speakers, voice prints refined only by confident long stretches, short stretches and
/// near-ties resolved in favour of whoever is talking.
pub struct SpeakerTracker {
    cfg: SpeakerTrackerConfig,
    speakers: Vec<Speaker>,
    current: Option<usize>,
    confirmed: usize,
    /// Speakers confirmed since the last `take_confirmed`.
    newly_confirmed: Vec<usize>,
}

impl SpeakerTracker {
    pub fn new(cfg: SpeakerTrackerConfig) -> Self {
        Self { cfg, speakers: Vec::new(), current: None, confirmed: 0, newly_confirmed: Vec::new() }
    }

    /// Confirmed speakers.
    pub fn num_speakers(&self) -> usize {
        self.confirmed
    }

    /// Voice prints of all speakers, including unconfirmed candidates.
    pub fn centroids(&self) -> Vec<Vec<f32>> {
        self.speakers.iter().map(Speaker::centroid).collect()
    }

    pub fn label(&self, id: usize) -> Option<String> {
        self.speakers.get(id).and_then(|s| s.label.clone())
    }

    pub fn take_confirmed(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.newly_confirmed)
    }

    /// Closest confirmed speaker if the voice clearly matches one: at least `min_sim`
    /// and `margin` ahead of the next one. Changes nothing.
    pub fn identify(&self, embedding: &[f32], min_sim: f32, margin: f32) -> Option<String> {
        let emb = unit(embedding);
        let mut sims: Vec<(f32, &String)> =
            self.speakers.iter().filter_map(|s| s.label.as_ref().map(|l| (dot(&emb, &s.centroid()), l))).collect();
        sims.sort_by(|a, b| b.0.total_cmp(&a.0));
        match sims.as_slice() {
            [] => None,
            [(best, l)] => (*best >= min_sim).then(|| (*l).clone()),
            [(best, l), (second, _), ..] => (*best >= min_sim && best - second >= margin).then(|| (*l).clone()),
        }
    }

    /// A voice unlike every known speaker that has been heard consistently for
    /// `duration_s`: create it as a confirmed speaker right away. None if it resembles
    /// someone already known (including candidates) or the speaker limit is reached.
    pub fn create_early(&mut self, embedding: &[f32], duration_s: f64) -> Option<String> {
        let emb = unit(embedding);
        let c = self.cfg.clone();
        let sims: Vec<f32> = self.speakers.iter().map(|s| dot(&emb, &s.centroid())).collect();
        if let Some(i) = (0..sims.len()).max_by(|&a, &b| sims[a].total_cmp(&sims[b])) {
            if sims[i] >= c.new_speaker_threshold {
                // an unconfirmed candidate heard before: confirm it now (its earlier
                // segments get relabeled with the next final result)
                if self.speakers[i].label.is_none() {
                    self.speakers[i].heard = self.speakers[i].heard.max(c.confirm_s);
                    self.current = Some(i);
                    self.confirm(i);
                    return self.label(i);
                }
                return None;
            }
        }
        if c.max_speakers > 0 && self.confirmed >= c.max_speakers {
            return None;
        }
        let total = emb.iter().map(|x| x * duration_s as f32).collect();
        self.speakers.push(Speaker { label: None, total, seconds: duration_s, heard: duration_s.max(c.confirm_s) });
        let i = self.speakers.len() - 1;
        self.current = Some(i);
        self.confirm(i);
        self.take_confirmed_early(i);
        self.label(i)
    }

    /// Early speakers have no earlier segments to relabel.
    fn take_confirmed_early(&mut self, i: usize) {
        self.newly_confirmed.retain(|&j| j != i);
    }

    pub fn similarity_to(&self, label: &str, embedding: &[f32]) -> f32 {
        let emb = unit(embedding);
        self.speakers.iter().find(|s| s.label.as_deref() == Some(label)).map_or(0.0, |s| dot(&emb, &s.centroid()))
    }

    fn confirm(&mut self, i: usize) {
        let sp = &mut self.speakers[i];
        if sp.label.is_none() && sp.heard >= self.cfg.confirm_s {
            self.confirmed += 1;
            sp.label = Some(format!("S{}", self.confirmed));
            self.newly_confirmed.push(i);
        }
    }

    fn create(&mut self, emb: &[f32], duration: f64) -> Assignment {
        let total = emb.iter().map(|x| x * duration as f32).collect();
        self.speakers.push(Speaker { label: None, total, seconds: duration, heard: duration });
        let i = self.speakers.len() - 1;
        self.current = Some(i);
        self.confirm(i);
        Assignment { id: Some(i), label: self.label(i), similarity: 1.0, new: true }
    }

    fn pick(&mut self, i: usize, sim: f32, emb: &[f32], duration: f64, update: bool) -> Assignment {
        let c = &self.cfg;
        let consistent = sim >= c.new_speaker_threshold;
        // a candidate's first print may come from a short or mixed stretch: let any
        // consistent speech refine it
        let update = update || (self.speakers[i].label.is_none() && consistent && duration >= c.min_update_s);
        let sp = &mut self.speakers[i];
        if update {
            for (t, e) in sp.total.iter_mut().zip(emb) {
                *t += e * duration as f32;
            }
            sp.seconds += duration;
        }
        if consistent && duration >= c.min_reliable_s {
            sp.heard += duration;
        }
        self.confirm(i);
        self.current = Some(i);
        Assignment { id: Some(i), label: self.label(i), similarity: sim, new: false }
    }

    pub fn assign(&mut self, embedding: &[f32], duration_s: f64) -> Assignment {
        let c = self.cfg.clone();
        let emb = unit(embedding);
        if self.speakers.is_empty() {
            return if duration_s >= c.min_new_speaker_s {
                self.create(&emb, duration_s)
            } else {
                Assignment { id: None, label: None, similarity: 0.0, new: false }
            };
        }
        let sims: Vec<f32> = self.speakers.iter().map(|s| dot(&emb, &s.centroid())).collect();
        let mut best = (0..sims.len()).fold(0, |b, i| if sims[i] > sims[b] { i } else { b });
        if let Some(cur) = self.current {
            if sims[cur] >= sims[best] - c.continuity_margin && sims[cur] >= c.new_speaker_threshold {
                best = cur;
            }
        }
        let sim = sims[best];
        if duration_s < c.min_reliable_s {
            // too little audio to judge: stay with the current speaker, unless another one
            // is clearly closer (a quick "yeah" from the other side)
            let top = (0..sims.len()).fold(0, |b, i| if sims[i] > sims[b] { i } else { b });
            let i = match self.current {
                Some(cur) if top != cur && sims[top] >= 0.5 && sims[top] - sims[cur] >= 0.3 => {
                    self.current = Some(top);
                    top
                }
                Some(cur) => cur,
                None => best,
            };
            return Assignment { id: Some(i), label: self.label(i), similarity: sims[i], new: false };
        }
        if sim >= c.update_threshold {
            return self.pick(best, sim, &emb, duration_s, duration_s >= c.min_update_s);
        }
        let full = c.max_speakers > 0 && self.confirmed >= c.max_speakers;
        if sim < c.new_speaker_threshold && duration_s >= c.min_new_speaker_s && !full {
            return self.create(&emb, duration_s);
        }
        self.pick(best, sim, &emb, duration_s, false)
    }
}

/// Group consecutive window embeddings into stretches of one voice: a new stretch starts
/// when a window and the one after it both fall below `threshold` against the mean of the
/// current stretch (which needs at least two windows first).
pub fn split_by_voice(windows: &[Vec<f32>], threshold: f32) -> Vec<std::ops::Range<usize>> {
    if windows.len() < 3 {
        return vec![0..windows.len()];
    }
    let mut ranges = vec![0..1];
    for i in 1..windows.len() {
        let cur = ranges.last().unwrap().clone();
        let mut mean = vec![0.0f32; windows[0].len()];
        for w in &windows[cur.clone()] {
            for (m, x) in mean.iter_mut().zip(w) {
                *m += x;
            }
        }
        let mean = unit(&mean);
        let below = |j: usize| dot(&mean, &windows[j]) < threshold;
        let next_below = i + 1 >= windows.len() || below(i + 1);
        if cur.len() >= 2 && below(i) && next_below {
            ranges.push(i..i + 1);
        } else {
            ranges.last_mut().unwrap().end = i + 1;
        }
    }
    ranges
}

/// One speaker over part of a segment; `start`/`end` are fractions of its duration.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerSpan {
    pub start: f32,
    pub end: f32,
    pub id: Option<usize>,
    pub label: Option<String>,
    pub similarity: f32,
    pub new: bool,
}

/// Voice fingerprint model (sherpa-onnx speaker embedding extractor).
pub trait Embedder: Send + Sync {
    fn embed(&self, samples: &[f32]) -> Vec<f32>;
}

const WINDOW: usize = 24000; // 1.5 s
const HOP: usize = 12000; // 0.75 s

/// A provisional label needs a clear match: this similarity to a known speaker, and this
/// much more than to the next one.
const PEEK_MIN_SIM: f32 = 0.5;
const PEEK_MARGIN: f32 = 0.1;
/// A voice nobody knows gets a label while it is still speaking once this many
/// consecutive windows (1.5 s each) agree with each other this well.
const EARLY_WINDOWS: usize = 3;
const EARLY_CONSISTENCY: f32 = 0.45;
/// Finished segments remembered for relabeling when a candidate speaker is confirmed.
const HISTORY: usize = 60;

/// Labels for one segment.
pub type SegmentLabels = (usize, Vec<SpeakerSpan>);

#[derive(Default)]
struct OpenSegment {
    seg: Option<usize>,
    /// (seconds into the segment where the window started, label); None marks a voice
    /// that is clearly not the previous speaker but not a known one either.
    peeks: Vec<(f64, Option<String>)>,
    /// Consecutive windows that matched nobody: (start in the segment, embedding).
    unknown: Vec<(f64, Vec<f32>)>,
}

pub struct Diarizer {
    embedder: Box<dyn Embedder>,
    tracker: Mutex<SpeakerTracker>,
    change_threshold: f32,
    history: Mutex<VecDeque<(usize, Vec<SpeakerSpan>)>>,
    open: Mutex<OpenSegment>,
}

impl Diarizer {
    pub fn new(embedder: Box<dyn Embedder>, cfg: SpeakerTrackerConfig) -> Self {
        let change_threshold = cfg.change_threshold;
        Self {
            embedder,
            tracker: Mutex::new(SpeakerTracker::new(cfg)),
            change_threshold,
            history: Mutex::new(VecDeque::new()),
            open: Mutex::new(OpenSegment::default()),
        }
    }

    /// Provisional speakers for a segment that is still being spoken, from its latest
    /// `window` of audio (ending `elapsed_s` into the segment). Only clear matches with a
    /// known speaker count; returns None until there is one.
    pub fn peek(&self, seg: usize, elapsed_s: f64, window: &[f32]) -> Option<Vec<SpeakerSpan>> {
        if self.history.lock().unwrap().iter().any(|(s, _)| *s == seg) {
            return None; // already labeled for good
        }
        let emb = unit(&self.embedder.embed(window));
        let start = (elapsed_s - window.len() as f64 / 16000.0).max(0.0);
        let mut open = self.open.lock().unwrap();
        if open.seg != Some(seg) {
            *open = OpenSegment { seg: Some(seg), ..Default::default() };
        }
        let mut tracker = self.tracker.lock().unwrap();
        let mut at = start;
        let label = match tracker.identify(&emb, PEEK_MIN_SIM, PEEK_MARGIN) {
            Some(l) => {
                open.unknown.clear();
                Some(l)
            }
            None => {
                open.unknown.push((start, emb.clone()));
                let run = &open.unknown[open.unknown.len().saturating_sub(EARLY_WINDOWS)..];
                let consistent = run.len() == EARLY_WINDOWS && run.windows(2).all(|p| dot(&p[0].1, &p[1].1) >= EARLY_CONSISTENCY);
                let created = if consistent {
                    let mut mean = vec![0.0f32; emb.len()];
                    for (_, e) in run {
                        for (m, x) in mean.iter_mut().zip(e) {
                            *m += x;
                        }
                    }
                    let first = run[0].0;
                    tracker.create_early(&mean, 1.5 * EARLY_WINDOWS as f64).map(|l| (first, l))
                } else {
                    None
                };
                match created {
                    // a new voice that has been talking for a few seconds: label it now
                    Some((first, l)) => {
                        open.unknown.clear();
                        at = first;
                        Some(l)
                    }
                    None => match open.peeks.last() {
                        // someone else took over: stop extending the previous label
                        Some((_, Some(prev))) if tracker.similarity_to(prev, &emb) < tracker.cfg.new_speaker_threshold => None,
                        _ => return self.open_spans(&open, elapsed_s),
                    },
                }
            }
        };
        drop(tracker);
        if open.peeks.last().map(|p| &p.1) != Some(&label) && (label.is_some() || !open.peeks.is_empty()) {
            // a label found early covers the windows that led to it
            while label.is_some() && open.peeks.last().is_some_and(|p| p.0 >= at && p.1.is_none()) {
                open.peeks.pop();
            }
            if open.peeks.last().map(|p| &p.1) != Some(&label) {
                open.peeks.push((at, label));
            }
        }
        self.open_spans(&open, elapsed_s)
    }

    fn open_spans(&self, open: &OpenSegment, elapsed_s: f64) -> Option<Vec<SpeakerSpan>> {
        if open.peeks.is_empty() || elapsed_s <= 0.0 {
            return None;
        }
        let n = open.peeks.len();
        Some(
            (0..n)
                .map(|k| SpeakerSpan {
                    start: if k == 0 { 0.0 } else { (open.peeks[k].0 / elapsed_s) as f32 },
                    end: if k + 1 == n { 1.0 } else { (open.peeks[k + 1].0 / elapsed_s) as f32 },
                    id: None,
                    label: open.peeks[k].1.clone(),
                    similarity: 0.0,
                    new: false,
                })
                .collect(),
        )
    }

    /// Final speakers of segment `seg` (first entry), plus earlier segments whose labels
    /// changed because a candidate speaker in them has now been confirmed.
    pub fn label_segment(&self, seg: usize, samples: &[f32]) -> Vec<SegmentLabels> {
        let spans = self.spans(samples);
        let mut tracker = self.tracker.lock().unwrap();
        let confirmed = tracker.take_confirmed();
        let relabel = |spans: &mut Vec<SpeakerSpan>| {
            for s in spans.iter_mut() {
                s.label = s.id.and_then(|i| tracker.label(i));
            }
        };
        let mut spans = spans;
        relabel(&mut spans);
        let mut out = vec![(seg, spans.clone())];
        let mut history = self.history.lock().unwrap();
        if !confirmed.is_empty() {
            for (s, past) in history.iter_mut() {
                if past.iter().any(|p| p.id.is_some_and(|i| confirmed.contains(&i))) {
                    relabel(past);
                    out.push((*s, past.clone()));
                }
            }
        }
        history.push_back((seg, spans));
        while history.len() > HISTORY {
            history.pop_front();
        }
        out
    }

    /// Segments of 3 s or more are scanned in 1.5 s windows so a turn change without a
    /// pause splits the segment.
    fn spans(&self, samples: &[f32]) -> Vec<SpeakerSpan> {
        let duration = samples.len() as f64 / 16000.0;
        if duration < 0.3 {
            return vec![SpeakerSpan { start: 0.0, end: 1.0, id: None, label: None, similarity: 0.0, new: false }];
        }
        let mut windows = Vec::new();
        let mut starts = Vec::new();
        if samples.len() >= WINDOW {
            let mut i = 0;
            while i + WINDOW <= samples.len() {
                windows.push(unit(&self.embedder.embed(&samples[i..i + WINDOW])));
                starts.push(i);
                i += HOP;
            }
        }
        let ranges = split_by_voice(&windows, self.change_threshold);
        let mut tracker = self.tracker.lock().unwrap();
        if ranges.len() <= 1 {
            let a = tracker.assign(&self.embedder.embed(samples), duration);
            return vec![SpeakerSpan { start: 0.0, end: 1.0, id: a.id, label: a.label, similarity: a.similarity, new: a.new }];
        }
        let n = samples.len() as f32;
        let last = ranges.len() - 1;
        ranges
            .iter()
            .enumerate()
            .map(|(k, r)| {
                let mut mean = vec![0.0f32; windows[0].len()];
                for w in &windows[r.clone()] {
                    for (m, x) in mean.iter_mut().zip(w) {
                        *m += x;
                    }
                }
                let dur = 0.75 * (r.len() + 1) as f64;
                let a = tracker.assign(&mean, dur);
                let start = if k == 0 { 0.0 } else { starts[r.start] as f32 / n };
                let end = if k == last { 1.0 } else { starts[ranges[k + 1].start] as f32 / n };
                SpeakerSpan { start, end, id: a.id, label: a.label, similarity: a.similarity, new: a.new }
            })
            .collect()
    }

    pub fn num_speakers(&self) -> usize {
        self.tracker.lock().unwrap().num_speakers()
    }

    pub fn centroids(&self) -> Vec<Vec<f32>> {
        self.tracker.lock().unwrap().centroids()
    }
}
