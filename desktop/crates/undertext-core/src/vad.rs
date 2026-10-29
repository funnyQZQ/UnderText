//! Voice activity detection interface and a simple energy-based VAD (fallback).

use std::collections::VecDeque;


pub trait Vad: Send {
    fn window_size(&self) -> usize;
    fn accept_waveform(&mut self, window: &[f32]);
    fn is_speech(&self) -> bool;
    /// Completed speech segments since the last call.
    fn pop_segments(&mut self) -> Vec<Vec<f32>>;
    /// Close any speech in progress so it is returned by the next `pop_segments`.
    fn flush(&mut self);
    fn reset(&mut self);
    /// Silence the VAD waits for before it closes a segment (used to estimate where
    /// speech actually ended).
    fn min_silence_s(&self) -> f64 {
        0.0
    }
}

/// RMS-threshold VAD with hangover; fine for clean audio.
pub struct EnergyVad {
    window_size: usize,
    threshold: f32,
    silence_windows: usize,
    min_speech_windows: usize,
    preroll_windows: usize,
    preroll: VecDeque<Vec<f32>>,
    current: Vec<Vec<f32>>,
    voiced: usize,
    silent: usize,
    in_speech: bool,
    segments: Vec<Vec<f32>>,
}

impl EnergyVad {
    pub fn new(sample_rate: usize, window_size: usize, threshold: f32, min_silence_s: f64, min_speech_s: f64) -> Self {
        let per_window = sample_rate as f64 / window_size as f64;
        Self {
            window_size,
            threshold,
            silence_windows: ((min_silence_s * per_window).round_ties_even() as usize).max(1),
            min_speech_windows: ((min_speech_s * per_window).round_ties_even() as usize).max(1),
            preroll_windows: 2,
            preroll: VecDeque::new(),
            current: Vec::new(),
            voiced: 0,
            silent: 0,
            in_speech: false,
            segments: Vec::new(),
        }
    }

    fn close(&mut self) {
        if self.voiced >= self.min_speech_windows && !self.current.is_empty() {
            let trailing = self.silent.saturating_sub(1);
            let keep = self.current.len() - trailing;
            self.segments.push(self.current[..keep].concat());
        }
        self.current.clear();
        self.preroll.clear();
        self.in_speech = false;
        self.voiced = 0;
        self.silent = 0;
    }
}

impl Vad for EnergyVad {
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn min_silence_s(&self) -> f64 {
        self.silence_windows as f64 * self.window_size as f64 / 16000.0
    }

    fn accept_waveform(&mut self, window: &[f32]) {
        let rms = if window.is_empty() {
            0.0
        } else {
            (window.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / window.len() as f64).sqrt()
        };
        let voiced = rms >= self.threshold as f64;
        if !self.in_speech {
            if voiced {
                self.in_speech = true;
                self.current = self.preroll.iter().cloned().collect();
                self.current.push(window.to_vec());
                self.voiced = 1;
                self.silent = 0;
            } else {
                self.preroll.push_back(window.to_vec());
                if self.preroll.len() > self.preroll_windows {
                    self.preroll.pop_front();
                }
            }
            return;
        }
        self.current.push(window.to_vec());
        if voiced {
            self.voiced += 1;
            self.silent = 0;
        } else {
            self.silent += 1;
            if self.silent >= self.silence_windows {
                self.close();
            }
        }
    }

    fn is_speech(&self) -> bool {
        self.in_speech
    }

    fn pop_segments(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.segments)
    }

    fn flush(&mut self) {
        if self.in_speech {
            self.close();
        }
    }

    fn reset(&mut self) {
        self.preroll.clear();
        self.current.clear();
        self.voiced = 0;
        self.silent = 0;
        self.in_speech = false;
        self.segments.clear();
    }
}
