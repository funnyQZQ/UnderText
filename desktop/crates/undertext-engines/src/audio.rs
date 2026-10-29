//! Audio IO: WAV files, microphone / loopback capture and playback (cpal).
//!
//! Every source yields 16 kHz mono f32 chunks. System audio comes from a loopback input
//! device such as BlackHole.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

pub const SR: u32 = 16000;

/// Whole-buffer linear resampler.
pub fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let duration = samples.len() as f64 / src_rate as f64;
    let n_out = (duration * dst_rate as f64).round() as usize;
    let step = src_rate as f64 / dst_rate as f64;
    (0..n_out)
        .map(|k| {
            let t = k as f64 * step;
            let i = t.floor() as usize;
            if i + 1 >= samples.len() {
                samples[samples.len() - 1]
            } else {
                let f = (t - i as f64) as f32;
                samples[i] * (1.0 - f) + samples[i + 1] * f
            }
        })
        .collect()
}

/// Linear resampler for a continuous stream of chunks (keeps state across chunks).
pub struct StreamResampler {
    step: f64,
    t: f64,
    buf: Vec<f32>,
}

impl StreamResampler {
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        Self { step: src_rate as f64 / dst_rate as f64, t: 0.0, buf: Vec::new() }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if (self.step - 1.0).abs() < 1e-12 {
            return input.to_vec();
        }
        self.buf.extend_from_slice(input);
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 1);
        while self.t + 1.0 < self.buf.len() as f64 {
            let i = self.t.floor() as usize;
            let f = (self.t - i as f64) as f32;
            out.push(self.buf[i] * (1.0 - f) + self.buf[i + 1] * f);
            self.t += self.step;
        }
        let consumed = (self.t.floor() as usize).min(self.buf.len());
        self.buf.drain(..consumed);
        self.t -= consumed as f64;
        out
    }
}

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect()
}

/// Length of a WAV file in seconds (header only).
pub fn wav_duration_s(path: &Path) -> Result<f64> {
    let reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(reader.duration() as f64 / reader.spec().sample_rate as f64)
}

/// Read a PCM/float WAV file as 16 kHz mono.
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    Ok(resample(&downmix(&samples, spec.channels as usize), spec.sample_rate, SR))
}

pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

pub fn list_input_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok());
    Ok(host
        .input_devices()?
        .filter_map(|d| d.name().ok())
        .map(|name| DeviceInfo { is_default: Some(&name) == default.as_ref(), name })
        .collect())
}

pub fn list_output_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default = host.default_output_device().and_then(|d| d.name().ok());
    Ok(host
        .output_devices()?
        .filter_map(|d| d.name().ok())
        .map(|name| DeviceInfo { is_default: Some(&name) == default.as_ref(), name })
        .collect())
}

fn pick_device(input: bool, name: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();
    if name.is_empty() {
        let d = if input { host.default_input_device() } else { host.default_output_device() };
        return d.ok_or_else(|| anyhow!("no default {} device", if input { "input" } else { "output" }));
    }
    let wanted = name.to_lowercase();
    let mut devices: Box<dyn Iterator<Item = cpal::Device>> =
        if input { Box::new(host.input_devices()?) } else { Box::new(host.output_devices()?) };
    devices
        .find(|d| d.name().map(|n| n.to_lowercase().contains(&wanted)).unwrap_or(false))
        .ok_or_else(|| anyhow!("no {} device matching {name:?}", if input { "input" } else { "output" }))
}

/// A source of 16 kHz mono chunks. Created on the thread that consumes it (cpal streams
/// are not `Send` on every platform).
pub trait AudioSource {
    fn next_chunk(&mut self) -> Option<Vec<f32>>;
    /// Seconds of audio delivered so far.
    fn position_s(&self) -> f64;
}

/// Iterate a source until it ends or `stop` is set.
pub struct SourceIter<'a> {
    pub source: &'a mut dyn AudioSource,
    pub stop: Arc<AtomicBool>,
}

impl Iterator for SourceIter<'_> {
    type Item = Vec<f32>;
    fn next(&mut self) -> Option<Vec<f32>> {
        if self.stop.load(Ordering::SeqCst) {
            return None;
        }
        self.source.next_chunk()
    }
}

// --- microphone / loopback input ---------------------------------------------------------

pub struct MicSource {
    _stream: cpal::Stream,
    rx: Receiver<Vec<f32>>,
    resampler: StreamResampler,
    pending: Vec<f32>,
    chunk: usize,
    delivered: usize,
    stop: Arc<AtomicBool>,
    pub device_name: String,
}

fn build_input<T>(device: &cpal::Device, config: &cpal::StreamConfig, tx: mpsc::Sender<Vec<f32>>) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let floats: Vec<f32> = data.iter().map(|s| s.to_sample::<f32>()).collect();
            let _ = tx.send(downmix(&floats, channels));
        },
        |e| log::error!("input stream error: {e}"),
        None,
    )?;
    Ok(stream)
}

impl MicSource {
    pub fn open(device_name: &str, chunk_ms: u32, stop: Arc<AtomicBool>) -> Result<Self> {
        let device = pick_device(true, device_name)?;
        let name = device.name().unwrap_or_default();
        let supported = device.default_input_config()?;
        let config: cpal::StreamConfig = supported.config();
        let (tx, rx) = mpsc::channel();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => build_input::<f32>(&device, &config, tx)?,
            cpal::SampleFormat::I16 => build_input::<i16>(&device, &config, tx)?,
            cpal::SampleFormat::I32 => build_input::<i32>(&device, &config, tx)?,
            cpal::SampleFormat::U16 => build_input::<u16>(&device, &config, tx)?,
            other => bail!("unsupported input sample format {other:?}"),
        };
        stream.play()?;
        log::info!("capturing from {name} ({} Hz, {} ch)", config.sample_rate.0, config.channels);
        Ok(Self {
            _stream: stream,
            rx,
            resampler: StreamResampler::new(config.sample_rate.0, SR),
            pending: Vec::new(),
            chunk: (SR * chunk_ms / 1000) as usize,
            delivered: 0,
            stop,
            device_name: name,
        })
    }
}

impl AudioSource for MicSource {
    fn next_chunk(&mut self) -> Option<Vec<f32>> {
        while self.pending.len() < self.chunk {
            if self.stop.load(Ordering::SeqCst) {
                return None;
            }
            match self.rx.recv_timeout(Duration::from_millis(200)) {
                Ok(block) => {
                    let out = self.resampler.process(&block);
                    self.pending.extend(out);
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
        let rest = self.pending.split_off(self.chunk);
        let chunk = std::mem::replace(&mut self.pending, rest);
        self.delivered += chunk.len();
        Some(chunk)
    }

    fn position_s(&self) -> f64 {
        self.delivered as f64 / SR as f64
    }
}

// --- playback ---------------------------------------------------------------------------

/// Plays a 16 kHz buffer through an output device. `heard()` is the number of 16 kHz
/// samples that have actually reached the speakers (handed to the device minus the
/// device's reported output latency); it is the clock that paces the pipeline.
pub struct Player {
    _stream: cpal::Stream,
    handed: Arc<AtomicUsize>,
    delay_us: Arc<AtomicU64>,
    rate: u32,
    total: usize,
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    data: Arc<Vec<f32>>,
    handed: Arc<AtomicUsize>,
    delay_us: Arc<AtomicU64>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let stream = device.build_output_stream(
        config,
        move |out: &mut [T], info: &cpal::OutputCallbackInfo| {
            let ts = info.timestamp();
            if let Some(d) = ts.playback.duration_since(&ts.callback) {
                delay_us.store(d.as_micros() as u64, Ordering::Relaxed);
            }
            let mut pos = handed.load(Ordering::Relaxed);
            for frame in out.chunks_mut(channels) {
                let v = data.get(pos).copied().unwrap_or(0.0);
                if pos < data.len() {
                    pos += 1;
                }
                for s in frame.iter_mut() {
                    *s = T::from_sample(v);
                }
            }
            handed.store(pos, Ordering::Relaxed);
        },
        |e| log::error!("output stream error: {e}"),
        None,
    )?;
    Ok(stream)
}

impl Player {
    pub fn start(samples_16k: &[f32], device_name: &str) -> Result<Self> {
        let device = pick_device(false, device_name)?;
        let supported = device.default_output_config()?;
        let config: cpal::StreamConfig = supported.config();
        let rate = config.sample_rate.0;
        let data = Arc::new(resample(samples_16k, SR, rate));
        let total = data.len();
        let handed = Arc::new(AtomicUsize::new(0));
        let delay_us = Arc::new(AtomicU64::new(0));
        let (h, d) = (handed.clone(), delay_us.clone());
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => build_output::<f32>(&device, &config, data, h, d)?,
            cpal::SampleFormat::I16 => build_output::<i16>(&device, &config, data, h, d)?,
            cpal::SampleFormat::I32 => build_output::<i32>(&device, &config, data, h, d)?,
            cpal::SampleFormat::U16 => build_output::<u16>(&device, &config, data, h, d)?,
            other => bail!("unsupported output sample format {other:?}"),
        };
        stream.play()?;
        Ok(Self { _stream: stream, handed, delay_us, rate, total })
    }

    /// 16 kHz samples that have reached the speakers.
    pub fn heard(&self) -> usize {
        let handed = self.handed.load(Ordering::Relaxed);
        let delay = (self.delay_us.load(Ordering::Relaxed) as f64 * self.rate as f64 / 1e6) as usize;
        (handed.saturating_sub(delay) as f64 * SR as f64 / self.rate as f64) as usize
    }

    pub fn finished(&self) -> bool {
        self.handed.load(Ordering::Relaxed) >= self.total
    }
}

// --- WAV source -------------------------------------------------------------------------

pub enum Pacing {
    /// As fast as possible.
    Fast,
    /// Wall-clock real time.
    Realtime,
    /// Played through the speakers; a chunk is released only after it was heard.
    Play { output_device: String },
}

pub struct WavSource {
    data: Vec<f32>,
    chunk: usize,
    pos: usize,
    pacing: Pacing,
    player: Option<Player>,
    started: Option<Instant>,
    stop: Arc<AtomicBool>,
}

impl WavSource {
    pub fn open(path: &Path, chunk_ms: u32, pacing: Pacing, stop: Arc<AtomicBool>) -> Result<Self> {
        Ok(Self::from_samples(read_wav(path)?, chunk_ms, pacing, stop))
    }

    pub fn from_samples(data: Vec<f32>, chunk_ms: u32, pacing: Pacing, stop: Arc<AtomicBool>) -> Self {
        Self { data, chunk: (SR * chunk_ms / 1000) as usize, pos: 0, pacing, player: None, started: None, stop }
    }

    pub fn duration_s(&self) -> f64 {
        self.data.len() as f64 / SR as f64
    }
}

impl AudioSource for WavSource {
    fn next_chunk(&mut self) -> Option<Vec<f32>> {
        if self.pos >= self.data.len() || self.stop.load(Ordering::SeqCst) {
            return None;
        }
        let end = (self.pos + self.chunk).min(self.data.len());
        match &self.pacing {
            Pacing::Fast => {}
            Pacing::Realtime => {
                let start = *self.started.get_or_insert_with(Instant::now);
                let due = start + Duration::from_secs_f64(end as f64 / SR as f64);
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
            }
            Pacing::Play { output_device } => {
                if self.player.is_none() {
                    match Player::start(&self.data, output_device) {
                        Ok(p) => self.player = Some(p),
                        Err(e) => {
                            log::error!("playback failed ({e:#}); continuing in real time without sound");
                            self.pacing = Pacing::Realtime;
                            return self.next_chunk();
                        }
                    }
                }
                let player = self.player.as_ref().unwrap();
                // if processing fell behind, the backlog is released at once
                while player.heard() < end && !player.finished() {
                    if self.stop.load(Ordering::SeqCst) {
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        let chunk = self.data[self.pos..end].to_vec();
        self.pos = end;
        Some(chunk)
    }

    fn position_s(&self) -> f64 {
        self.pos as f64 / SR as f64
    }
}
