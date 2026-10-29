//! sherpa-onnx adapters: Silero VAD, Parakeet-TDT (offline transducer), speaker embeddings.

use std::path::Path;

use anyhow::{anyhow, Result};
use undertext_core::diarization::Embedder;
use undertext_core::vad::Vad;
use undertext_core::OfflineRecognizer;
use sherpa_onnx::{
    OfflineRecognizer as SherpaOffline, OfflineRecognizerConfig, OfflineTransducerModelConfig, SileroVadModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, VadModelConfig, VoiceActivityDetector,
};

const SR: i32 = 16000;

fn path_str(p: &Path) -> Result<String> {
    if !p.exists() {
        return Err(anyhow!("model file not found: {}", p.display()));
    }
    Ok(p.to_string_lossy().into_owned())
}

/// Silero VAD.
pub struct SileroVad {
    vad: VoiceActivityDetector,
    window_size: usize,
    min_silence_s: f32,
}

impl SileroVad {
    pub fn new(model: &Path, threshold: f32, min_silence_s: f32) -> Result<Self> {
        let window_size = 512;
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(path_str(model)?),
                threshold,
                min_silence_duration: min_silence_s,
                min_speech_duration: 0.25,
                window_size: window_size as i32,
                max_speech_duration: 30.0,
            },
            sample_rate: SR,
            num_threads: 1,
            provider: Some("cpu".into()),
            ..Default::default()
        };
        let vad = VoiceActivityDetector::create(&config, 120.0).ok_or_else(|| anyhow!("failed to create Silero VAD"))?;
        Ok(Self { vad, window_size, min_silence_s })
    }
}

impl Vad for SileroVad {
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn min_silence_s(&self) -> f64 {
        self.min_silence_s as f64
    }

    fn accept_waveform(&mut self, window: &[f32]) {
        self.vad.accept_waveform(window);
    }

    fn is_speech(&self) -> bool {
        self.vad.detected()
    }

    fn pop_segments(&mut self) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        while let Some(seg) = self.vad.front() {
            out.push(seg.samples().to_vec());
            self.vad.pop();
        }
        out
    }

    fn flush(&mut self) {
        self.vad.flush();
    }

    fn reset(&mut self) {
        self.vad.reset();
    }
}

/// Offline transducer recognizer (Parakeet-TDT 0.6B with punctuation and casing).
pub struct TransducerRecognizer {
    rec: SherpaOffline,
}

pub struct TransducerFiles<'a> {
    pub encoder: &'a Path,
    pub decoder: &'a Path,
    pub joiner: &'a Path,
    pub tokens: &'a Path,
}

impl TransducerRecognizer {
    pub fn new(files: TransducerFiles, model_type: &str, num_threads: i32, provider: &str) -> Result<Self> {
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(path_str(files.encoder)?),
            decoder: Some(path_str(files.decoder)?),
            joiner: Some(path_str(files.joiner)?),
        };
        config.model_config.tokens = Some(path_str(files.tokens)?);
        config.model_config.model_type = Some(model_type.into());
        config.model_config.num_threads = num_threads;
        config.model_config.provider = Some(provider.into());
        config.decoding_method = Some("greedy_search".into());
        let rec = SherpaOffline::create(&config).ok_or_else(|| anyhow!("failed to create the ASR recognizer"))?;
        Ok(Self { rec })
    }
}

impl OfflineRecognizer for TransducerRecognizer {
    fn transcribe(&self, samples: &[f32]) -> String {
        undertext_core::metrics::ASR.time_units(samples.len() as u64, || {
            let stream = self.rec.create_stream();
            stream.accept_waveform(SR, samples);
            self.rec.decode(&stream);
            stream.get_result().map(|r| r.text.trim().to_string()).unwrap_or_default()
        })
    }
}

/// Speaker embedding model (NeMo TitaNet by default).
pub struct SpeakerEmbedder {
    extractor: SpeakerEmbeddingExtractor,
}

impl SpeakerEmbedder {
    pub fn new(model: &Path, num_threads: i32) -> Result<Self> {
        let config = SpeakerEmbeddingExtractorConfig { model: Some(path_str(model)?), num_threads, ..Default::default() };
        let extractor =
            SpeakerEmbeddingExtractor::create(&config).ok_or_else(|| anyhow!("failed to create the speaker model"))?;
        Ok(Self { extractor })
    }
}

impl Embedder for SpeakerEmbedder {
    fn embed(&self, samples: &[f32]) -> Vec<f32> {
        undertext_core::metrics::SPEAKER.time(|| {
            let Some(stream) = self.extractor.create_stream() else { return vec![] };
            stream.accept_waveform(SR, samples);
            stream.input_finished();
            self.extractor.compute(&stream).unwrap_or_default()
        })
    }
}
