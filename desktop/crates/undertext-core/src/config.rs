//! Application configuration (TOML), shared by the CLI and the desktop app.
//!
//! Model paths are relative to `models_dir` (the project's `models/` folder by default).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paragraphs::ParagraphConfig;
use crate::controller::ControllerConfig;
use crate::diarization::SpeakerTrackerConfig;
use crate::vad_chunked::VadChunkedConfig;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// Microphone or any input device (e.g. BlackHole for system audio).
    Mic,
    /// A WAV file, optionally played through the speakers.
    Wav,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub source: SourceKind,
    /// Input device name substring; empty = system default.
    pub device: String,
    pub wav_path: String,
    /// WAV: play through the speakers and pace the pipeline by playback.
    pub play: bool,
    /// Output device name substring; empty = system default.
    pub output_device: String,
    pub chunk_ms: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self { source: SourceKind::Mic, device: String::new(), wav_path: String::new(), play: true, output_device: String::new(), chunk_ms: 100 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AsrConfig {
    /// onnxruntime provider: cpu | coreml | cuda | directml
    pub provider: String,
    pub num_threads: i32,
    pub encoder: String,
    pub decoder: String,
    pub joiner: String,
    pub tokens: String,
    pub model_type: String,
    /// "silero" | "energy"
    pub vad: String,
    pub vad_model: String,
    pub vad_threshold: f32,
    pub energy_threshold: f32,
    pub min_silence_s: f32,
    #[serde(flatten)]
    pub chunked: VadChunkedConfig,
}

const PARAKEET: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            provider: "cpu".into(),
            num_threads: 2,
            encoder: format!("{PARAKEET}/encoder.int8.onnx"),
            decoder: format!("{PARAKEET}/decoder.int8.onnx"),
            joiner: format!("{PARAKEET}/joiner.int8.onnx"),
            tokens: format!("{PARAKEET}/tokens.txt"),
            model_type: "nemo_transducer".into(),
            vad: "silero".into(),
            vad_model: "silero_vad.onnx".into(),
            vad_threshold: 0.5,
            energy_threshold: 0.01,
            min_silence_s: 0.4,
            chunked: VadChunkedConfig::default(),
        }
    }
}

/// The translation (commit) LLM, served by its own llama-server process.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// "llama-server" | "none"
    pub backend: String,
    /// GGUF file (relative to models_dir).
    pub model: String,
    pub port: u16,
    /// chatml | llama3 | plain
    pub template: String,
    /// Qwen3.5 / Qwen3: an empty thinking block keeps the model in non-thinking mode.
    pub assistant_preamble: String,
    pub max_tokens: usize,
    pub system_prompt: String,
}

impl LlmConfig {
    fn qwen(model: &str, port: u16, max_tokens: usize) -> Self {
        Self {
            backend: "llama-server".into(),
            model: model.into(),
            port,
            template: "chatml".into(),
            assistant_preamble: "<think>\n\n</think>\n\n".into(),
            max_tokens,
            system_prompt: String::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.backend == "llama-server"
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self::qwen("Qwen3.5-2B-Q4_K_M.gguf", 8080, 160)
    }
}

/// Files a converted Opus-MT (CTranslate2) folder must contain.
pub const OPUS_MT_FILES: [&str; 4] = ["model.bin", "shared_vocabulary.json", "source.spm", "target.spm"];

/// Fast draft translation shown until the commit LLM has produced Chinese.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DraftConfig {
    /// "opus-mt" (Marian on CTranslate2) | "none"
    pub backend: String,
    /// backend = "opus-mt": converted model folder (relative to models_dir).
    pub model_dir: String,
    /// Target-language token the multi-target en-zh model needs; empty = none.
    pub target_token: String,
    pub beam_size: usize,
    pub threads: usize,
}

impl DraftConfig {
    /// Anything but "none" means Opus-MT (older settings may still say "llama-server").
    pub fn enabled(&self) -> bool {
        self.backend != "none"
    }
}

impl Default for DraftConfig {
    fn default() -> Self {
        Self {
            backend: "opus-mt".into(),
            model_dir: "opus-mt-en-zh-ct2".into(),
            target_token: ">>cmn_Hans<<".into(),
            beam_size: 1,
            threads: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaServerConfig {
    /// Path to `llama-server`; empty = search PATH and Homebrew locations.
    pub binary: String,
    /// CPU threads (-t / -tb). With every layer on the GPU the CPU only orchestrates;
    /// 0 = llama.cpp's default (all performance cores, which mostly spin and heat up).
    pub threads: i32,
    /// llama.cpp prompt caching (reuse of the prompt prefix + its RAM cache). Off by
    /// default: Qwen3.5's hybrid layers can't reuse it and the RAM cache grows to GBs.
    /// Worth turning on for full-attention models (Qwen2.5 / Qwen3).
    pub prompt_cache: bool,
    pub n_gpu_layers: i32,
    pub ctx_size: u32,
    pub startup_timeout_s: u64,
    pub extra_args: Vec<String>,
}

impl Default for LlamaServerConfig {
    fn default() -> Self {
        Self { binary: String::new(), threads: 2, prompt_cache: false, n_gpu_layers: 99, ctx_size: 4096, startup_timeout_s: 120, extra_args: vec![] }
    }
}

/// Trade a little latency for much less heat (fanless laptops).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    /// Speech recognition on the efficiency cores with fewer re-decodes of long
    /// sentences, and fewer, shorter LLM calls.
    pub power_saving: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DiarizationConfig {
    pub enabled: bool,
    /// Speaker embedding model (relative to models_dir). Thresholds in `tracker` are
    /// tuned for NeMo TitaNet; other models need other values.
    pub embedding_model: String,
    pub num_threads: i32,
    #[serde(flatten)]
    pub tracker: SpeakerTrackerConfig,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            embedding_model: "nemo_en_titanet_small.onnx".into(),
            num_threads: 2,
            tracker: SpeakerTrackerConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    /// Finished paragraphs kept above the current one; the subtitle window scrolls to them.
    pub history: usize,
    /// How sentences are grouped into paragraphs (pause length, speaker changes).
    pub paragraphs: ParagraphConfig,
    /// Chinese text size (px).
    pub font_size: u32,
    /// English text size (px).
    pub font_size_en: u32,
    /// Background opacity of the subtitle overlay, 0..1.
    pub opacity: f32,
    pub show_english: bool,
    /// "stacked" (English above Chinese) | "side_by_side" (English left, Chinese right)
    pub layout: String,
    /// JSONL event log for latency analysis; empty = off.
    pub log_path: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self { history: 2, paragraphs: ParagraphConfig::default(), font_size: 22, font_size_en: 15, opacity: 0.55, show_english: true, layout: "stacked".into(), log_path: String::new() }
    }
}

const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// Settings format; older files are upgraded when loaded (see `from_toml`).
    #[serde(default)]
    pub version: u32,
    /// Folder holding all model files; relative model paths are resolved against it.
    pub models_dir: String,
    pub max_sentence_words: usize,
    /// Translate into Chinese and show it. Off: English subtitles only, no translation
    /// models are loaded (no llama-server, no Opus-MT).
    pub translate: bool,
    pub audio: AudioConfig,
    pub asr: AsrConfig,
    pub draft: DraftConfig,
    pub commit: LlmConfig,
    pub llama_server: LlamaServerConfig,
    pub pipeline: ControllerConfig,
    pub performance: PerformanceConfig,
    pub diarization: DiarizationConfig,
    pub ui: UiConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            models_dir: "models".into(),
            max_sentence_words: 40,
            translate: true,
            audio: AudioConfig::default(),
            asr: AsrConfig::default(),
            draft: DraftConfig::default(),
            commit: LlmConfig::default(),
            llama_server: LlamaServerConfig::default(),
            pipeline: ControllerConfig::default(),
            performance: PerformanceConfig::default(),
            diarization: DiarizationConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

pub fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    PathBuf::from(p)
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_toml(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let mut cfg: Self = toml::from_str(text)?;
        if cfg.version < 1 {
            // version 1 raised the new-speaker threshold; keep values the user changed
            let t = &mut cfg.diarization.tracker.new_speaker_threshold;
            if (*t - 0.3).abs() < 1e-6 {
                *t = SpeakerTrackerConfig::default().new_speaker_threshold;
            }
        }
        cfg.version = CONFIG_VERSION;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, self.to_toml()?)?;
        Ok(())
    }

    /// Resolve a model path: absolute or `~/...` paths are used as-is, others are
    /// relative to `models_dir`.
    pub fn model_path(&self, p: &str) -> PathBuf {
        let path = expand_home(p);
        if path.is_absolute() {
            path
        } else {
            expand_home(&self.models_dir).join(path)
        }
    }

    /// Model files the current settings need, with a flag telling whether each exists.
    pub fn required_models(&self) -> Vec<(String, PathBuf, bool)> {
        // (label, path, files that must exist inside it; empty = the path itself)
        let mut need: Vec<(&str, &str, &[&str])> = vec![
            ("ASR encoder", &self.asr.encoder, &[]),
            ("ASR decoder", &self.asr.decoder, &[]),
            ("ASR joiner", &self.asr.joiner, &[]),
            ("ASR tokens", &self.asr.tokens, &[]),
        ];
        if self.asr.vad == "silero" {
            need.push(("Silero VAD", &self.asr.vad_model, &[]));
        }
        if self.translate && self.commit.enabled() {
            need.push(("commit LLM", &self.commit.model, &[]));
        }
        if self.translate && self.draft.enabled() {
            need.push(("draft model", &self.draft.model_dir, &OPUS_MT_FILES));
        }
        if self.diarization.enabled {
            need.push(("speaker model", &self.diarization.embedding_model, &[]));
        }
        let mut out = vec![];
        for (what, p, files) in need {
            let path = self.model_path(p);
            let ok = if files.is_empty() { path.exists() } else { files.iter().all(|f| path.join(f).exists()) };
            out.push((what.to_string(), path, ok));
        }
        out
    }
}
