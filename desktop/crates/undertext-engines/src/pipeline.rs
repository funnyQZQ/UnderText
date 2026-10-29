//! Build a complete pipeline (engine + servers + workers) from an `AppConfig`.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use undertext_core::board::Listener;
use undertext_core::config::{AppConfig, LlmConfig, SourceKind};
use undertext_core::diarization::Diarizer;
use undertext_core::eventlog::EventLog;
use undertext_core::translate::{IncrementalTranslator, PromptTemplate, PromptTranslator, Translator, DEFAULT_SYSTEM_PROMPT, COMPACT_SYSTEM_PROMPT};
use undertext_core::vad::{EnergyVad, Vad};
use undertext_core::vad_chunked::VadChunkedAsr;
use undertext_core::workers::{ThreadWorker, Worker};
use undertext_core::{Engine, SentenceTracker, SubtitleBoard, TranslationController};

use crate::audio::{AudioSource, MicSource, Pacing, WavSource};
use crate::llama::{find_llama_server, LlamaServerClient, LlamaServerProcess, ServerSpec};
use crate::sherpa::{SileroVad, SpeakerEmbedder, TransducerFiles, TransducerRecognizer};

/// Overrides typically set from the command line or the UI for one run.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub no_draft: bool,
    pub no_llm: bool,
}

pub struct Pipeline {
    pub engine: Engine,
    pub board: Arc<SubtitleBoard>,
    /// Kept alive for the duration of the run; killed on drop.
    pub servers: Vec<LlamaServerProcess>,
    pub warnings: Vec<String>,
}

impl Pipeline {
    pub fn shutdown(&mut self) {
        self.engine.shutdown();
        for s in &mut self.servers {
            s.stop();
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn translator(cfg: &LlmConfig, compact: bool, prompt_cache: bool) -> Result<PromptTranslator<LlamaServerClient>> {
    let template = PromptTemplate::by_name(&cfg.template).with_context(|| format!("unknown template {:?}", cfg.template))?;
    let mut client = LlamaServerClient::new(cfg.port);
    client.cache_prompt = prompt_cache;
    let mut t = PromptTranslator::new(client, template);
    t.assistant_preamble = cfg.assistant_preamble.clone();
    t.max_tokens = cfg.max_tokens;
    t.system_prompt = match (cfg.system_prompt.is_empty(), compact) {
        (false, _) => cfg.system_prompt.clone(),
        (true, true) => COMPACT_SYSTEM_PROMPT.into(),
        (true, false) => DEFAULT_SYSTEM_PROMPT.into(),
    };
    Ok(t)
}

#[cfg(feature = "opus-mt")]
fn opus_draft(cfg: &AppConfig) -> Result<Option<Arc<dyn Translator>>> {
    let d = &cfg.draft;
    let t = crate::opus::OpusMtTranslator::from_dir(&cfg.model_path(&d.model_dir), &d.target_token, d.beam_size, d.threads)?;
    Ok(Some(Arc::new(t)))
}

#[cfg(not(feature = "opus-mt"))]
fn opus_draft(_cfg: &AppConfig) -> Result<Option<Arc<dyn Translator>>> {
    Ok(None)
}

/// Translation scheduling; power saving calls the LLM after more new words and with one
/// context sentence (each call re-reads the whole prompt: Qwen3.5 can't reuse a prefix).
pub fn controller_config(cfg: &AppConfig) -> undertext_core::ControllerConfig {
    let mut c = cfg.pipeline.clone();
    c.skip_hidden_drafts = true;
    if cfg.performance.power_saving {
        c.commit_min_new_words = c.commit_min_new_words.max(5);
        c.context_sentences = c.context_sentences.min(1);
    }
    c
}

/// Build everything; `status` receives human-readable progress messages.
pub fn build(cfg: &AppConfig, opts: &RunOptions, listener: Option<Listener>, log: Option<Arc<EventLog>>, status: &dyn Fn(&str)) -> Result<Pipeline> {
    let mut warnings = Vec::new();
    let power_saving = cfg.performance.power_saving;
    if power_saving {
        // the calling thread runs the engine (speech recognition); onnxruntime's thread
        // pool, created below, inherits this
        undertext_core::power::prefer_efficiency_cores();
    }
    // translate = false: English only, no translation models at all
    let mut draft_on = cfg.translate && cfg.draft.enabled() && !opts.no_draft;
    let llm_on = cfg.translate && cfg.commit.enabled() && !opts.no_llm;

    // check model files first so the user gets one clear message
    let mut missing = Vec::new();
    for (what, path, ok) in cfg.required_models() {
        if ok {
            continue;
        }
        match what.as_str() {
            "draft model" if draft_on => {
                warnings.push(format!("draft model missing ({}), running without drafts", path.display()));
                draft_on = false;
            }
            "draft model" => {}
            "commit LLM" if !llm_on => {}
            _ => missing.push(format!("{what}: {}", path.display())),
        }
    }
    if !missing.is_empty() {
        bail!("missing model files (models_dir = {}):\n  {}", cfg.models_dir, missing.join("\n  "));
    }

    // translation LLM server
    let mut servers = Vec::new();
    if llm_on {
        let binary = find_llama_server(&cfg.llama_server.binary)?;
        let ls = &cfg.llama_server;
        let llm = &cfg.commit;
        {
            status(&format!("starting translation model ({})", llm.model));
            let model = cfg.model_path(&llm.model);
            servers.push(LlamaServerProcess::start(&ServerSpec {
                binary: &binary,
                model: &model,
                port: llm.port,
                n_gpu_layers: ls.n_gpu_layers,
                ctx_size: ls.ctx_size,
                threads: ls.threads,
                prompt_cache: ls.prompt_cache,
                extra_args: &ls.extra_args,
            })?);
        }
        status("loading the translation model ...");
        for s in &mut servers {
            s.wait_ready(Duration::from_secs(ls.startup_timeout_s))?;
        }
    }

    status("loading speech models ...");
    let a = &cfg.asr;
    let vad: Box<dyn Vad> = match a.vad.as_str() {
        "silero" => Box::new(SileroVad::new(&cfg.model_path(&a.vad_model), a.vad_threshold, a.min_silence_s)?),
        "energy" => Box::new(EnergyVad::new(16000, 512, a.energy_threshold, a.min_silence_s as f64, 0.2)),
        other => bail!("unknown vad {other:?}"),
    };
    let recognizer = TransducerRecognizer::new(
        TransducerFiles {
            encoder: &cfg.model_path(&a.encoder),
            decoder: &cfg.model_path(&a.decoder),
            joiner: &cfg.model_path(&a.joiner),
            tokens: &cfg.model_path(&a.tokens),
        },
        &a.model_type,
        a.num_threads,
        &a.provider,
    )?;
    let mut chunked = a.chunked.clone();
    chunked.adaptive_interval |= power_saving;
    let asr = VadChunkedAsr::new(vad, Box::new(recognizer), &chunked);

    let mut listener = listener;
    if let Some(log) = &log {
        let log = log.clone();
        let inner = listener.take();
        listener = Some(Box::new(move |kind, view| {
            log.on_board(kind, view);
            if let Some(f) = &inner {
                f(kind, view);
            }
        }));
    }
    let board = Arc::new(SubtitleBoard::new(500, listener));

    let draft: Option<Arc<dyn Translator>> = if !draft_on {
        None
    } else {
        match opus_draft(cfg)? {
            Some(t) => Some(t),
            None => {
                warnings.push("this build has no Opus-MT support (feature `opus-mt`), running without drafts".into());
                None
            }
        }
    };
    let llm: Option<Arc<dyn IncrementalTranslator>> = if llm_on { Some(Arc::new(translator(&cfg.commit, power_saving, cfg.llama_server.prompt_cache)?)) } else { None };
    let controller = TranslationController::new(
        board.clone(),
        draft,
        llm,
        controller_config(cfg),
        // drafts and voice prints may run a bit slower but cooler
        Some(Arc::new(ThreadWorker::background("draft"))),
        Some(Arc::new(ThreadWorker::new("llm"))),
    );
    let mut engine = Engine::new(Box::new(asr), SentenceTracker::new(cfg.max_sentence_words), controller, log);

    if cfg.diarization.enabled {
        let d = &cfg.diarization;
        let embedder = SpeakerEmbedder::new(&cfg.model_path(&d.embedding_model), d.num_threads)?;
        let worker: Arc<dyn Worker> = Arc::new(ThreadWorker::background("diarize"));
        engine = engine.with_diarizer(Arc::new(Diarizer::new(Box::new(embedder), d.tracker.clone())), worker);
    }

    for w in &warnings {
        log::warn!("{w}");
    }
    Ok(Pipeline { engine, board, servers, warnings })
}

/// Open the configured audio source (call on the thread that will read it).
pub fn open_source(cfg: &AppConfig, stop: Arc<AtomicBool>) -> Result<Box<dyn AudioSource>> {
    let au = &cfg.audio;
    Ok(match au.source {
        SourceKind::Mic => Box::new(MicSource::open(&au.device, au.chunk_ms, stop)?),
        SourceKind::Wav => {
            if au.wav_path.is_empty() {
                bail!("no WAV file selected");
            }
            let pacing = if au.play { Pacing::Play { output_device: au.output_device.clone() } } else { Pacing::Realtime };
            Box::new(WavSource::open(Path::new(&undertext_core::config::expand_home(&au.wav_path)), au.chunk_ms, pacing, stop)?)
        }
    })
}
