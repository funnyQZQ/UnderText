//! `undertext-cli`: the UnderText pipeline without the UI.
//!
//!   undertext-cli init-config config.toml            write a default config
//!   undertext-cli check -c config.toml               models + llama-server present?
//!   undertext-cli devices                            list audio devices
//!   undertext-cli run -c config.toml --wav talk.wav --play --log run.jsonl
//!   undertext-cli report run.jsonl                   latency per sentence
//!   undertext-cli soak / soak-report / compare       long runs, model comparison
//!   undertext-cli diarize / bench-asr                speaker and speed diagnostics

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use undertext_core::config::{AppConfig, SourceKind};
use undertext_core::eventlog::EventLog;
use undertext_core::report::{latency_report, summarize};
use undertext_core::{ParagraphConfig, SubtitleBoard};
use undertext_engines::audio::{list_input_devices, list_output_devices, SourceIter};
use undertext_engines::llama::find_llama_server;
use undertext_engines::pipeline::{self, RunOptions};

#[derive(Parser)]
#[command(name = "undertext-cli", about = "Local real-time English -> Chinese subtitles")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a default config file.
    InitConfig { path: PathBuf },
    /// Check that model files and llama-server are available.
    Check {
        #[arg(short, long)]
        config: Option<PathBuf>,
        #[arg(long)]
        models: Option<String>,
    },
    /// List audio input and output devices.
    Devices,
    /// Run the pipeline and show subtitles in the terminal.
    Run {
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Read a WAV file instead of the configured source.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// With --wav: play it; subtitles follow playback.
        #[arg(long)]
        play: bool,
        /// Input device name substring (e.g. BlackHole).
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        models: Option<String>,
        #[arg(long)]
        no_draft: bool,
        #[arg(long)]
        no_llm: bool,
        #[arg(long)]
        diarize: bool,
        /// English subtitles only (no translation models).
        #[arg(long)]
        no_translate: bool,
        /// JSONL event log for latency analysis.
        #[arg(long)]
        log: Option<PathBuf>,
    },
    /// Per-sentence latency report from an event log.
    Report { log: PathBuf },
    /// Long run in real time (e.g. a recording of an hour or more) that records every
    /// stage's speed, the backlog behind live audio and the macOS thermal state.
    Soak {
        #[arg(short, long)]
        config: Option<PathBuf>,
        #[arg(long)]
        models: Option<String>,
        wav: PathBuf,
        /// stop after this many minutes (default: whole file)
        #[arg(long)]
        minutes: Option<f64>,
        /// don't play the audio (played by default)
        #[arg(long)]
        no_play: bool,
        /// translation (commit) model file in models_dir (default: config)
        #[arg(long)]
        commit_model: Option<String>,
        /// no speaker labels (on by default)
        #[arg(long)]
        no_diarize: bool,
        /// turn power saving on (default: as in the config)
        #[arg(long)]
        power_saving: bool,
        /// event log (default: logs/soak-<time>.jsonl next to the models folder)
        #[arg(long)]
        log: Option<PathBuf>,
    },
    /// Compare two soak runs on the same audio (e.g. two translation models): sentences
    /// side by side with differences marked, plus cost and latency, as an HTML page.
    Compare {
        a: PathBuf,
        b: PathBuf,
        /// HTML output (default: next to `b`, named compare.html)
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Analyse a soak log: per time bucket speed of every stage, start vs end verdict.
    SoakReport {
        log: PathBuf,
        /// bucket length in minutes
        #[arg(long, default_value_t = 5.0)]
        bucket: f64,
        /// also write the table as CSV
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// Speaker diagnosis on a WAV: per segment the online label (as in the app), the
    /// similarity to every known speaker, and voice changes inside long segments.
    /// `--out` writes everything (incl. embeddings) as JSON for offline analysis.
    Diarize {
        #[arg(short, long)]
        config: Option<PathBuf>,
        #[arg(long)]
        models: Option<String>,
        wav: PathBuf,
        /// only the first N seconds
        #[arg(long)]
        max_seconds: Option<f64>,
        /// speaker embedding model file (default: from config)
        #[arg(long)]
        model: Option<String>,
        /// new-speaker threshold (default: from config)
        #[arg(long)]
        threshold: Option<f32>,
        #[arg(long)]
        out: Option<PathBuf>,
        /// replay in audio time like the app: print when each label (provisional or
        /// final) appears, from this many seconds on
        #[arg(long)]
        live_from: Option<f64>,
    },
    /// Time one ASR decode of the first N seconds of a WAV (the pipeline re-decodes the
    /// growing segment every `decode_interval_s`, so this must stay well below it).
    BenchAsr {
        #[arg(short, long)]
        config: Option<PathBuf>,
        #[arg(long)]
        models: Option<String>,
        wav: PathBuf,
        /// onnxruntime threads (default: from config)
        #[arg(long)]
        threads: Option<i32>,
        /// also load Opus-MT and translate in a loop on another thread (CPU contention test)
        #[arg(long)]
        with_opus: bool,
    },
}

fn load(config: &Option<PathBuf>) -> Result<AppConfig> {
    match config {
        Some(p) => AppConfig::load(p),
        None => Ok(AppConfig::default()),
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    match Cli::parse().cmd {
        Cmd::InitConfig { path } => {
            AppConfig::default().save(&path)?;
            println!("wrote {}", path.display());
        }
        Cmd::Check { config, models } => {
            let mut cfg = load(&config)?;
            if let Some(m) = models {
                cfg.models_dir = m;
            }
            check(&cfg)
        }
        Cmd::Devices => {
            println!("input devices:");
            for d in list_input_devices()? {
                println!("  {}{}", d.name, if d.is_default { "  (default)" } else { "" });
            }
            println!("output devices:");
            for d in list_output_devices()? {
                println!("  {}{}", d.name, if d.is_default { "  (default)" } else { "" });
            }
        }
        Cmd::Run { config, wav, play, device, models, no_draft, no_llm, diarize, no_translate, log } => {
            let mut cfg = load(&config)?;
            if let Some(w) = wav {
                cfg.audio.source = SourceKind::Wav;
                cfg.audio.wav_path = w.to_string_lossy().into();
                cfg.audio.play = play;
            }
            if let Some(d) = device {
                cfg.audio.source = SourceKind::Mic;
                cfg.audio.device = d;
            }
            if let Some(m) = models {
                cfg.models_dir = m;
            }
            if diarize {
                cfg.diarization.enabled = true;
            }
            if no_translate {
                cfg.translate = false;
            }
            run(&cfg, &RunOptions { no_draft, no_llm }, log)?;
        }
        Cmd::Report { log } => report(&log)?,
        Cmd::Soak { config, models, wav, minutes, no_play, commit_model, no_diarize, power_saving, log } => {
            let mut cfg = load(&config)?;
            if let Some(m) = commit_model {
                cfg.commit.model = m;
            }
            if let Some(m) = models {
                cfg.models_dir = m;
            }
            cfg.audio.source = SourceKind::Wav;
            cfg.audio.wav_path = wav.to_string_lossy().into();
            cfg.audio.play = !no_play;
            cfg.diarization.enabled = !no_diarize;
            cfg.performance.power_saving |= power_saving;
            soak(&cfg, minutes, log)?;
        }
        Cmd::SoakReport { log, bucket, csv } => soak_report(&log, bucket, csv)?,
        Cmd::Compare { a, b, out } => {
            use undertext_core::compare::{render_html, render_text, Run};
            let ra = Run::from_records("A", &read_log(&a)?);
            let rb = Run::from_records("B", &read_log(&b)?);
            let out = out.unwrap_or_else(|| b.with_file_name("compare.html"));
            std::fs::write(&out, render_html(&ra, &rb))?;
            print!("{}", render_text(&ra, &rb));
            println!("wrote {} (open it in a browser)", out.display());
        }
        Cmd::Diarize { config, models, wav, max_seconds, model, threshold, out, live_from } => {
            let mut cfg = load(&config)?;
            if let Some(m) = models {
                cfg.models_dir = m;
            }
            if let Some(m) = model {
                cfg.diarization.embedding_model = m;
            }
            if let Some(t) = threshold {
                cfg.diarization.tracker.new_speaker_threshold = t;
            }
            match live_from {
                Some(from) => diarize_live(&cfg, &wav, max_seconds, from)?,
                None => diarize(&cfg, &wav, max_seconds, out)?,
            }
        }
        Cmd::BenchAsr { config, models, wav, threads, with_opus } => {
            let mut cfg = load(&config)?;
            if let Some(m) = models {
                cfg.models_dir = m;
            }
            if let Some(t) = threads {
                cfg.asr.num_threads = t;
            }
            bench_asr(&cfg, &wav, with_opus)?;
        }
    }
    Ok(())
}

fn check(cfg: &AppConfig) {
    println!("models_dir: {}", cfg.models_dir);
    for (what, path, ok) in cfg.required_models() {
        println!("  [{}] {what}: {}", if ok { "ok" } else { "MISSING" }, path.display());
    }
    match find_llama_server(&cfg.llama_server.binary) {
        Ok(p) => println!("  [ok] llama-server: {}", p.display()),
        Err(e) => println!("  [MISSING] {e}"),
    }
}

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

fn render(board: &SubtitleBoard, history: usize, para_cfg: &ParagraphConfig, chinese: bool) -> String {
    let mut out = String::new();
    for p in board.paragraphs(para_cfg, history + 1) {
        let tag = p.speaker.as_ref().map(|s| format!("[{s}] ")).unwrap_or_default();
        let (mut en, mut zh) = (String::new(), String::new());
        for d in &p.sentences {
            for (stable, unstable, sep) in [(&d.en_stable, &d.en_unstable, " "), (&d.zh_stable, &d.zh_unstable, "")] {
                let line = if sep.is_empty() { &mut zh } else { &mut en };
                if !line.is_empty() && !stable.is_empty() {
                    line.push_str(sep);
                }
                line.push_str(stable);
                if !unstable.is_empty() {
                    let s = if line.is_empty() { "" } else { sep };
                    line.push_str(&format!("{DIM}{s}{unstable}{RESET}"));
                }
            }
        }
        if chinese {
            out.push_str(&format!("   {tag}{en}\n中 {zh}\n\n"));
        } else {
            out.push_str(&format!("{tag}{en}\n\n"));
        }
    }
    out
}

fn run(cfg: &AppConfig, opts: &RunOptions, log_path: Option<PathBuf>) -> Result<()> {
    let log = match &log_path {
        Some(p) => Some(Arc::new(EventLog::new(Box::new(std::fs::File::create(p).with_context(|| p.display().to_string())?)))),
        None => None,
    };
    let mut pipeline = pipeline::build(cfg, opts, None, log, &|s| eprintln!("{s}"))?;
    for w in &pipeline.warnings {
        eprintln!("warning: {w}");
    }
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;

    // terminal renderer
    let board = pipeline.board.clone();
    let done = Arc::new(AtomicBool::new(false));
    let d = done.clone();
    let (history, para_cfg, chinese) = (cfg.ui.history, cfg.ui.paragraphs.clone(), cfg.translate);
    let renderer = std::thread::spawn(move || {
        let mut seen = u64::MAX;
        loop {
            let finished = d.load(Ordering::SeqCst);
            if board.version() != seen {
                seen = board.version();
                let text = render(&board, history, &para_cfg, chinese);
                // paragraphs wrap over several terminal rows: redraw the whole screen
                let mut so = std::io::stdout().lock();
                let _ = write!(so, "\x1b[H\x1b[2J{text}");
                let _ = so.flush();
            }
            if finished {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    });

    eprintln!("listening (Ctrl+C to stop) ...");
    let mut source = pipeline::open_source(cfg, stop.clone())?;
    pipeline.engine.run(SourceIter { source: source.as_mut(), stop });
    done.store(true, Ordering::SeqCst);
    let _ = renderer.join();
    eprintln!("\n{}", pipeline.engine.stats_summary());
    pipeline.shutdown();
    Ok(())
}

/// Feeds the audio through ASR and the diarizer the way the app does (provisional labels
/// while a segment is spoken, final ones when it ends) and prints, in audio time, when
/// each label appears.
fn diarize_live(cfg: &AppConfig, wav: &PathBuf, max_seconds: Option<f64>, from: f64) -> Result<()> {
    use std::collections::VecDeque;
    use undertext_core::asr::AsrBackend;
    use undertext_core::diarization::{Diarizer, SpeakerSpan};
    use undertext_core::vad_chunked::VadChunkedAsr;
    use undertext_engines::sherpa::{SileroVad, SpeakerEmbedder, TransducerFiles, TransducerRecognizer};

    let a = &cfg.asr;
    let vad = SileroVad::new(&cfg.model_path(&a.vad_model), a.vad_threshold, a.min_silence_s)?;
    let rec = TransducerRecognizer::new(
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
    let mut asr = VadChunkedAsr::new(Box::new(vad), Box::new(rec), &a.chunked);
    let d = &cfg.diarization;
    let diarizer = Diarizer::new(Box::new(SpeakerEmbedder::new(&cfg.model_path(&d.embedding_model), d.num_threads)?), d.tracker.clone());
    let mut audio = undertext_engines::audio::read_wav(wav)?;
    if let Some(m) = max_seconds {
        audio.truncate((m * 16000.0) as usize);
    }
    let fmt = |spans: &[SpeakerSpan], start: f64, dur: f64| {
        spans
            .iter()
            .map(|s| {
                let l = s.label.clone().unwrap_or_else(|| "-".into());
                format!("{l} {:.1}-{:.1}", start + s.start as f64 * dur, start + s.end as f64 * dur)
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let (win, every) = (24000usize, 1.5f64);
    let mut recent: VecDeque<f32> = VecDeque::new();
    let mut fed = 0u64;
    let mut last_peek: Option<(usize, f64)> = None;
    let mut seg_start = std::collections::HashMap::new();
    let mut last_line = String::new();
    println!("{:>7}  {:<5} {:<8} spans (audio seconds)", "time", "seg", "kind");
    for chunk in audio.chunks(1600) {
        recent.extend(chunk);
        let extra = recent.len().saturating_sub(5 * 16000);
        recent.drain(..extra);
        fed += chunk.len() as u64;
        let now = fed as f64 / 16000.0;
        for u in asr.accept(chunk) {
            seg_start.insert(u.segment_id, u.start_s);
            if u.is_final {
                let Some(samples) = &u.samples else { continue };
                for (seg, spans) in diarizer.label_segment(u.segment_id, samples) {
                    let st = seg_start.get(&seg).copied().unwrap_or(u.start_s);
                    let dur = if seg == u.segment_id { samples.len() as f64 / 16000.0 } else { f64::NAN };
                    if now >= from {
                        let kind = if seg == u.segment_id { "final" } else { "relabel" };
                        println!("{now:7.1}  {seg:<5} {kind:<8} {}", fmt(&spans, st, dur));
                    }
                }
                continue;
            }
            let elapsed = u.end_s - u.start_s;
            if elapsed < 1.5 || last_peek.is_some_and(|(s, at)| s == u.segment_id && elapsed - at < every) {
                continue;
            }
            let end = (u.end_s * 16000.0) as u64;
            let first = fed - recent.len() as u64;
            if end > fed || end < first + win as u64 {
                continue;
            }
            let i = (end - win as u64 - first) as usize;
            let window: Vec<f32> = recent.range(i..i + win).copied().collect();
            last_peek = Some((u.segment_id, elapsed));
            let line = match diarizer.peek(u.segment_id, elapsed, &window) {
                Some(spans) => fmt(&spans, u.start_s, elapsed),
                None => "(no label)".into(),
            };
            if now >= from && line != last_line {
                println!("{now:7.1}  {:<5} {:<8} {line}", u.segment_id, "peek");
                last_line = line;
            }
        }
    }
    Ok(())
}

fn diarize(cfg: &AppConfig, wav: &PathBuf, max_seconds: Option<f64>, out: Option<PathBuf>) -> Result<()> {
    use undertext_core::asr::AsrBackend;
    use undertext_core::diarization::{Diarizer, Embedder};
    use undertext_core::vad_chunked::{VadChunkedAsr, VadChunkedConfig};
    use undertext_engines::sherpa::{SileroVad, SpeakerEmbedder, TransducerFiles, TransducerRecognizer};
    use serde_json::json;
    use std::time::Instant;

    let a = &cfg.asr;
    let vad = SileroVad::new(&cfg.model_path(&a.vad_model), a.vad_threshold, a.min_silence_s)?;
    let rec = TransducerRecognizer::new(
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
    // only final decodes are needed here
    let chunked = VadChunkedConfig { decode_interval_s: 1e9, ..a.chunked.clone() };
    let mut asr = VadChunkedAsr::new(Box::new(vad), Box::new(rec), &chunked);
    let d = &cfg.diarization;
    let probe = SpeakerEmbedder::new(&cfg.model_path(&d.embedding_model), d.num_threads)?;
    let diarizer = Diarizer::new(Box::new(SpeakerEmbedder::new(&cfg.model_path(&d.embedding_model), d.num_threads)?), d.tracker.clone());

    let mut audio = undertext_engines::audio::read_wav(wav)?;
    if let Some(m) = max_seconds {
        audio.truncate((m * 16000.0) as usize);
    }
    eprintln!("{:.0} s of audio, model {}, {:?}", audio.len() as f64 / 16000.0, d.embedding_model, d.tracker);
    let mut finals = Vec::new();
    for chunk in audio.chunks(16000) {
        finals.extend(asr.accept(chunk).into_iter().filter(|u| u.is_final));
    }
    finals.extend(asr.flush().into_iter().filter(|u| u.is_final));

    let unit = |v: Vec<f32>| {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect::<Vec<f32>>()
    };
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut records = Vec::new();
    let mut busy = 0.0;
    println!("{:>7} {:>5}  {:<14} {:<26} text", "start", "dur", "speaker(s)", "sims to voices 1..");
    for u in &finals {
        let Some(samples) = &u.samples else { continue };
        let dur = samples.len() as f64 / 16000.0;
        let emb = unit(probe.embed(samples));
        let sims: Vec<f32> = diarizer.centroids().iter().map(|c| dot(&emb, c)).collect();
        let t0 = Instant::now();
        let mut updates = diarizer.label_segment(u.segment_id, samples);
        busy += t0.elapsed().as_secs_f64();
        let relabeled: Vec<usize> = updates[1..].iter().map(|x| x.0).collect();
        let spans = updates.swap_remove(0).1;
        let who = spans
            .iter()
            .map(|s| s.label.clone().unwrap_or_else(|| "-".into()) + if s.new { "*" } else { "" })
            .collect::<Vec<_>>()
            .join(">");
        let sims_txt = sims.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>().join(" ");
        println!("{:7.1} {:5.1}  {:<14} {:<26} {}", u.start_s, dur, who, sims_txt, u.committed.chars().take(70).collect::<String>());
        if !relabeled.is_empty() {
            println!("{:>14}  speaker confirmed, {} earlier segment(s) relabeled", "", relabeled.len());
        }
        records.push(json!({"start": u.start_s, "end": u.end_s, "dur": dur, "text": u.committed, "spans": spans,
                            "sims_before": sims, "embedding": emb}));
    }
    println!(
        "\n{} segments, {} speakers (* = new voice, - = not confirmed yet, A>B = turn change inside one segment); speaker work {:.1} s for {:.0} s of audio",
        records.len(),
        diarizer.num_speakers(),
        busy,
        audio.len() as f64 / 16000.0
    );
    if let Some(p) = out {
        std::fs::write(&p, serde_json::to_string(&json!({"model": d.embedding_model, "tracker": d.tracker, "segments": records}))?)?;
        println!("wrote {}", p.display());
    }
    Ok(())
}

fn bench_asr(cfg: &AppConfig, wav: &PathBuf, with_opus: bool) -> Result<()> {
    use undertext_core::asr::OfflineRecognizer;
    use undertext_engines::sherpa::{TransducerFiles, TransducerRecognizer};
    use std::time::Instant;
    let a = &cfg.asr;
    let rec = TransducerRecognizer::new(
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
    let audio = undertext_engines::audio::read_wav(wav)?;
    let stop = Arc::new(AtomicBool::new(false));
    let mut busy = None;
    if with_opus {
        #[cfg(feature = "opus-mt")]
        {
            let d = &cfg.draft;
            let tr = undertext_engines::opus::OpusMtTranslator::from_dir(&cfg.model_path(&d.model_dir), &d.target_token, d.beam_size, d.threads)?;
            let s = stop.clone();
            busy = Some(std::thread::spawn(move || {
                use undertext_core::translate::Translator;
                let mut n = 0;
                while !s.load(Ordering::SeqCst) {
                    let _ = tr.translate("So what will happen if he works out for one hundred days?", &[]);
                    n += 1;
                }
                n
            }));
        }
    }
    let _ = rec.transcribe(&audio[..audio.len().min(16000)]); // warm-up
    println!("asr threads = {}, provider = {}{}", a.num_threads, a.provider, if busy.is_some() { ", Opus-MT busy on another thread" } else { "" });
    for secs in [2usize, 5, 10, 15, 20] {
        let n = (secs * 16000).min(audio.len());
        let mut times: Vec<f64> = (0..3)
            .map(|_| {
                let t0 = Instant::now();
                let _ = rec.transcribe(&audio[..n]);
                t0.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("{secs:3}s audio: {:7.0} ms", times[1]);
    }
    stop.store(true, Ordering::SeqCst);
    if let Some(h) = busy {
        println!("(Opus-MT translations meanwhile: {})", h.join().unwrap_or(0));
    }
    Ok(())
}

/// macOS thermal state via ProcessInfo (0 nominal .. 3 critical); None elsewhere.
fn thermal_state() -> Option<&'static str> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("osascript")
        .args(["-l", "JavaScript", "-e", "ObjC.import('Foundation'); $.NSProcessInfo.processInfo.thermalState"])
        .output()
        .ok()?;
    let n: usize = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    ["nominal", "fair", "serious", "critical"].get(n).copied()
}

fn soak(cfg: &AppConfig, minutes: Option<f64>, log_path: Option<PathBuf>) -> Result<()> {
    let log_path = log_path.unwrap_or_else(|| {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let dir = undertext_core::config::expand_home(&cfg.models_dir).join("../logs");
        let _ = std::fs::create_dir_all(&dir);
        dir.join(format!("soak-{secs}.jsonl"))
    });
    let log = Arc::new(EventLog::new(Box::new(std::fs::File::create(&log_path).with_context(|| log_path.display().to_string())?)));
    log.write(
        "soak_config",
        serde_json::json!({"wav": cfg.audio.wav_path, "power_saving": cfg.performance.power_saving,
                           "diarize": cfg.diarization.enabled, "translate": cfg.translate,
                           "commit_model": cfg.commit.model, "draft": cfg.draft.backend,
                           "prompt_cache": cfg.llama_server.prompt_cache}),
    );
    let mut pipeline = pipeline::build(cfg, &RunOptions::default(), None, Some(log.clone()), &|s| eprintln!("{s}"))?;
    for w in &pipeline.warnings {
        eprintln!("warning: {w}");
    }
    pipeline.engine.set_stats_every(Duration::from_secs(10));
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;

    // progress: how much audio has gone in, and how far behind real time we are
    let total_s = undertext_engines::audio::wav_duration_s(std::path::Path::new(&undertext_core::config::expand_home(&cfg.audio.wav_path)))?;
    let target_s = minutes.map_or(total_s, |m| (m * 60.0).min(total_s));
    let fed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let first_chunk: Arc<std::sync::Mutex<Option<std::time::Instant>>> = Arc::default();

    let done = Arc::new(AtomicBool::new(false));
    let (d, st, lg, fd, fc, board) = (done.clone(), stop.clone(), log.clone(), fed.clone(), first_chunk.clone(), pipeline.board.clone());
    let monitor = std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let (mut next_thermal, mut next_line) = (Duration::ZERO, Duration::from_secs(300));
        let mut state: Option<&str> = None;
        let mmss = |s: f64| format!("{:02}:{:02}", (s / 60.0) as u64, (s % 60.0) as u64);
        while !d.load(Ordering::SeqCst) {
            let el = t0.elapsed();
            if el >= next_thermal {
                state = thermal_state();
                if let Some(s) = state {
                    lg.write("thermal", serde_json::json!({"state": s}));
                }
                next_thermal += Duration::from_secs(60);
            }
            let audio_s = fd.load(Ordering::Relaxed) as f64 / 16000.0;
            let lag = fc.lock().unwrap().map_or(0.0, |t| (t.elapsed().as_secs_f64() - audio_s).max(0.0));
            let frac = (audio_s / target_s).clamp(0.0, 1.0);
            let bar: String = (0..30).map(|i| if (i as f64) < frac * 30.0 { '█' } else { '░' }).collect();
            let sentences = board.snapshot(Some(1)).first().map_or(0, |v| v.sid + 1);
            let line = format!(
                "[{bar}] {:5.1}%  {} / {}  {} left | behind {:.1} s | {} sentences | thermal {}",
                frac * 100.0,
                mmss(audio_s),
                mmss(target_s),
                mmss((target_s - audio_s).max(0.0) + lag),
                lag,
                sentences,
                state.unwrap_or("n/a")
            );
            if el >= next_line {
                // keep a line every 5 minutes as history
                eprintln!("\r{line}");
                next_line += Duration::from_secs(300);
            } else {
                eprint!("\r{line}\x1b[K");
            }
            if audio_s >= target_s && minutes.is_some() {
                st.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        eprintln!();
    });

    eprintln!(
        "soak test: {} ({:.1} min, model {}, play {}, speakers {}, power saving {}), Ctrl+C to stop early\nlog: {}",
        cfg.audio.wav_path,
        target_s / 60.0,
        cfg.commit.model,
        cfg.audio.play,
        cfg.diarization.enabled,
        cfg.performance.power_saving,
        log_path.display()
    );
    let mut source = pipeline::open_source(cfg, stop.clone())?;
    let chunks = SourceIter { source: source.as_mut(), stop }.inspect(|c| {
        first_chunk.lock().unwrap().get_or_insert_with(std::time::Instant::now);
        fed.fetch_add(c.len() as u64, Ordering::Relaxed);
    });
    pipeline.engine.run(chunks);
    done.store(true, Ordering::SeqCst);
    let _ = monitor.join();
    pipeline.shutdown();
    drop(pipeline);
    eprintln!("\n{}", log_path.display());
    soak_report(&log_path, 5.0, None)
}

fn read_log(path: &PathBuf) -> Result<Vec<serde_json::Value>> {
    let f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    Ok(std::io::BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect())
}

fn soak_report(path: &PathBuf, bucket_min: f64, csv: Option<PathBuf>) -> Result<()> {
    use undertext_core::soak;
    let records = read_log(path)?;
    let bs = soak::buckets(&records, bucket_min * 60.0);
    if bs.is_empty() {
        anyhow::bail!("no stats records in {} (logs from before the soak test have none)", path.display());
    }
    if let Some(c) = records.iter().find(|r| r["kind"] == "soak_config") {
        println!("config: power_saving={} diarize={} translate={}", c["power_saving"], c["diarize"], c["translate"]);
    }
    println!(
        "\nper {bucket_min} min: ASR ms/s = recognition time per second of audio decoded; draft / LLM = ms per call;\n\
         ms/token = LLM time per prompt token; lag = how far behind the live audio; 1st zh / final zh = median latency (s)\n"
    );
    print!("{}", soak::render_table(&bs));
    println!("\nstart vs end (\"!\" = more than 25% slower):");
    for line in soak::verdict(&bs, 2) {
        println!("  {line}");
    }
    if let Some(p) = csv {
        std::fs::write(&p, soak::to_csv(&bs))?;
        println!("\nwrote {}", p.display());
    }
    Ok(())
}

fn report(path: &PathBuf) -> Result<()> {
    let f = std::fs::File::open(path)?;
    let records: Vec<serde_json::Value> = std::io::BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    let rows = latency_report(&records);
    let f = |x: Option<f64>| x.map(|v| format!("{v:5.2}s")).unwrap_or_else(|| "   –  ".into());
    for r in &rows {
        let tag = r.speaker.as_ref().map(|s| format!("[{s}] ")).unwrap_or_default();
        println!("#{:<3} first-zh {}  final {}  | {tag}{}\n{:39}| {}", r.sid, f(r.first_zh_lag()), f(r.final_lag()), r.en, "", r.zh);
    }
    let first: Vec<f64> = rows.iter().filter_map(|r| r.first_zh_lag()).collect();
    let fin: Vec<f64> = rows.iter().filter_map(|r| r.final_lag()).collect();
    println!("\nfirst Chinese after first English : {}", summarize(&first));
    println!("final Chinese after English closes: {}", summarize(&fin));
    if let Some(st) = records.iter().rev().find(|r| r["kind"] == "stats") {
        println!("\n{}", undertext_core::metrics::summarize(st, st["elapsed_s"].as_f64().unwrap_or(1.0)));
    }
    Ok(())
}
