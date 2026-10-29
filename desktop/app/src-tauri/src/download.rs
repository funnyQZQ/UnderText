//! First-run download of the models (and llama-server when it isn't installed).
//!
//! Files come from Hugging Face and GitHub releases. Before starting, both Hugging Face and
//! its mirror hf-mirror.com are tried with a small request and the faster one is used; if
//! it fails or stalls partway, the download moves to the other one. Each file is fetched
//! with the system `curl` into `<name>.part`, which is resumed on the next attempt (also
//! from the other server) and renamed when complete. Progress goes to the settings window
//! as `download` events.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use undertext_core::config::{AppConfig, OPUS_MT_FILES};

const HF: &str = "https://huggingface.co";
const HF_MIRROR: &str = "https://hf-mirror.com";
/// Fetched (first MB only) from each Hugging Face server to see which one is faster.
const PROBE: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/encoder.int8.onnx";
/// A transfer slower than this for `STALL_S` seconds is dropped and the next server tried.
const STALL_BYTES_PER_S: u32 = 20_000;
const STALL_S: u32 = 30;
const LLAMA_BUILD: &str = "b11785";
const LLAMA_SIZE: u64 = 16_122_351;

/// Where a file can be fetched from, tried in order.
enum Source {
    /// (repository, file) on Hugging Face; tried on both servers, faster one first.
    Hf(&'static str, &'static str),
    Url(&'static str),
}

/// Known downloads, by path relative to the models folder: (sources, size in bytes).
fn catalog(rel: &str) -> Option<(Vec<Source>, u64)> {
    use Source::*;
    const PARAKEET: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
    const OPUS: &str = "gaudi/opus-mt-en-zh-ctranslate2";
    Some(match rel {
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx" => (vec![Hf(PARAKEET, "encoder.int8.onnx")], 652_184_296),
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx" => (vec![Hf(PARAKEET, "decoder.int8.onnx")], 7_257_753),
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx" => (vec![Hf(PARAKEET, "joiner.int8.onnx")], 1_739_080),
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt" => (vec![Hf(PARAKEET, "tokens.txt")], 9_384),
        "silero_vad.onnx" => (
            vec![Url("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"), Hf("csukuangfj/vad", "silero_vad.onnx")],
            643_854,
        ),
        "nemo_en_titanet_small.onnx" => (
            vec![
                Hf("csukuangfj/speaker-embedding-models", "nemo_en_titanet_small.onnx"),
                Url("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"),
            ],
            40_257_283,
        ),
        "Qwen3.5-2B-Q4_K_M.gguf" => (vec![Hf("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf")], 1_280_835_840),
        "opus-mt-en-zh-ct2/model.bin" => (vec![Hf(OPUS, "model.bin")], 155_502_615),
        "opus-mt-en-zh-ct2/shared_vocabulary.json" => (vec![Hf(OPUS, "shared_vocabulary.json")], 1_303_887),
        "opus-mt-en-zh-ct2/source.spm" => (vec![Hf(OPUS, "source.spm")], 806_435),
        "opus-mt-en-zh-ct2/target.spm" => (vec![Hf(OPUS, "target.spm")], 804_600),
        _ => return None,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanItem {
    /// What it is for, shown to the user.
    pub label: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub items: Vec<PlanItem>,
    pub total: u64,
    /// Missing files the app doesn't know how to download (custom model names).
    pub unknown: Vec<String>,
}

enum Url {
    /// Path on a Hugging Face server ("repo/resolve/main/file").
    Hf(String),
    Full(String),
}

struct Job {
    label: String,
    urls: Vec<Url>,
    dest: PathBuf,
    size: u64,
    /// A .tar.gz whose single top folder is unpacked into `dest` (a directory).
    unpack: bool,
}

fn group_label(rel: &str) -> &'static str {
    if rel.starts_with("sherpa-onnx-nemo-parakeet") {
        "Speech recognition (Parakeet-TDT)"
    } else if rel == "silero_vad.onnx" {
        "Voice activity (Silero VAD)"
    } else if rel.contains("titanet") {
        "Speaker labels (TitaNet)"
    } else if rel.ends_with(".gguf") {
        "Translation (Qwen3.5-2B)"
    } else if rel.starts_with("opus-mt") {
        "Quick drafts (Opus-MT)"
    } else {
        "Model file"
    }
}

/// Files still missing for `cfg`, plus llama-server when translation needs it and none is
/// installed (`llama_dir` is where it would be unpacked).
fn jobs(cfg: &AppConfig, llama_dir: &Path) -> (Vec<Job>, Vec<String>) {
    let models = undertext_core::config::expand_home(&cfg.models_dir);
    let mut wanted: Vec<PathBuf> = Vec::new();
    for (what, path, ok) in cfg.required_models() {
        if ok {
            continue;
        }
        if what == "draft model" {
            wanted.extend(OPUS_MT_FILES.iter().map(|f| path.join(f)).filter(|p| !p.exists()));
        } else {
            wanted.push(path);
        }
    }
    // the speaker model is small; fetch it too so speaker labels can be switched on later
    if !cfg.diarization.enabled {
        let p = cfg.model_path(&cfg.diarization.embedding_model);
        if !p.exists() {
            wanted.push(p);
        }
    }
    let mut out = Vec::new();
    let mut unknown = Vec::new();
    for path in wanted {
        let rel = path.strip_prefix(&models).map(|r| r.to_string_lossy().replace('\\', "/"));
        match rel.ok().and_then(|r| catalog(&r).map(|c| (r, c))) {
            Some((rel, (sources, size))) => out.push(Job {
                label: group_label(&rel).into(),
                urls: sources
                    .iter()
                    .map(|s| match s {
                        Source::Hf(repo, file) => Url::Hf(format!("{repo}/resolve/main/{file}")),
                        Source::Url(u) => Url::Full(u.to_string()),
                    })
                    .collect(),
                dest: path,
                size,
                unpack: false,
            }),
            None => unknown.push(path.display().to_string()),
        }
    }
    if cfg.translate && cfg.commit.enabled() && undertext_engines::llama::find_llama_server(&cfg.llama_server.binary).is_err() {
        out.push(Job {
            label: "Translation runtime (llama-server)".into(),
            urls: vec![Url::Full(format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-macos-arm64.tar.gz"
            ))],
            dest: llama_dir.to_path_buf(),
            size: LLAMA_SIZE,
            unpack: true,
        });
    }
    (out, unknown)
}

pub fn plan(cfg: &AppConfig, llama_dir: &Path) -> Plan {
    let (jobs, unknown) = jobs(cfg, llama_dir);
    // one row per model, not per file
    let mut items: Vec<PlanItem> = Vec::new();
    for j in &jobs {
        match items.iter_mut().find(|i| i.label == j.label) {
            Some(i) => i.size += j.size,
            None => items.push(PlanItem { label: j.label.clone(), size: j.size }),
        }
    }
    Plan { total: items.iter().map(|i| i.size).sum(), items, unknown }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Progress {
    /// "running" | "done" | "error" | "cancelled"
    pub state: String,
    pub label: String,
    pub done: u64,
    pub total: u64,
    pub bytes_per_s: f64,
    pub message: String,
}

#[derive(Default)]
pub struct Downloader {
    cancel: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
}

impl Downloader {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        if let Some(c) = self.child.lock().unwrap().as_mut() {
            let _ = c.kill();
        }
    }

    /// Download everything missing on a background thread. `on_llama` receives the
    /// llama-server path once it has been unpacked.
    pub fn start(&self, app: AppHandle, cfg: AppConfig, llama_dir: PathBuf, on_llama: impl FnOnce(PathBuf) + Send + 'static) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        self.cancel.store(false, Ordering::SeqCst);
        let (cancel, running, child) = (self.cancel.clone(), self.running.clone(), self.child.clone());
        std::thread::spawn(move || {
            let (jobs, _) = jobs(&cfg, &llama_dir);
            let total: u64 = jobs.iter().map(|j| j.size).sum();
            let mut p = Progress { state: "running".into(), total, label: "Connecting…".into(), ..Default::default() };
            let _ = app.emit_to("settings", "download", p.clone());
            let mut servers = if jobs.iter().any(|j| j.urls.iter().any(|u| matches!(u, Url::Hf(_)))) {
                fastest_hf()
            } else {
                [HF, HF_MIRROR]
            };
            let result = run(&app, &jobs, &mut p, &mut servers, &cancel, &child);
            p.bytes_per_s = 0.0;
            match result {
                Ok(()) => {
                    if let Some(j) = jobs.iter().find(|j| j.unpack) {
                        on_llama(j.dest.join("llama-server"));
                    }
                    p.state = "done".into();
                    p.done = total;
                }
                Err(_) if cancel.load(Ordering::SeqCst) => p.state = "cancelled".into(),
                Err(e) => {
                    p.state = "error".into();
                    p.message = e;
                }
            }
            let _ = app.emit_to("settings", "download", p);
            running.store(false, Ordering::SeqCst);
        });
    }
}

/// Both Hugging Face servers, faster first: each gets 15 s to send the first MB of a model.
/// Unreachable counts as slowest; with no answer at all the main server goes first.
fn fastest_hf() -> [&'static str; 2] {
    let speed = |server: &'static str| {
        std::thread::spawn(move || {
            let out = Command::new("curl")
                .args(["-L", "--fail", "--silent", "--max-time", "15", "--range", "0-1048575", "-o", "/dev/null", "-w", "%{speed_download}"])
                .arg(format!("{server}/{PROBE}"))
                .output();
            match out {
                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().unwrap_or(0.0),
                _ => 0.0,
            }
        })
    };
    let (a, b) = (speed(HF), speed(HF_MIRROR));
    let (a, b) = (a.join().unwrap_or(0.0), b.join().unwrap_or(0.0));
    log::info!("download speed check: huggingface.co {:.0} KB/s, hf-mirror.com {:.0} KB/s", a / 1e3, b / 1e3);
    if b > a {
        [HF_MIRROR, HF]
    } else {
        [HF, HF_MIRROR]
    }
}

fn run(
    app: &AppHandle,
    jobs: &[Job],
    p: &mut Progress,
    servers: &mut [&'static str; 2],
    cancel: &AtomicBool,
    child: &Mutex<Option<Child>>,
) -> Result<(), String> {
    // what is still needed, minus what earlier attempts left in .part files
    let need: u64 = jobs
        .iter()
        .map(|j| {
            let part = if j.unpack { j.dest.with_extension("tar.gz.part") } else { append(&j.dest, ".part") };
            j.size.saturating_sub(file_len(&part)) + if j.unpack { j.size * 3 } else { 0 }
        })
        .sum();
    if let Some(dest) = jobs.first().map(|j| j.dest.as_path()) {
        if let Some(free) = free_space(dest) {
            // keep some room for the system
            if free < need + 500_000_000 {
                return Err(format!(
                    "Not enough disk space.\nThe download needs {} and only {} is free on this disk. Free up some space and click Try Again.",
                    gb(need + 500_000_000),
                    gb(free)
                ));
            }
        }
    }
    let mut finished = 0u64;
    let started = Instant::now();
    let mut fetched = 0u64; // bytes fetched in this session, for the speed
    for job in jobs {
        p.label = job.label.clone();
        let part = if job.unpack { job.dest.with_extension("tar.gz.part") } else { append(&job.dest, ".part") };
        if let Some(dir) = part.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("Could not create the folder {}: {e}.", dir.display()))?;
        }
        let mut reasons: Vec<String> = Vec::new();
        let mut codes: Vec<(Option<i32>, String)> = Vec::new();
        let mut ok = false;
        // (url, Hugging Face server it is on)
        let urls: Vec<(String, Option<&'static str>)> = job
            .urls
            .iter()
            .flat_map(|u| match u {
                Url::Hf(path) => servers.iter().map(|s| (format!("{s}/{path}"), Some(*s))).collect::<Vec<_>>(),
                Url::Full(u) => vec![(u.clone(), None)],
            })
            .collect();
        for (url, server) in &urls {
            let before = file_len(&part);
            let mut cmd = Command::new("curl");
            cmd.args(["-L", "--fail", "--silent", "--show-error", "--retry", "3", "--connect-timeout", "20"])
                .args(["--speed-limit", &STALL_BYTES_PER_S.to_string(), "--speed-time", &STALL_S.to_string()])
                .args(["-C", "-", "-o"])
                .arg(&part)
                .arg(url)
                .args(["-w", "%{http_code}"])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let c = cmd.spawn().map_err(|e| format!("Could not start the download: curl is missing or cannot run ({e})."))?;
            *child.lock().unwrap() = Some(c);
            // report progress while curl runs
            let status = loop {
                std::thread::sleep(Duration::from_millis(250));
                let now = file_len(&part);
                p.done = finished + now.min(job.size);
                let session = fetched + now.saturating_sub(before);
                p.bytes_per_s = session as f64 / started.elapsed().as_secs_f64().max(0.5);
                let _ = app.emit_to("settings", "download", p.clone());
                let mut guard = child.lock().unwrap();
                match guard.as_mut().map(|c| c.try_wait()) {
                    Some(Ok(Some(s))) => {
                        use std::io::Read;
                        let (mut err, mut http) = (String::new(), String::new());
                        if let Some(c) = guard.as_mut() {
                            if let Some(mut e) = c.stderr.take() {
                                let _ = e.read_to_string(&mut err);
                            }
                            if let Some(mut o) = c.stdout.take() {
                                let _ = o.read_to_string(&mut http);
                            }
                        }
                        *guard = None;
                        break (s, err, http);
                    }
                    Some(Ok(None)) => {}
                    _ => break (std::process::ExitStatus::default(), "curl stopped".into(), String::new()),
                }
            };
            fetched += file_len(&part).saturating_sub(before);
            if cancel.load(Ordering::SeqCst) {
                return Err("cancelled".into());
            }
            // a finished .part makes curl's resume fail with "range not satisfiable"
            if status.0.success() || (job.size > 0 && file_len(&part) >= job.size) {
                // keep using the server that worked
                if let Some(s) = server {
                    if servers[0] != *s {
                        servers.swap(0, 1);
                    }
                }
                ok = true;
                break;
            }
            log::warn!("download from {url} failed ({:?}, HTTP {}): {}", status.0.code(), status.2.trim(), status.1.trim());
            codes.push((status.0.code(), status.2.trim().to_string()));
            reasons.push(format!("{}: {}.", host(url), explain(status.0.code(), status.2.trim(), status.1.trim())));
        }
        if !ok {
            let hint = if codes.iter().any(|c| c.0 == Some(23)) {
                "Free up some disk space and click Try Again."
            } else if codes.iter().all(|c| c.0 == Some(22) && c.1 == "404") {
                "The file may have moved. Please report this on the project's GitHub page."
            } else if codes.iter().all(|c| c.0 == Some(22)) {
                "The servers may be busy. Wait a few minutes and click Try Again."
            } else {
                "Check your internet connection (and proxy or VPN, if you use one) and click Try Again."
            };
            return Err(format!(
                "Could not download {}.\n{}\n{hint} What was already downloaded is kept.",
                job.label,
                reasons.join("\n")
            ));
        }
        if job.unpack {
            if let Err(e) = unpack(&part, &job.dest) {
                let _ = std::fs::remove_file(&part); // a broken archive: fetch it again next time
                return Err(e);
            }
            let _ = std::fs::remove_file(&part);
        } else {
            std::fs::rename(&part, &job.dest).map_err(|e| format!("Could not save {}: {e}.", job.dest.display()))?;
        }
        finished += job.size;
    }
    Ok(())
}

/// Unpack a llama.cpp release (one top-level folder) into `dir`.
fn unpack(archive: &Path, dir: &Path) -> Result<(), String> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let out = Command::new("tar")
        .args(["-xzf"])
        .arg(archive)
        .args(["--strip-components", "1", "-C"])
        .arg(dir)
        .output()
        .map_err(|e| format!("Could not unpack llama-server: tar cannot run ({e})."))?;
    if !out.status.success() {
        return Err(format!("Could not unpack llama-server: {}. Click Try Again to download it again.", String::from_utf8_lossy(&out.stderr).trim()));
    }
    if !dir.join("llama-server").exists() {
        return Err("The llama.cpp download does not contain llama-server. Install it with `brew install llama.cpp` instead, then reopen the settings.".into());
    }
    Ok(())
}

/// Server name of a URL, for messages.
fn host(url: &str) -> &str {
    url.split("://").nth(1).and_then(|r| r.split('/').next()).unwrap_or(url)
}

/// Why curl failed, in words: from its exit code and the HTTP status.
fn explain(code: Option<i32>, http: &str, stderr: &str) -> String {
    // through a proxy curl may print 000 and give the status only in its message
    let http_n: u16 = match http.parse().unwrap_or(0) {
        0 => stderr.rsplit("error: ").next().and_then(|t| t.trim().get(..3)).and_then(|t| t.parse().ok()).unwrap_or(0),
        n => n,
    };
    match code {
        Some(6) => "the server name could not be found (no internet connection, or DNS is blocked)".into(),
        Some(7) => "could not connect to the server (it may be blocked on this network)".into(),
        Some(28) => format!("the connection timed out or was too slow (under {} KB/s for {STALL_S} s)", STALL_BYTES_PER_S / 1000),
        Some(35) | Some(60) | Some(51) | Some(58) | Some(77) | Some(83) | Some(90) | Some(91) => {
            "the secure connection failed (a proxy or firewall may be interfering)".into()
        }
        Some(5) => "the proxy server could not be found".into(),
        Some(18) | Some(52) | Some(55) | Some(56) | Some(92) => "the connection was dropped".into(),
        Some(23) => "the file could not be saved (is the disk full?)".into(),
        Some(22) => match http_n {
            404 => "the file is not on the server (HTTP 404)".into(),
            401 | 403 => format!("the server refused access (HTTP {http_n})"),
            429 => "too many requests; wait a few minutes (HTTP 429)".into(),
            500..=599 => format!("the server had an error (HTTP {http_n})"),
            0 => "the server returned an error".into(),
            n => format!("the server returned HTTP {n}"),
        },
        None => "the download was stopped".into(),
        Some(n) if !stderr.is_empty() => {
            let msg = stderr.lines().last().unwrap_or(stderr).trim_start_matches("curl: ").trim_start_matches(&format!("({n}) "));
            format!("{} (curl error {n})", msg.trim_end_matches('.'))
        }
        Some(n) => format!("curl error {n}"),
    }
}

/// Free bytes on the disk holding `dir` (or its nearest existing parent).
fn free_space(dir: &Path) -> Option<u64> {
    let mut d = dir;
    while !d.exists() {
        d = d.parent()?;
    }
    let out = Command::new("df").arg("-Pk").arg(d).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
    Some(kb * 1024)
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

fn append(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curl_errors_in_words() {
        assert!(explain(Some(6), "000", "").contains("could not be found"));
        assert!(explain(Some(22), "404", "").contains("HTTP 404"));
        assert!(explain(Some(22), "000", "curl: (22) The requested URL returned error: 403").contains("refused access (HTTP 403)"));
        assert!(explain(Some(22), "503", "").contains("server had an error"));
        assert!(explain(Some(28), "000", "").contains("too slow"));
        assert_eq!(explain(Some(99), "000", "curl: (99) odd thing."), "odd thing (curl error 99)");
        assert_eq!(host("https://hf-mirror.com/a/b"), "hf-mirror.com");
    }

    #[test]
    fn plan_covers_every_default_model() {
        let dir = std::env::temp_dir().join("undertext-plan-test");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = AppConfig { models_dir: dir.join("models").display().to_string(), ..Default::default() };
        let p = plan(&cfg, &dir.join("llama.cpp"));
        assert!(p.unknown.is_empty(), "{:?}", p.unknown);
        let labels: Vec<_> = p.items.iter().map(|i| i.label.as_str()).collect();
        for l in ["Speech recognition (Parakeet-TDT)", "Voice activity (Silero VAD)", "Translation (Qwen3.5-2B)", "Quick drafts (Opus-MT)", "Speaker labels (TitaNet)"] {
            assert!(labels.contains(&l), "{labels:?}");
        }
        // one file already there: no longer planned
        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::write(dir.join("models/silero_vad.onnx"), b"x").unwrap();
        assert!(!plan(&cfg, &dir.join("llama.cpp")).items.iter().any(|i| i.label.contains("Silero")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
