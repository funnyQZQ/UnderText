//! First-run download of the models.
//!
//! Every model is in the project's own repositories, laid out exactly like the models
//! folder: Hugging Face (funnyQZQ/undertext-models) and ModelScope (funnyqzq/undertext-models,
//! fast in mainland China). Before starting, both are tried with a small request and the
//! faster one is used; if a file fails or stalls there, it is fetched from the other one.
//! Each file is fetched with the system `curl` into `<name>.part`, which is resumed on the
//! next attempt (also from the other server) and renamed when complete. Progress goes to
//! the settings window as `download` events. When neither server works, the error asks the
//! settings window to explain how to download the files by hand (`manual`).

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use undertext_core::config::{AppConfig, OPUS_MT_FILES};

pub struct Server {
    pub name: &'static str,
    /// Files are at `{base}/{path in the models folder}`.
    base: &'static str,
    /// Where people can download the files by hand.
    pub page: &'static str,
}

pub static HUGGING_FACE: Server = Server {
    name: "Hugging Face",
    base: "https://huggingface.co/funnyQZQ/undertext-models/resolve/main",
    page: "https://huggingface.co/funnyQZQ/undertext-models/tree/main",
};
pub static MODELSCOPE: Server = Server {
    name: "ModelScope",
    base: "https://modelscope.cn/models/funnyqzq/undertext-models/resolve/master",
    page: "https://modelscope.cn/models/funnyqzq/undertext-models/files",
};
static SERVERS: [&Server; 2] = [&HUGGING_FACE, &MODELSCOPE];

/// Fetched (first MB only) from each server to see which one is faster.
const PROBE: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx";
/// A transfer slower than this for `STALL_S` seconds is dropped and the other server tried.
const STALL_BYTES_PER_S: u32 = 20_000;
const STALL_S: u32 = 30;

/// Size in bytes of each known file, by path in the models folder (= path in the repositories).
fn size_of(rel: &str) -> Option<u64> {
    Some(match rel {
        "Qwen3.5-2B-Q4_K_M.gguf" => 1_280_835_840,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx" => 652_184_296,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx" => 7_257_753,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx" => 1_739_080,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt" => 9_384,
        "silero_vad.onnx" => 643_854,
        "nemo_en_titanet_small.onnx" => 40_257_283,
        "opus-mt-en-zh-ct2/config.json" => 215,
        "opus-mt-en-zh-ct2/model.bin" => 155_502_615,
        "opus-mt-en-zh-ct2/shared_vocabulary.json" => 1_303_887,
        "opus-mt-en-zh-ct2/source.spm" => 806_435,
        "opus-mt-en-zh-ct2/target.spm" => 804_600,
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
    /// The missing files, by path in the models folder (for downloading by hand).
    pub files: Vec<String>,
    /// Missing files the app doesn't know how to download (custom model names).
    pub unknown: Vec<String>,
}

struct Job {
    label: String,
    /// Path in the models folder and in the repositories.
    rel: String,
    dest: PathBuf,
    size: u64,
}

fn group_label(rel: &str) -> &'static str {
    if rel.starts_with("sherpa-onnx-nemo-parakeet") {
        "语音识别（Parakeet-TDT）"
    } else if rel == "silero_vad.onnx" {
        "人声检测（Silero VAD）"
    } else if rel.contains("titanet") {
        "说话人识别（TitaNet）"
    } else if rel.ends_with(".gguf") {
        "翻译（Qwen3.5-2B）"
    } else if rel.starts_with("opus-mt") {
        "即时草稿（Opus-MT）"
    } else {
        "模型文件"
    }
}

/// Model files still missing for `cfg`.
fn jobs(cfg: &AppConfig) -> (Vec<Job>, Vec<String>) {
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
        match rel.ok().and_then(|r| size_of(&r).map(|size| (r, size))) {
            Some((rel, size)) => out.push(Job { label: group_label(&rel).into(), rel, dest: path, size }),
            None => unknown.push(path.display().to_string()),
        }
    }
    (out, unknown)
}

pub fn plan(cfg: &AppConfig) -> Plan {
    let (jobs, unknown) = jobs(cfg);
    // one row per model, not per file
    let mut items: Vec<PlanItem> = Vec::new();
    for j in &jobs {
        match items.iter_mut().find(|i| i.label == j.label) {
            Some(i) => i.size += j.size,
            None => items.push(PlanItem { label: j.label.clone(), size: j.size }),
        }
    }
    let files = jobs.iter().map(|j| j.rel.clone()).collect();
    Plan { total: items.iter().map(|i| i.size).sum(), items, files, unknown }
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
    /// Neither server worked: show how to download the files by hand.
    pub manual: bool,
}

#[derive(Default)]
pub struct Downloader {
    cancel: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
}

/// Why a download stopped; `manual` when downloading by hand is the way forward.
struct Failure {
    message: String,
    manual: bool,
}

impl Downloader {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        if let Some(c) = self.child.lock().unwrap().as_mut() {
            let _ = c.kill();
        }
    }

    /// Download everything missing on a background thread.
    pub fn start(&self, app: AppHandle, cfg: AppConfig) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        self.cancel.store(false, Ordering::SeqCst);
        let (cancel, running, child) = (self.cancel.clone(), self.running.clone(), self.child.clone());
        std::thread::spawn(move || {
            let (jobs, _) = jobs(&cfg);
            let total: u64 = jobs.iter().map(|j| j.size).sum();
            let mut p = Progress { state: "running".into(), total, label: "正在测试下载速度…".into(), ..Default::default() };
            let _ = app.emit_to("settings", "download", p.clone());
            let mut servers = if jobs.is_empty() { SERVERS } else { fastest() };
            let result = run(&app, &jobs, &mut p, &mut servers, &cancel, &child);
            p.bytes_per_s = 0.0;
            match result {
                Ok(()) => {
                    p.state = "done".into();
                    p.done = total;
                }
                Err(_) if cancel.load(Ordering::SeqCst) => p.state = "cancelled".into(),
                Err(f) => {
                    p.state = "error".into();
                    p.message = f.message;
                    p.manual = f.manual;
                }
            }
            let _ = app.emit_to("settings", "download", p);
            running.store(false, Ordering::SeqCst);
        });
    }
}

/// Both servers, faster first: each gets 10 s to send the first MB of a model. A server
/// that sends next to nothing counts as slowest; when neither answers, Hugging Face goes first.
fn fastest() -> [&'static Server; 2] {
    let speed = |server: &'static Server| {
        std::thread::spawn(move || {
            // no --fail: the speed is read even when the request ends early
            let out = Command::new("curl")
                .args(["-L", "--silent", "--max-time", "10", "--range", "0-1048575", "-o", "/dev/null"])
                .args(["-w", "%{http_code} %{speed_download} %{size_download}"])
                .arg(format!("{}/{PROBE}", server.base))
                .output();
            let text = out.map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
            let f: Vec<&str> = text.split_whitespace().collect();
            let ok = f.first().is_some_and(|c| c.starts_with('2'));
            let bytes: f64 = f.get(2).and_then(|v| v.parse().ok()).unwrap_or(0.0);
            if ok && bytes >= 64_000.0 { f.get(1).and_then(|v| v.parse().ok()).unwrap_or(0.0) } else { 0.0 }
        })
    };
    let (a, b) = (speed(&HUGGING_FACE), speed(&MODELSCOPE));
    let (a, b): (f64, f64) = (a.join().unwrap_or(0.0), b.join().unwrap_or(0.0));
    log::info!("download speed check: Hugging Face {:.0} KB/s, ModelScope {:.0} KB/s", a / 1e3, b / 1e3);
    if b > a {
        [&MODELSCOPE, &HUGGING_FACE]
    } else {
        [&HUGGING_FACE, &MODELSCOPE]
    }
}

fn run(
    app: &AppHandle,
    jobs: &[Job],
    p: &mut Progress,
    servers: &mut [&'static Server; 2],
    cancel: &AtomicBool,
    child: &Mutex<Option<Child>>,
) -> Result<(), Failure> {
    // what is still needed, minus what earlier attempts left in .part files
    let need: u64 = jobs.iter().map(|j| j.size.saturating_sub(file_len(&append(&j.dest, ".part")))).sum();
    if let Some(dest) = jobs.first().map(|j| j.dest.as_path()) {
        if let Some(free) = free_space(dest) {
            // keep some room for the system
            if free < need + 500_000_000 {
                return Err(Failure {
                    message: format!(
                        "磁盘空间不足。\n下载需要 {}，这块磁盘只剩 {}。请清理一些空间后点“重试”。",
                        gb(need + 500_000_000),
                        gb(free)
                    ),
                    manual: false,
                });
            }
        }
    }
    let local = |message: String| Failure { message, manual: false };
    let mut finished = 0u64;
    let started = Instant::now();
    let mut fetched = 0u64; // bytes fetched in this session, for the speed
    for job in jobs {
        p.label = job.label.clone();
        let part = append(&job.dest, ".part");
        if let Some(dir) = part.parent() {
            std::fs::create_dir_all(dir).map_err(|e| local(format!("无法创建文件夹 {}：{e}。", dir.display())))?;
        }
        let mut reasons: Vec<String> = Vec::new();
        let mut codes: Vec<(Option<i32>, String)> = Vec::new();
        let mut ok = false;
        for server in servers.clone() {
            let url = format!("{}/{}", server.base, job.rel);
            let before = file_len(&part);
            let mut cmd = Command::new("curl");
            cmd.args(["-L", "--fail", "--silent", "--show-error", "--retry", "3", "--connect-timeout", "20"])
                .args(["--speed-limit", &STALL_BYTES_PER_S.to_string(), "--speed-time", &STALL_S.to_string()])
                .args(["-C", "-", "-o"])
                .arg(&part)
                .arg(&url)
                .args(["-w", "%{http_code}"])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let c = cmd.spawn().map_err(|e| local(format!("无法开始下载：找不到 curl 或无法运行（{e}）。")))?;
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
                return Err(local("cancelled".into()));
            }
            // a finished .part makes curl's resume fail with "range not satisfiable"
            if status.0.success() || (job.size > 0 && file_len(&part) >= job.size) {
                // keep using the server that worked
                if !std::ptr::eq(servers[0], server) {
                    servers.swap(0, 1);
                }
                ok = true;
                break;
            }
            log::warn!("download from {url} failed ({:?}, HTTP {}): {}", status.0.code(), status.2.trim(), status.1.trim());
            codes.push((status.0.code(), status.2.trim().to_string()));
            reasons.push(format!("{}: {}。", server.name, explain(status.0.code(), status.2.trim(), status.1.trim())));
        }
        if !ok {
            if codes.iter().any(|c| c.0 == Some(23)) {
                return Err(local(format!(
                    "无法下载{}。\n{}\n请清理一些磁盘空间后点“重试”。已经下载的部分会保留。",
                    job.label,
                    reasons.join("\n")
                )));
            }
            let hint = if codes.iter().all(|c| c.0 == Some(22)) {
                "两个下载站都暂时用不了。可以等几分钟再点“重试”，或者按下面的步骤手动下载。"
            } else {
                "两个下载站都连不上。请检查网络（用了代理或 VPN 的话也检查一下）后点“重试”，或者按下面的步骤手动下载。"
            };
            return Err(Failure {
                message: format!("无法下载{}。\n{}\n{hint}", job.label, reasons.join("\n")),
                manual: true,
            });
        }
        std::fs::rename(&part, &job.dest).map_err(|e| local(format!("无法保存 {}：{e}。", job.dest.display())))?;
        finished += job.size;
    }
    Ok(())
}

/// Why curl failed, in words: from its exit code and the HTTP status.
fn explain(code: Option<i32>, http: &str, stderr: &str) -> String {
    // through a proxy curl may print 000 and give the status only in its message
    let http_n: u16 = match http.parse().unwrap_or(0) {
        0 => stderr.rsplit("error: ").next().and_then(|t| t.trim().get(..3)).and_then(|t| t.parse().ok()).unwrap_or(0),
        n => n,
    };
    match code {
        Some(6) => "找不到服务器地址（没有联网，或 DNS 被拦截）".into(),
        Some(7) => "连不上服务器（当前网络可能屏蔽了它）".into(),
        Some(28) => format!("连接超时或速度太慢（连续 {STALL_S} 秒低于 {} KB/s）", STALL_BYTES_PER_S / 1000),
        Some(35) | Some(60) | Some(51) | Some(58) | Some(77) | Some(83) | Some(90) | Some(91) => {
            "安全连接失败（可能是代理或防火墙在干扰）".into()
        }
        Some(5) => "找不到代理服务器".into(),
        Some(18) | Some(52) | Some(55) | Some(56) | Some(92) => "连接中断了".into(),
        Some(23) => "文件无法保存（磁盘满了吗？）".into(),
        Some(22) => match http_n {
            404 => "服务器上没有这个文件（HTTP 404）".into(),
            401 | 403 => format!("服务器拒绝访问（HTTP {http_n}）"),
            429 => "请求太多，请等几分钟（HTTP 429）".into(),
            500..=599 => format!("服务器出错（HTTP {http_n}）"),
            0 => "服务器返回了错误".into(),
            n => format!("服务器返回 HTTP {n}"),
        },
        None => "下载被中止了".into(),
        Some(n) if !stderr.is_empty() => {
            let msg = stderr.lines().last().unwrap_or(stderr).trim_start_matches("curl: ").trim_start_matches(&format!("({n}) "));
            format!("{}（curl 错误 {n}）", msg.trim_end_matches('.'))
        }
        Some(n) => format!("curl 错误 {n}"),
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
        assert!(explain(Some(6), "000", "").contains("找不到服务器"));
        assert!(explain(Some(22), "404", "").contains("HTTP 404"));
        assert!(explain(Some(22), "000", "curl: (22) The requested URL returned error: 403").contains("拒绝访问（HTTP 403）"));
        assert!(explain(Some(22), "503", "").contains("服务器出错"));
        assert!(explain(Some(28), "000", "").contains("太慢"));
        assert_eq!(explain(Some(99), "000", "curl: (99) odd thing."), "odd thing（curl 错误 99）");
    }

    #[test]
    fn plan_covers_every_default_model() {
        let dir = std::env::temp_dir().join("undertext-plan-test");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = AppConfig { models_dir: dir.join("models").display().to_string(), ..Default::default() };
        let p = plan(&cfg);
        assert_eq!(p.files.len(), 12, "{:?}", p.files);
        assert!(p.unknown.is_empty(), "{:?}", p.unknown);
        let labels: Vec<_> = p.items.iter().map(|i| i.label.as_str()).collect();
        for l in ["语音识别（Parakeet-TDT）", "人声检测（Silero VAD）", "翻译（Qwen3.5-2B）", "即时草稿（Opus-MT）", "说话人识别（TitaNet）"] {
            assert!(labels.contains(&l), "{labels:?}");
        }
        // one file already there: no longer planned
        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::write(dir.join("models/silero_vad.onnx"), b"x").unwrap();
        assert!(!plan(&cfg).items.iter().any(|i| i.label.contains("Silero")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn urls_follow_the_models_folder() {
        assert_eq!(
            format!("{}/{}", MODELSCOPE.base, "opus-mt-en-zh-ct2/model.bin"),
            "https://modelscope.cn/models/funnyqzq/undertext-models/resolve/master/opus-mt-en-zh-ct2/model.bin"
        );
        assert!(size_of(PROBE).unwrap() > 1_048_576);
    }
}
