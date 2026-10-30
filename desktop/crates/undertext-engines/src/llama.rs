//! llama.cpp integration: `llama-server` processes managed by the app + an HTTP client.
//!
//! Running the official llama-server (Homebrew on macOS, release zip on Windows) instead of
//! linking llama.cpp keeps new model architectures (Qwen3.5) available as soon as
//! llama.cpp supports them, and isolates model crashes from the app.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use undertext_core::translate::CompletionBackend;
use serde_json::{json, Value};

/// Locations searched when no binary is configured. GUI apps on macOS get a minimal
/// PATH, so the Homebrew prefixes are listed explicitly.
pub fn find_llama_server(configured: &str) -> Result<PathBuf> {
    if !configured.is_empty() {
        let p = PathBuf::from(configured);
        return if p.exists() { Ok(p) } else { Err(anyhow!("llama-server not found at {configured}")) };
    }
    let exe = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
    dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].iter().map(PathBuf::from));
    if let Ok(cur) = std::env::current_exe() {
        if let Some(d) = cur.parent() {
            dirs.push(d.to_path_buf()); // bundled next to the app binary
        }
    }
    dirs.into_iter().map(|d| d.join(exe)).find(|p| p.is_file()).ok_or_else(|| {
        anyhow!("llama-server not found. Install it (macOS: `brew install llama.cpp`) or set llama_server.binary")
    })
}

/// Blocking client for llama-server's `/completion` endpoint.
pub struct LlamaServerClient {
    base: String,
    agent: ureq::Agent,
    /// Ask the server to reuse the cached prompt prefix (`cache_prompt`).
    pub cache_prompt: bool,
}

impl LlamaServerClient {
    pub fn new(port: u16) -> Self {
        Self::with_url(&format!("http://127.0.0.1:{port}"))
    }

    pub fn with_url(url: &str) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).build();
        Self { base: url.trim_end_matches('/').to_string(), agent, cache_prompt: false }
    }

    pub fn healthy(&self) -> bool {
        matches!(self.agent.get(&format!("{}/health", self.base)).timeout(Duration::from_secs(2)).call(), Ok(r) if r.status() == 200)
    }
}

impl CompletionBackend for LlamaServerClient {
    fn complete(&self, prompt: &str, max_tokens: usize, stop: &[String]) -> Result<String> {
        let body = json!({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "stop": stop,
            "cache_prompt": self.cache_prompt,
        });
        let resp: Value = undertext_core::metrics::LLM.time(|| -> Result<Value> {
            self.agent
                .post(&format!("{}/completion", self.base))
                .send_json(body)
                .map_err(|e| anyhow!("llama-server request failed: {e}"))?
                .into_json()
                .context("llama-server returned invalid JSON")
        })?;
        let n = |v: &Value| v.as_u64().unwrap_or(0);
        let t = &resp["timings"];
        // tokens_cached = tokens in the slot afterwards = reused + evaluated + generated
        let (evaluated, generated) = (n(&t["prompt_n"]), n(&t["predicted_n"]));
        let reused = n(&resp["tokens_cached"]).saturating_sub(evaluated + generated);
        undertext_core::metrics::LLM.add_tokens(evaluated, reused, generated);
        Ok(resp.get("content").and_then(Value::as_str).unwrap_or("").to_string())
    }
}

/// A llama-server child process; killed when dropped.
pub struct LlamaServerProcess {
    child: Option<Child>,
    pub port: u16,
    log_path: PathBuf,
}

pub struct ServerSpec<'a> {
    pub binary: &'a Path,
    pub model: &'a Path,
    pub port: u16,
    pub n_gpu_layers: i32,
    pub ctx_size: u32,
    /// CPU threads; 0 = llama.cpp default.
    pub threads: i32,
    /// Keep llama.cpp's prompt caches (host-RAM cache, hybrid-model checkpoints).
    pub prompt_cache: bool,
    pub extra_args: &'a [String],
}

/// `llama-server --help` output, to pass optional flags only to builds that know them.
fn help_text(binary: &Path) -> String {
    Command::new(binary)
        .arg("--help")
        .output()
        .map(|o| format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
        .unwrap_or_default()
}

/// Flags that keep a fully GPU-offloaded server cool and small. Optional flags are only
/// passed when this llama-server build lists them in `--help`.
/// * `-t/-tb`: few CPU threads (the CPU only orchestrates);
/// * `--poll 0`: threads sleep instead of busy-polling while the GPU works;
/// * `--cache-ram 0`: no host-RAM prompt cache. Newer builds keep a copy of the model
///   state for past prompts there (up to 8 GiB by default), which grows over a long run
///   and is never hit here: every prompt is new;
/// * `--ctx-checkpoints 0`: no state snapshots for hybrid models like Qwen3.5 (each one
///   tens of MiB, and our prompts can't reuse them either).
fn low_power_args(spec: &ServerSpec, help: &str) -> Vec<String> {
    let mut a = Vec::new();
    if spec.threads > 0 {
        a.extend(["-t".to_string(), spec.threads.to_string(), "-tb".to_string(), spec.threads.to_string()]);
    }
    for (flag, value) in [("--poll", "0"), ("--cache-ram", "0"), ("--ctx-checkpoints", "0")] {
        if spec.prompt_cache && flag != "--poll" {
            continue;
        }
        if help.contains(flag) {
            a.extend([flag.to_string(), value.to_string()]);
        }
    }
    a
}

impl LlamaServerProcess {
    /// Start a server, or adopt one already answering on the port (e.g. started by hand).
    pub fn start(spec: &ServerSpec) -> Result<Self> {
        let log_path = std::env::temp_dir().join(format!("undertext-llama-server-{}.log", spec.port));
        if LlamaServerClient::new(spec.port).healthy() {
            log::warn!("reusing the llama-server already running on port {}", spec.port);
            return Ok(Self { child: None, port: spec.port, log_path });
        }
        if !spec.model.exists() {
            bail!("model not found: {}", spec.model.display());
        }
        let log = File::create(&log_path).with_context(|| format!("creating {}", log_path.display()))?;
        let child = Command::new(spec.binary)
            .arg("-m")
            .arg(spec.model)
            .args(["--host", "127.0.0.1", "--port", &spec.port.to_string()])
            .args(["-ngl", &spec.n_gpu_layers.to_string(), "-c", &spec.ctx_size.to_string(), "-np", "1"])
            .args(low_power_args(spec, &help_text(spec.binary)))
            .args(spec.extra_args)
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()
            .with_context(|| format!("starting {}", spec.binary.display()))?;
        Ok(Self { child: Some(child), port: spec.port, log_path })
    }

    /// Wait until the model is loaded (`/health` returns 200).
    pub fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let client = LlamaServerClient::new(self.port);
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if client.healthy() {
                return Ok(());
            }
            if let Some(child) = &mut self.child {
                if let Some(status) = child.try_wait()? {
                    bail!("llama-server exited ({status}):\n{}", self.log_tail(15));
                }
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        bail!("llama-server on port {} not ready after {:?}:\n{}", self.port, timeout, self.log_tail(15))
    }

    pub fn log_tail(&self, n: usize) -> String {
        let Ok(f) = File::open(&self.log_path) else { return String::new() };
        let lines: Vec<String> = BufReader::new(f).lines().map_while(Result::ok).collect();
        lines[lines.len().saturating_sub(n)..].join("\n")
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for LlamaServerProcess {
    fn drop(&mut self) {
        self.stop();
    }
}
