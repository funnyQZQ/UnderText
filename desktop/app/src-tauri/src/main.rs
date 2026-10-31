//! UnderText desktop app (Tauri 2).
//!
//! * overlay window : transparent, always-on-top subtitles (English + Chinese)
//! * settings window: models folder, audio source, LLMs, diarization, appearance
//! * tray menu      : start / stop, click-through, show / hide, settings, quit
//!
//! The interface is in Chinese; engine messages are translated in `zh.rs`.
//!
//! The pipeline runs on its own thread (built by `undertext_engines::pipeline`); a second
//! thread pushes subtitle snapshots to the overlay at up to 10 Hz.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod download;
mod zh;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use undertext_core::board::SubtitleBoard;
use undertext_core::config::AppConfig;
use undertext_core::eventlog::EventLog;
use undertext_core::Paragraph;
use undertext_engines::audio::{list_input_devices, list_output_devices, SourceIter};
use undertext_engines::llama::find_llama_server;
use undertext_engines::pipeline::{self, RunOptions};
use serde::Serialize;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, State, Wry};
use tauri_plugin_dialog::DialogExt;

struct Runner {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

struct TrayItems {
    start: MenuItem<Wry>,
    click_through: CheckMenuItem<Wry>,
}

struct AppState {
    config: Mutex<AppConfig>,
    config_path: PathBuf,
    runner: Mutex<Option<Runner>>,
    /// Runs that were asked to stop but may still be shutting down (their llama-servers
    /// must be gone before a new run starts them again). Only joined on background
    /// threads: the main thread must never block (tray updates need it).
    stopping: Mutex<Vec<JoinHandle<()>>>,
    status: Mutex<Status>,
    tray: Mutex<Option<TrayItems>>,
    /// Subtitles of the current (or last) run and when it started, for saving to a file.
    board: Mutex<Option<(Arc<SubtitleBoard>, chrono::DateTime<chrono::Local>)>>,
    downloader: download::Downloader,
}

#[derive(Debug, Clone, Serialize, Default)]
struct Status {
    running: bool,
    message: String,
    error: Option<String>,
    warnings: Vec<String>,
}

/// Only what changed since the last push: long histories would otherwise send (and
/// re-render) every paragraph ten times a second.
#[derive(Serialize, Clone)]
struct SubtitlesPayload {
    /// Start over (new run): drop what the overlay has.
    reset: bool,
    /// Paragraph ids (first sentence id) in display order.
    order: Vec<usize>,
    /// Paragraphs that are new or changed.
    changed: Vec<ParagraphUpdate>,
}

#[derive(Serialize, Clone)]
struct ParagraphUpdate {
    id: usize,
    #[serde(flatten)]
    paragraph: Paragraph,
}

#[derive(Serialize)]
struct ModelCheck {
    what: String,
    path: String,
    ok: bool,
}

#[derive(Serialize)]
struct SetupCheck {
    models: Vec<ModelCheck>,
    llama_server: Result<String, String>,
    config_path: String,
}

#[derive(Serialize)]
struct Devices {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

fn set_status(app: &AppHandle, f: impl FnOnce(&mut Status)) {
    let state = app.state::<AppState>();
    let status = {
        let mut s = state.status.lock().unwrap();
        f(&mut s);
        s.clone()
    };
    // clone the handle out of the lock: set_text hops to the main thread, which may be
    // waiting for this same lock in a tray handler
    let item = state.tray.lock().unwrap().as_ref().map(|t| t.start.clone());
    if let Some(item) = item {
        let _ = item.set_text(if status.running { "停止字幕" } else { "开始字幕" });
    }
    let _ = app.emit("status", status);
}

// --- commands ----------------------------------------------------------------------------

#[tauri::command]
fn get_config(state: State<AppState>) -> AppConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
fn save_config(app: AppHandle, state: State<AppState>, config: AppConfig) -> Result<(), String> {
    config.save(&state.config_path).map_err(|e| format!("{e:#}"))?;
    *state.config.lock().unwrap() = config.clone();
    let _ = app.emit("config", config);
    Ok(())
}

#[tauri::command]
fn get_status(state: State<AppState>) -> Status {
    state.status.lock().unwrap().clone()
}

#[tauri::command]
fn check_setup(state: State<AppState>, config: Option<AppConfig>) -> SetupCheck {
    let cfg = config.unwrap_or_else(|| state.config.lock().unwrap().clone());
    SetupCheck {
        models: cfg
            .required_models()
            .into_iter()
            .map(|(what, path, ok)| ModelCheck { what, path: path.display().to_string(), ok })
            .collect(),
        llama_server: find_llama_server(&cfg.llama_server.binary).map(|p| p.display().to_string()).map_err(|e| zh::error(&e.to_string())),
        config_path: state.config_path.display().to_string(),
    }
}

/// What the first-run download would fetch for the current settings.
#[tauri::command]
fn download_plan(state: State<AppState>, config: Option<AppConfig>) -> download::Plan {
    let cfg = config.unwrap_or_else(|| state.config.lock().unwrap().clone());
    download::plan(&cfg)
}

/// Download the missing models in the background; progress arrives as `download` events.
#[tauri::command]
fn download_start(app: AppHandle, state: State<AppState>, config: Option<AppConfig>) {
    let cfg = config.unwrap_or_else(|| state.config.lock().unwrap().clone());
    state.downloader.start(app, cfg);
}

#[tauri::command]
fn download_cancel(state: State<AppState>) {
    state.downloader.cancel();
}

/// Open one of the model download pages, or the models folder in Finder, for
/// downloading the models by hand. Only these three targets can be opened.
#[tauri::command]
fn open_for_manual_download(state: State<AppState>, which: String, config: Option<AppConfig>) -> Result<(), String> {
    let target = match which.as_str() {
        "huggingface" => download::HUGGING_FACE.page.to_string(),
        "modelscope" => download::MODELSCOPE.page.to_string(),
        "models" => {
            let cfg = config.unwrap_or_else(|| state.config.lock().unwrap().clone());
            let dir = undertext_core::config::expand_home(&cfg.models_dir);
            std::fs::create_dir_all(&dir).map_err(|e| format!("无法创建模型文件夹 {}：{e}", dir.display()))?;
            dir.display().to_string()
        }
        other => return Err(format!("unknown target {other:?}")),
    };
    std::process::Command::new("open").arg(&target).spawn().map(|_| ()).map_err(|e| format!("无法打开 {target}：{e}"))
}

#[tauri::command]
fn list_devices() -> Result<Devices, String> {
    let names = |r: anyhow::Result<Vec<undertext_engines::audio::DeviceInfo>>| -> Result<Vec<String>, String> {
        Ok(r.map_err(|e| format!("无法读取音频设备：{}", zh::error(&format!("{e:#}"))))?.into_iter().map(|d| d.name).collect())
    };
    Ok(Devices { inputs: names(list_input_devices())?, outputs: names(list_output_devices())? })
}

#[tauri::command]
async fn pick_wav(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .set_title("选择 WAV 文件")
        .add_filter("WAV 音频", &["wav"])
        .blocking_pick_file()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.display().to_string())
}

#[tauri::command]
async fn pick_folder(app: AppHandle) -> Option<String> {
    app.dialog().file().set_title("选择模型文件夹").blocking_pick_folder().and_then(|p| p.into_path().ok()).map(|p| p.display().to_string())
}

/// Start the subtitles. Returns false when models are missing: the settings window then
/// shows the download dialog instead.
#[tauri::command]
fn start(app: AppHandle) -> Result<bool, String> {
    start_pipeline(&app)
}

#[tauri::command]
fn stop(app: AppHandle) {
    stop_pipeline(&app);
}

#[tauri::command]
fn set_click_through(app: AppHandle, on: bool) -> Result<(), String> {
    apply_click_through(&app, on)
}

/// Ask where to save, then write what was said in the current (or last) run as text:
/// sentences after `after` (the last sentence cleared from the window), or all of them.
/// Returns the path, or None when the user cancelled.
#[tauri::command]
async fn save_transcript(app: AppHandle, after: Option<i64>) -> Result<Option<String>, String> {
    let state = app.state::<AppState>();
    let Some((board, started)) = state.board.lock().unwrap().clone() else {
        return Err("还没有可以保存的字幕".into());
    };
    let para_cfg = state.config.lock().unwrap().ui.paragraphs.clone();
    let after = after.unwrap_or(-1);
    let paragraphs: Vec<_> = board
        .transcript(&para_cfg)
        .into_iter()
        .filter_map(|(start, mut p)| {
            let cut = p.sentences.first().is_some_and(|s| s.sid as i64 <= after);
            p.sentences.retain(|s| s.sid as i64 > after);
            // a paragraph cut by the clear starts with its first remaining sentence
            let start = if cut { p.sentences.first().and_then(|s| board.sentence_start(s.sid)) } else { start };
            (!p.sentences.is_empty()).then_some((start, p))
        })
        .collect();
    if paragraphs.iter().all(|(_, p)| p.sentences.is_empty()) {
        return Err("还没有可以保存的字幕".into());
    }
    let Some(path) = app
        .dialog()
        .file()
        .set_title("保存字幕")
        .set_file_name(format!("undertext-{}.txt", chrono::Local::now().format("%Y-%m-%d-%H%M")))
        .add_filter("文本", &["txt"])
        .blocking_save_file()
        .and_then(|p| p.into_path().ok())
    else {
        return Ok(None);
    };
    let stamp = |t: f64| (started + chrono::Duration::milliseconds((t.max(0.0) * 1000.0) as i64)).format("%Y-%m-%d %H:%M:%S %:z").to_string();
    let first = paragraphs.iter().find_map(|(t, _)| *t);
    let mut out = format!(
        "UnderText 字幕{}\n",
        first.map(|t| format!("，开始于 {}", stamp(t))).unwrap_or_default()
    );
    for (start, p) in paragraphs {
        let join = |f: &dyn Fn(&undertext_core::board::SentenceDisplay) -> String| {
            p.sentences.iter().map(f).filter(|t| !t.trim().is_empty()).collect::<Vec<_>>()
        };
        let en = join(&|s| format!("{}{}", s.en_stable, if s.en_unstable.is_empty() { String::new() } else { format!(" {}", s.en_unstable) }));
        let zh = join(&|s| format!("{}{}", s.zh_stable, s.zh_unstable));
        if en.is_empty() && zh.is_empty() {
            continue;
        }
        let at = start.map(|t| format!("[{}]", stamp(t)));
        let head = [at, p.speaker.clone()].into_iter().flatten().collect::<Vec<_>>().join(" ");
        out.push('\n');
        if !head.is_empty() {
            out.push_str(&head);
            out.push('\n');
        }
        if !en.is_empty() {
            out.push_str(&en.join(" "));
            out.push('\n');
        }
        if !zh.is_empty() {
            out.push_str(&zh.join(""));
            out.push('\n');
        }
    }
    std::fs::write(&path, out).map_err(|e| format!("无法保存：{e}"))?;
    Ok(Some(path.display().to_string()))
}

#[tauri::command]
fn open_settings(app: AppHandle) {
    show_window(&app, "settings");
}

// --- pipeline lifecycle ------------------------------------------------------------------

/// Whether the current settings can run: every model file there and, when translating,
/// llama-server found (it comes with the app). The download dialog is only offered for
/// missing model files.
fn needs_download(cfg: &AppConfig) -> bool {
    let missing = cfg.required_models().iter().any(|m| !m.2)
        || (cfg.translate && cfg.commit.enabled() && find_llama_server(&cfg.llama_server.binary).is_err());
    // only offer the dialog when there is something it can fetch (custom model names
    // can't be downloaded; those fall through to the usual error)
    missing && !download::plan(cfg).items.is_empty()
}

/// Bring up the settings window with the model download dialog.
fn offer_download(app: &AppHandle) {
    show_window(app, "settings");
    let _ = app.emit_to("settings", "offer_download", ());
    set_status(app, |s| {
        s.running = false;
        s.error = None;
        s.message = "请先下载模型".into();
    });
}

/// Start the pipeline. Ok(false): models are missing and the download dialog was shown.
fn start_pipeline(app: &AppHandle) -> Result<bool, String> {
    let state = app.state::<AppState>();
    let cfg = state.config.lock().unwrap().clone();
    if needs_download(&cfg) {
        offer_download(app);
        return Ok(false);
    }
    let mut runner = state.runner.lock().unwrap();
    if let Some(r) = runner.as_ref() {
        if !r.thread.is_finished() {
            return Ok(true); // already running
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    let (app2, stop2) = (app.clone(), stop.clone());
    let thread = std::thread::Builder::new()
        .name("pipeline".into())
        .spawn(move || {
            let previous: Vec<_> = app2.state::<AppState>().stopping.lock().unwrap().drain(..).collect();
            if !previous.is_empty() {
                set_status(&app2, |s| s.message = "正在等待上一次运行结束…".into());
                for prev in previous {
                    let _ = prev.join();
                }
            }
            set_status(&app2, |s| {
                *s = Status { running: true, message: "正在加载模型…".into(), ..Default::default() };
            });
            if let Err(e) = run_pipeline(&app2, &cfg, stop2) {
                log::error!("{e:#}");
                set_status(&app2, |s| {
                    s.running = false;
                    s.message = "出错了".into();
                    s.error = Some(zh::error(&format!("{e:#}")));
                });
            } else {
                set_status(&app2, |s| {
                    s.running = false;
                    s.message = "已停止".into();
                });
            }
        })
        .map_err(|e| e.to_string())?;
    *runner = Some(Runner { stop, thread });
    Ok(true)
}

fn run_pipeline(app: &AppHandle, cfg: &AppConfig, stop: Arc<AtomicBool>) -> anyhow::Result<()> {
    let log = if cfg.ui.log_path.is_empty() {
        None
    } else {
        use anyhow::Context;
        let path = undertext_core::config::expand_home(&cfg.ui.log_path);
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(dir);
        }
        let file = std::fs::File::create(&path)
            .with_context(|| format!("无法创建延迟日志 {}（设置 → 高级）", path.display()))?;
        Some(Arc::new(EventLog::new(Box::new(file))))
    };
    let app_s = app.clone();
    let mut pipe = pipeline::build(cfg, &RunOptions::default(), None, log, &move |msg| {
        let msg = zh::status(msg);
        set_status(&app_s, |s| s.message = msg);
    })?;
    let warnings: Vec<String> = pipe.warnings.iter().map(|w| zh::warning(w)).collect();
    let source_name = match cfg.audio.source {
        undertext_core::config::SourceKind::Mic if cfg.audio.device.is_empty() => "默认输入设备".to_string(),
        undertext_core::config::SourceKind::Mic => cfg.audio.device.clone(),
        undertext_core::config::SourceKind::Wav => cfg.audio.wav_path.rsplit(['/', '\\']).next().unwrap_or("").to_string(),
    };
    set_status(app, |s| {
        s.message = format!("正在收听：{source_name}");
        s.warnings = warnings;
    });

    // push subtitle snapshots to the overlay (only when something changed)
    let board = pipe.board.clone();
    let (history, para_cfg) = (cfg.ui.history, cfg.ui.paragraphs.clone());
    let done = Arc::new(AtomicBool::new(false));
    let (app_p, done_p) = (app.clone(), done.clone());
    let pusher = std::thread::spawn(move || {
        let mut seen = u64::MAX;
        let mut sent: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
        let mut reset = true;
        loop {
            let finished = done_p.load(Ordering::SeqCst);
            let v = board.version();
            if v != seen {
                seen = v;
                // current paragraph + `history` earlier ones; the overlay scrolls
                let paragraphs = board.paragraphs(&para_cfg, history + 1);
                let mut order = Vec::new();
                let mut changed = Vec::new();
                let mut now = std::collections::HashMap::new();
                for p in paragraphs {
                    let Some(id) = p.sentences.first().map(|s| s.sid) else { continue };
                    let json = serde_json::to_string(&p).unwrap_or_default();
                    if sent.get(&id) != Some(&json) {
                        changed.push(ParagraphUpdate { id, paragraph: p });
                    }
                    order.push(id);
                    now.insert(id, json);
                }
                if reset || !changed.is_empty() || order.len() != sent.len() {
                    let _ = app_p.emit_to("overlay", "subtitles", SubtitlesPayload { reset, order, changed });
                    reset = false;
                }
                sent = now;
            }
            if finished {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    });

    let result = (|| -> anyhow::Result<()> {
        let mut source = pipeline::open_source(cfg, stop.clone())?;
        // stream time 0 = now: subtitle times in saved files are this plus their offset
        *app.state::<AppState>().board.lock().unwrap() = Some((pipe.board.clone(), chrono::Local::now()));
        pipe.engine.run(SourceIter { source: source.as_mut(), stop });
        Ok(())
    })();
    done.store(true, Ordering::SeqCst);
    let _ = pusher.join();
    pipe.shutdown();
    result
}

fn stop_pipeline(app: &AppHandle) {
    let state = app.state::<AppState>();
    let runner = state.runner.lock().unwrap().take();
    if let Some(r) = runner {
        set_status(app, |s| s.message = "正在停止…".into());
        r.stop.store(true, Ordering::SeqCst);
        // the pipeline finishes pending translations first; a later start joins it
        state.stopping.lock().unwrap().push(r.thread);
    }
}

fn toggle_pipeline(app: &AppHandle) {
    let running = app.state::<AppState>().status.lock().unwrap().running;
    if running {
        stop_pipeline(app);
    } else if let Err(e) = start_pipeline(app) {
        set_status(app, |s| s.error = Some(e));
    }
}

// --- windows -----------------------------------------------------------------------------

/// Where the subtitle window was last left (physical pixels), kept across launches.
#[derive(serde::Serialize, serde::Deserialize)]
struct OverlayGeometry {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

fn overlay_geometry_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("overlay-window.json"))
}

/// Put the subtitle window where it was last time, or (first launch, or that spot is no
/// longer on any screen) centred near the bottom of the main screen, like subtitles.
fn place_overlay(app: &AppHandle) {
    let Some(w) = app.get_webview_window("overlay") else { return };
    let saved: Option<OverlayGeometry> = overlay_geometry_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok());
    if let Some(g) = saved {
        // the middle of its top edge must be on a screen, or the window can't be dragged back
        let on_screen = w.monitor_from_point((g.x + g.width as i32 / 2) as f64, (g.y + 10) as f64).ok().flatten().is_some();
        if on_screen {
            let _ = w.set_size(tauri::PhysicalSize::new(g.width, g.height));
            let _ = w.set_position(tauri::PhysicalPosition::new(g.x, g.y));
            return;
        }
    }
    let Some(m) = w.primary_monitor().ok().flatten().or_else(|| w.current_monitor().ok().flatten()) else { return };
    let area = m.work_area(); // the screen minus the menu bar and the Dock
    let scale = m.scale_factor();
    let margin = (40.0 * scale) as i32;
    let Ok(size) = w.outer_size() else { return };
    let width = size.width.min(area.size.width.saturating_sub(2 * margin as u32));
    let height = size.height;
    if width != size.width {
        let _ = w.set_size(tauri::PhysicalSize::new(width, height));
    }
    let x = area.position.x + (area.size.width as i32 - width as i32) / 2;
    let y = area.position.y + area.size.height as i32 - height as i32 - margin;
    let _ = w.set_position(tauri::PhysicalPosition::new(x, y));
}

fn save_overlay_geometry(window: &tauri::Window) {
    let (Ok(pos), Ok(size)) = (window.outer_position(), window.outer_size()) else { return };
    let g = OverlayGeometry { x: pos.x, y: pos.y, width: size.width, height: size.height };
    if let (Some(p), Ok(text)) = (overlay_geometry_path(window.app_handle()), serde_json::to_string(&g)) {
        let _ = std::fs::write(p, text);
    }
}

fn show_window(app: &AppHandle, label: &str) {
    if let Some(w) = app.get_webview_window(label) {
        let _ = w.show();
        let _ = w.set_focus();
    }
}

fn apply_click_through(app: &AppHandle, on: bool) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("overlay") {
        w.set_ignore_cursor_events(on).map_err(|e| e.to_string())?;
    }
    let item = app.state::<AppState>().tray.lock().unwrap().as_ref().map(|t| t.click_through.clone());
    if let Some(item) = item {
        let _ = item.set_checked(on);
    }
    let _ = app.emit("click-through", on);
    Ok(())
}

fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    let start = MenuItem::with_id(app, "toggle", "开始字幕", true, None::<&str>)?;
    let click = CheckMenuItem::with_id(app, "click_through", "锁定字幕窗口（鼠标穿透）", true, false, None::<&str>)?;
    let show = MenuItem::with_id(app, "show", "显示 / 隐藏字幕", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "设置…", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "退出 UnderText", true, None::<&str>)?;
    let sep = PredefinedMenuItem::separator(app)?;
    let menu = Menu::with_items(app, &[&start, &click, &show, &sep, &settings, &quit])?;
    TrayIconBuilder::with_id("main")
        .icon(app.default_window_icon().cloned().expect("app icon"))
        .tooltip("UnderText")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "toggle" => toggle_pipeline(app),
            "click_through" => {
                let item = app.state::<AppState>().tray.lock().unwrap().as_ref().map(|t| t.click_through.clone());
                let on = item.and_then(|i| i.is_checked().ok()).unwrap_or(false);
                let _ = apply_click_through(app, on);
            }
            "show" => {
                if let Some(w) = app.get_webview_window("overlay") {
                    if w.is_visible().unwrap_or(true) {
                        let _ = w.hide();
                    } else {
                        let _ = w.show();
                    }
                }
            }
            "settings" => show_window(app, "settings"),
            "quit" => {
                stop_pipeline(app);
                // wait off the main thread so the llama-server processes are killed first
                let app = app.clone();
                std::thread::spawn(move || {
                    let pending: Vec<_> = app.state::<AppState>().stopping.lock().unwrap().drain(..).collect();
                    for t in pending {
                        let _ = t.join();
                    }
                    app.exit(0);
                });
            }
            _ => {}
        })
        .build(app)?;
    *app.state::<AppState>().tray.lock().unwrap() = Some(TrayItems { start, click_through: click });
    Ok(())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let dir = app.path().app_config_dir()?;
            let config_path = dir.join("config.toml");
            let first_run = !config_path.exists();
            let config = if first_run {
                let cfg = AppConfig {
                    models_dir: app.path().app_data_dir()?.join("models").display().to_string(),
                    ..Default::default()
                };
                let _ = cfg.save(&config_path);
                cfg
            } else {
                AppConfig::load(&config_path).unwrap_or_else(|e| {
                    log::error!("{e:#}; using defaults");
                    AppConfig::default()
                })
            };
            let setup_needed = first_run || config.required_models().iter().any(|m| !m.2);
            app.manage(AppState {
                config: Mutex::new(config),
                config_path,
                runner: Mutex::new(None),
                stopping: Mutex::new(Vec::new()),
                status: Mutex::new(Status { message: "未开始".into(), ..Default::default() }),
                tray: Mutex::new(None),
                board: Mutex::new(None),
                downloader: Default::default(),
            });
            build_tray(app.handle())?;
            place_overlay(app.handle());
            show_window(app.handle(), "overlay");
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory); // tray app, no Dock icon
            if setup_needed {
                show_window(app.handle(), "settings");
            }
            Ok(())
        })
        .on_window_event(|window, event| match event {
            // closing the settings window only hides it
            tauri::WindowEvent::CloseRequested { api, .. } if window.label() == "settings" => {
                api.prevent_close();
                let _ = window.hide();
            }
            // remember where the subtitle window was put
            tauri::WindowEvent::Moved(_) | tauri::WindowEvent::Resized(_) if window.label() == "overlay" => {
                save_overlay_geometry(window);
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            get_status,
            check_setup,
            list_devices,
            pick_wav,
            pick_folder,
            start,
            stop,
            set_click_through,
            save_transcript,
            open_settings,
            download_plan,
            download_start,
            download_cancel,
            open_for_manual_download
        ])
        .run(tauri::generate_context!())
        .expect("error while running UnderText");
}
