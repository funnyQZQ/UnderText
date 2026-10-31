//! Chinese wording for messages that come from the shared engine crates.
//!
//! The engines (also used by `undertext-cli`) report progress, warnings and errors in
//! English. The app shows them in Chinese: known phrases are replaced, and anything not
//! listed (paths, library messages) is passed through unchanged.

/// Progress messages from `pipeline::build`.
pub fn status(msg: &str) -> String {
    if let Some(model) = msg.strip_prefix("starting translation model (").and_then(|m| m.strip_suffix(')')) {
        return format!("正在启动翻译模型（{model}）…");
    }
    match msg {
        "loading speech models ..." => "正在加载语音模型…".into(),
        "loading the translation model ..." => "正在加载翻译模型…".into(),
        _ => replace_all(msg, STATUS),
    }
}

/// Warnings from `pipeline::build` (the run continues).
pub fn warning(msg: &str) -> String {
    if let Some(path) = msg.strip_prefix("draft model missing (").and_then(|m| m.strip_suffix("), running without drafts")) {
        return format!("缺少草稿模型（{path}），本次不显示即时草稿");
    }
    if msg.starts_with("this build has no Opus-MT support") {
        return "这个版本不支持 Opus-MT，本次不显示即时草稿".into();
    }
    msg.to_string()
}

/// An error chain as printed by `{:#}` ("context: cause: cause").
pub fn error(msg: &str) -> String {
    // the one multi-line error that lists files
    if let Some(rest) = msg.strip_prefix("missing model files (models_dir = ") {
        if let Some((dir, list)) = rest.split_once("):\n") {
            return format!("缺少模型文件（模型文件夹：{dir}）：\n{list}\n请在设置里点“下载…”下载模型。");
        }
    }
    replace_all(msg, ERRORS)
}

const STATUS: &[(&str, &str)] = &[];

/// Phrases at the start of a message or of one part of an error chain.
const ERRORS: &[(&str, &str)] = &[
    ("opening ", "无法打开 "),
    ("no default input device", "找不到默认输入设备"),
    ("no default output device", "找不到默认输出设备"),
    ("no input device matching ", "找不到这个输入设备："),
    ("no output device matching ", "找不到这个输出设备："),
    ("unsupported input sample format ", "不支持的输入采样格式 "),
    ("unsupported output sample format ", "不支持的输出采样格式 "),
    ("Opus-MT model folder is missing ", "Opus-MT 模型文件夹里缺少 "),
    ("loading Opus-MT from ", "无法加载 Opus-MT："),
    ("loading source.spm", "无法加载 source.spm"),
    ("loading target.spm", "无法加载 target.spm"),
    ("Opus-MT returned no hypothesis", "Opus-MT 没有返回翻译结果"),
    ("no WAV file selected", "还没有选择 WAV 文件（设置 → 音频来源）"),
    ("llama-server not found at ", "找不到 llama-server："),
    (
        "llama-server not found. Install it (macOS: `brew install llama.cpp`) or set llama_server.binary",
        "找不到 llama-server。它本来在 UnderText 应用里，请重新安装应用；从源码运行的话，请用 `brew install llama.cpp` 安装",
    ),
    ("llama-server request failed", "llama-server 请求失败"),
    ("llama-server returned invalid JSON", "llama-server 返回的数据无法解析"),
    ("llama-server exited ", "llama-server 意外退出 "),
    ("llama-server on port ", "llama-server 没有按时启动，端口 "),
    ("model not found: ", "找不到模型："),
    ("model file not found: ", "找不到模型文件："),
    ("failed to create Silero VAD", "无法创建人声检测（Silero VAD）"),
    ("failed to create the ASR recognizer", "无法创建语音识别器"),
    ("failed to create the speaker model", "无法创建说话人识别模型"),
    ("creating ", "无法创建 "),
    ("starting ", "无法启动 "),
    ("reading ", "无法读取 "),
    ("parsing ", "无法解析 "),
    ("unknown template ", "未知的提示词模板 "),
    ("unknown vad ", "未知的 VAD 类型 "),
    ("No such file or directory", "文件或文件夹不存在"),
    ("Permission denied", "没有权限"),
];

/// Replace each phrase where it starts the message or follows ": " / a new line.
fn replace_all(msg: &str, table: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(msg.len() + 16);
    let mut rest = msg;
    let mut at_start = true;
    'outer: while !rest.is_empty() {
        if at_start {
            for (en, zh) in table {
                if let Some(r) = rest.strip_prefix(en) {
                    out.push_str(zh);
                    rest = r;
                    at_start = false;
                    continue 'outer;
                }
            }
        }
        let c = rest.chars().next().unwrap();
        out.push(c);
        rest = &rest[c.len_utf8()..];
        at_start = c == '\n' || out.ends_with(": ");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_messages_in_chinese() {
        assert_eq!(status("loading speech models ..."), "正在加载语音模型…");
        assert_eq!(status("starting translation model (qwen.gguf)"), "正在启动翻译模型（qwen.gguf）…");
        assert_eq!(
            error("opening /a/b.wav: No such file or directory (os error 2)"),
            "无法打开 /a/b.wav: 文件或文件夹不存在 (os error 2)"
        );
        assert!(error("missing model files (models_dir = ~/m):\n  a.onnx").starts_with("缺少模型文件（模型文件夹：~/m）"));
        // paths that happen to contain a phrase are left alone
        assert_eq!(error("model not found: /x/opening y"), "找不到模型：/x/opening y");
        assert_eq!(warning("draft model missing (/m/opus), running without drafts"), "缺少草稿模型（/m/opus），本次不显示即时草稿");
        assert_eq!(error("something else"), "something else");
    }
}
