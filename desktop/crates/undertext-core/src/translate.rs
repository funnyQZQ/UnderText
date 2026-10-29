//! Translator interfaces + LLM prompt construction.
//!
//! Forcing the already-committed Chinese as the start of the assistant turn is what makes
//! incremental ("continue") translation possible: the model only has to extend it.

use anyhow::Result;

/// (english, chinese) pairs of previous sentences, oldest first.
pub type ContextPair = (String, String);

/// Whole-text translator (drafts, and the "refine" final mode).
pub trait Translator: Send + Sync {
    fn translate(&self, text: &str, context: &[ContextPair]) -> Result<String>;
}

/// Translator that can continue from a fixed Chinese prefix; the result must start with
/// `target_prefix`. `partial = true` tells the model the English is unfinished.
pub trait IncrementalTranslator: Translator {
    fn continue_translation(&self, source: &str, target_prefix: &str, context: &[ContextPair], partial: bool) -> Result<String>;
}

pub const DEFAULT_SYSTEM_PROMPT: &str = "你是一名专业的英译中同声传译员。把用户给出的英文翻译成自然、准确的简体中文。\n\
要求：\n\
1. 采用顺句驱动：尽量按英文语序翻译，必要时拆句，不要为了调整语序而等待后文。\n\
2. 只输出中文译文，不要解释，不要加引号或前缀。\n\
3. 专有名词、人名、产品名可保留英文。\n\
4. 如果英文是一个未说完的片段，只翻译已有内容，不要猜测或补全后文。";

/// About a third of the default prompt's tokens. Every LLM call re-reads the whole prompt
/// (Qwen3.5's hybrid layers can't reuse a cached prefix), so this directly cuts GPU work.
pub const COMPACT_SYSTEM_PROMPT: &str = "英译中同声传译。按英文语序译成自然的简体中文，只输出译文；未说完的片段只译已有部分，不要补全。";

/// Chat template expressed as plain strings (so prompts can be prefilled).
#[derive(Debug, Clone, PartialEq)]
pub struct PromptTemplate {
    pub system: &'static str,
    pub user: &'static str,
    pub assistant: &'static str,
    pub stop: &'static [&'static str],
}

impl PromptTemplate {
    pub fn by_name(name: &str) -> Option<Self> {
        Some(match name {
            // Qwen2.5 / Qwen3 / Qwen3.5 and most ChatML models
            "chatml" => Self {
                system: "<|im_start|>system\n{content}<|im_end|>\n",
                user: "<|im_start|>user\n{content}<|im_end|>\n",
                assistant: "<|im_start|>assistant\n",
                stop: &["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
            },
            "llama3" => Self {
                system: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
                user: "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
                assistant: "<|start_header_id|>assistant<|end_header_id|>\n\n",
                stop: &["<|eot_id|>", "<|end_of_text|>"],
            },
            "plain" => Self { system: "{content}\n\n", user: "{content}\n", assistant: "中文：", stop: &["\nEN:", "\n\n"] },
            _ => return None,
        })
    }

    pub fn render(&self, system_text: &str, user_text: &str, assistant_preamble: &str) -> String {
        let mut s = self.system.replace("{content}", system_text);
        s.push_str(&self.user.replace("{content}", user_text));
        s.push_str(self.assistant);
        s.push_str(assistant_preamble);
        s
    }
}

pub fn build_user_message(source: &str, context: &[ContextPair], partial: bool) -> String {
    let mut lines: Vec<String> = Vec::new();
    if !context.is_empty() {
        lines.push("前文（仅供参考，不要重复翻译）：".into());
        for (en, zh) in context {
            lines.push(format!("EN: {en}"));
            lines.push(format!("ZH: {zh}"));
        }
        lines.push(String::new());
    }
    lines.push(if partial {
        "请翻译下面这段尚未说完的英文（只翻译已有部分）：".into()
    } else {
        "请翻译下面这句英文：".into()
    });
    lines.push(source.trim().into());
    lines.join("\n")
}

const LABELS: &[&str] = &["中文", "译文", "翻译", "ZH", "Chinese"];
const QUOTES: &[char] = &['"', '“', '”', '「', '」', '\''];

/// Drop a leading label such as "中文：" that some models put before the translation.
fn strip_label(s: &str) -> &str {
    let t = s.trim_start();
    for label in LABELS {
        let Some(head) = t.get(..label.len()) else { continue };
        if head.eq_ignore_ascii_case(label) {
            let rest = t[label.len()..].trim_start();
            if let Some(r) = rest.strip_prefix(':').or_else(|| rest.strip_prefix('：')) {
                return r.trim_start();
            }
        }
    }
    s
}

/// Combine forced prefix + generated text into a single-line translation.
pub fn clean_continuation(target_prefix: &str, generated: &str) -> String {
    let mut text: String = if target_prefix.is_empty() {
        generated.trim_start().split('\n').next().unwrap_or("").to_string()
    } else {
        generated.split('\n').next().unwrap_or("").to_string()
    };
    if target_prefix.is_empty() {
        text = strip_label(&text).trim().trim_matches(QUOTES).to_string();
    }
    format!("{target_prefix}{text}").trim_end().to_string()
}

/// A text-completion backend (llama-server).
pub trait CompletionBackend: Send + Sync {
    fn complete(&self, prompt: &str, max_tokens: usize, stop: &[String]) -> Result<String>;
}

/// LLM translator built on a completion backend.
pub struct PromptTranslator<B: CompletionBackend> {
    pub backend: B,
    pub template: PromptTemplate,
    pub system_prompt: String,
    pub assistant_preamble: String,
    pub max_tokens: usize,
}

impl<B: CompletionBackend> PromptTranslator<B> {
    pub fn new(backend: B, template: PromptTemplate) -> Self {
        Self { backend, template, system_prompt: DEFAULT_SYSTEM_PROMPT.into(), assistant_preamble: String::new(), max_tokens: 160 }
    }

    pub fn build_prompt(&self, source: &str, context: &[ContextPair], partial: bool) -> String {
        self.template.render(&self.system_prompt, &build_user_message(source, context, partial), &self.assistant_preamble)
    }
}

impl<B: CompletionBackend> Translator for PromptTranslator<B> {
    fn translate(&self, text: &str, context: &[ContextPair]) -> Result<String> {
        self.continue_translation(text, "", context, false)
    }
}

impl<B: CompletionBackend> IncrementalTranslator for PromptTranslator<B> {
    fn continue_translation(&self, source: &str, target_prefix: &str, context: &[ContextPair], partial: bool) -> Result<String> {
        if source.trim().is_empty() {
            return Ok(target_prefix.to_string());
        }
        let prompt = self.build_prompt(source, context, partial) + target_prefix;
        let mut stop: Vec<String> = self.template.stop.iter().map(|s| s.to_string()).collect();
        if !target_prefix.is_empty() {
            stop.push("\n".into()); // mid-line: stop at the first newline
        }
        let generated = self.backend.complete(&prompt, self.max_tokens, &stop)?;
        Ok(clean_continuation(target_prefix, &generated))
    }
}
