//! Engine-independent core of the real-time English -> Chinese subtitle pipeline.

pub mod align;
pub mod asr;
pub mod board;
pub mod compare;
pub mod config;
pub mod controller;
pub mod diarization;
pub mod engine;
pub mod eventlog;
pub mod local_agreement;
pub mod metrics;
pub mod paragraphs;
pub mod power;
pub mod report;
pub mod sentences;
pub mod soak;
pub mod stabilizer;
pub mod text;
pub mod translate;
pub mod vad;
pub mod vad_chunked;
pub mod workers;

pub use asr::{AsrBackend, AsrUpdate, OfflineRecognizer};
pub use board::{SentenceDisplay, SentenceView, SubtitleBoard};
pub use config::AppConfig;
pub use paragraphs::{Paragraph, ParagraphConfig};
pub use controller::{ControllerConfig, TranslationController};
pub use engine::Engine;
pub use sentences::{SentenceEvent, SentenceTracker};
