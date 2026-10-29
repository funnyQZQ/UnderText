//! Model and device adapters for the subtitle pipeline.
//!
//! * `sherpa`:   Silero VAD, Parakeet-TDT recognizer, speaker embeddings (sherpa-onnx)
//! * `opus`:     Opus-MT draft translator on CTranslate2 (feature `opus-mt`)
//! * `llama`:    managed `llama-server` processes and the HTTP completion client
//! * `audio`:    WAV files, microphone / loopback capture, playback (cpal)
//! * `pipeline`: builds a ready-to-run `Engine` from an `AppConfig`

pub mod audio;
pub mod llama;
#[cfg(feature = "opus-mt")]
pub mod opus;
pub mod pipeline;
pub mod sherpa;
