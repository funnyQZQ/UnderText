//! Fast draft translation with Opus-MT (Marian) on CTranslate2, from a converted model
//! folder (`model.bin`, `shared_vocabulary.json`, `source.spm`, `target.spm`).

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ct2rs::sys::{ComputeType, Config, Device, TranslationOptions, Translator as Ct2Translator};
use undertext_core::config::OPUS_MT_FILES;
use undertext_core::translate::{ContextPair, Translator};
use sentencepiece_rust::SentencePieceProcessor;

pub struct OpusMtTranslator {
    translator: Ct2Translator,
    sp_source: SentencePieceProcessor,
    sp_target: SentencePieceProcessor,
    /// The multi-target en-zh model needs a language token on the source side
    /// (`>>cmn_Hans<<` for Simplified Mandarin); empty = none.
    pub target_token: String,
    pub beam_size: usize,
    pub max_decoding_length: usize,
}

impl OpusMtTranslator {
    pub fn from_dir(dir: &Path, target_token: &str, beam_size: usize, threads: usize) -> Result<Self> {
        for f in OPUS_MT_FILES {
            let p = dir.join(f);
            if !p.exists() {
                return Err(anyhow!("Opus-MT model folder is missing {}", p.display()));
            }
        }
        let config = Config {
            device: Device::CPU,
            compute_type: ComputeType::INT8,
            num_threads_per_replica: threads,
            ..Config::default()
        };
        let translator = Ct2Translator::new(dir, &config).with_context(|| format!("loading Opus-MT from {}", dir.display()))?;
        let sp_source = SentencePieceProcessor::open(dir.join("source.spm")).context("loading source.spm")?;
        let sp_target = SentencePieceProcessor::open(dir.join("target.spm")).context("loading target.spm")?;
        Ok(Self {
            translator,
            sp_source,
            sp_target,
            target_token: target_token.to_string(),
            beam_size: beam_size.max(1),
            max_decoding_length: 256,
        })
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let mut pieces = Vec::new();
        if !self.target_token.is_empty() {
            pieces.push(self.target_token.clone());
        }
        pieces.extend(self.sp_source.encode_pieces(text)?);
        pieces.push("</s>".into());
        Ok(pieces)
    }

    pub fn detokenize(&self, pieces: &[String]) -> Result<String> {
        // pieces missing from target.spm decode as unk
        let vocab = self.sp_target.vocab();
        let ids: Vec<i32> = pieces
            .iter()
            .filter(|p| !matches!(p.as_str(), "</s>" | "<pad>" | "<unk>"))
            .map(|p| vocab.piece_to_id(p.as_bytes()).unwrap_or(vocab.unk_id))
            .collect();
        Ok(self.sp_target.decode(&ids)?.trim().to_string())
    }
}

impl Translator for OpusMtTranslator {
    fn translate(&self, text: &str, _context: &[ContextPair]) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }
        let source = vec![self.tokenize(text)?];
        let options = TranslationOptions {
            beam_size: self.beam_size,
            max_decoding_length: self.max_decoding_length,
            ..TranslationOptions::default()
        };
        let results = undertext_core::metrics::DRAFT.time(|| self.translator.translate_batch(&source, &options, None))?;
        let hyp = results
            .first()
            .and_then(|r| r.output())
            .ok_or_else(|| anyhow!("Opus-MT returned no hypothesis"))?;
        self.detokenize(hyp)
    }
}
