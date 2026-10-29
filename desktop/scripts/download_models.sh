#!/usr/bin/env bash
# Download the models UnderText uses into ../models (next to desktop/).
#
#   scripts/download_models.sh            # everything that is missing
#   HF_ENDPOINT=https://hf-mirror.com scripts/download_models.sh   # Hugging Face mirror
#   MODELS=/path/to/models scripts/download_models.sh
#
# Files that already exist are skipped, so it is safe to run again after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-../models}
HF=${HF_ENDPOINT:-https://huggingface.co}
SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download
mkdir -p "$MODELS"

fetch() {  # url dest
  if [ -e "$2" ]; then echo "  exists: $2"; return; fi
  echo "  downloading $1"
  curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
  mv "$2.part" "$2"
}

fetch_tar() {  # url (a .tar.bz2 that unpacks into a folder of the same name)
  local name; name=$(basename "$1" .tar.bz2)
  if [ -d "$MODELS/$name" ]; then echo "  exists: $MODELS/$name"; return; fi
  fetch "$1" "$MODELS/$name.tar.bz2"
  tar -xjf "$MODELS/$name.tar.bz2" -C "$MODELS"
  rm "$MODELS/$name.tar.bz2"
}

echo "[speech] Parakeet-TDT 0.6B v2 (int8) + Silero VAD"
fetch_tar "$SHERPA/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
fetch "$SHERPA/asr-models/silero_vad.onnx" "$MODELS/silero_vad.onnx"

echo "[speakers] NeMo TitaNet-small"
fetch "$SHERPA/speaker-recongition-models/nemo_en_titanet_small.onnx" "$MODELS/nemo_en_titanet_small.onnx"

echo "[translation] Qwen3.5-2B (Q4_K_M, 1.3 GB)"
fetch "$HF/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf" "$MODELS/Qwen3.5-2B-Q4_K_M.gguf"

echo "[draft] Opus-MT en-zh (CTranslate2)"
mkdir -p "$MODELS/opus-mt-en-zh-ct2"
for f in model.bin shared_vocabulary.json source.spm target.spm; do
  fetch "$HF/gaudi/opus-mt-en-zh-ctranslate2/resolve/main/$f" "$MODELS/opus-mt-en-zh-ct2/$f"
done
echo "done: $MODELS"
