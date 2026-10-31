#!/usr/bin/env bash
# Download the models UnderText uses into ../models (next to desktop/), from the project's
# own model repository (the same files the app downloads on first launch).
#
#   scripts/download_models.sh                          # from Hugging Face
#   MODELS_FROM=modelscope scripts/download_models.sh   # from ModelScope (fast in mainland China)
#   MODELS=/path/to/models scripts/download_models.sh
#
# Files that already exist with the right size are skipped and interrupted downloads
# resume, so it is safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-../models}
case "${MODELS_FROM:-huggingface}" in
  huggingface) BASE=https://huggingface.co/funnyQZQ/undertext-models/resolve/main ;;
  modelscope)  BASE=https://modelscope.cn/models/funnyqzq/undertext-models/resolve/master ;;
  *) echo "MODELS_FROM must be huggingface or modelscope" >&2; exit 1 ;;
esac

# path in the models folder and size in bytes (same list as app/src-tauri/src/download.rs)
FILES="
Qwen3.5-2B-Q4_K_M.gguf 1280835840
sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx 652184296
sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx 7257753
sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx 1739080
sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt 9384
silero_vad.onnx 643854
nemo_en_titanet_small.onnx 40257283
opus-mt-en-zh-ct2/config.json 215
opus-mt-en-zh-ct2/model.bin 155502615
opus-mt-en-zh-ct2/shared_vocabulary.json 1303887
opus-mt-en-zh-ct2/source.spm 806435
opus-mt-en-zh-ct2/target.spm 804600
"

size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }  # GNU, then BSD/macOS

echo "$FILES" | while read -r rel bytes; do
  [ -n "$rel" ] || continue
  dest="$MODELS/$rel"
  if [ -f "$dest" ] && [ "$(size "$dest")" = "$bytes" ]; then
    echo "  ok: $rel"
    continue
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  downloading $rel"
  curl -fL --retry 3 --progress-bar -C - -o "$dest.part" "$BASE/$rel"
  if [ "$(size "$dest.part")" != "$bytes" ]; then
    echo "  $rel: got $(size "$dest.part") bytes, expected $bytes. Run again to download it afresh." >&2
    rm -f "$dest.part"
    exit 1
  fi
  mv "$dest.part" "$dest"
done
echo "done: $MODELS"
