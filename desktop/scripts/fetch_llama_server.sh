#!/usr/bin/env bash
# Put llama-server (from the official llama.cpp macOS release) in third_party/llama.cpp,
# so `cargo tauri build` can copy it into UnderText.app/Contents/MacOS.
#
#   scripts/fetch_llama_server.sh
#
# Runs automatically before bundling (beforeBundleCommand in tauri.conf.json) and does
# nothing when the files are already there. Only llama-server and the libraries it loads
# are kept; they find each other through @loader_path, so they must stay in one folder.
set -euo pipefail

BUILD=b11785
SHA256=55729a3b847d85c340b9083a936a3bbaa1c0ebfdfd0474b16e7bbd56485b28dc
URL="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-macos-arm64.tar.gz"
# names as llama-server refers to them (the release ships them as symlinks to versioned files)
FILES=(llama-server libllama-server-impl.dylib libllama-common.0.dylib libmtmd.0.dylib libllama.0.dylib
       libggml.0.dylib libggml-base.0.dylib libggml-cpu.0.dylib libggml-blas.0.dylib libggml-metal.0.dylib
       libggml-rpc.0.dylib LICENSE)

cd "$(dirname "$0")/.."
OUT=third_party/llama.cpp

complete() {
  [ "$(cat "$OUT/BUILD" 2>/dev/null)" = "$BUILD" ] || return 1
  for f in "${FILES[@]}"; do [ -f "$OUT/$f" ] || return 1; done
}
if complete; then
  echo "llama-server $BUILD already in $OUT"
  exit 0
fi

# reuse the same release if the app downloaded it earlier (also from a fresh_install.sh backup)
SUPPORT="$HOME/Library/Application Support"
for d in "$SUPPORT/dev.undertext.desktop/llama.cpp" "$SUPPORT/dev.undertext.desktop.backup/llama.cpp"; do
  [ -x "$d/llama-server" ] || continue
  "$d/llama-server" --version 2>&1 | grep -q "${BUILD#b}" || continue
  missing=""; for f in "${FILES[@]}"; do [ -e "$d/$f" ] || missing=1; done
  [ -z "$missing" ] || continue
  rm -rf "$OUT" && mkdir -p "$OUT"
  for f in "${FILES[@]}"; do cp -L "$d/$f" "$OUT/$f"; done
  chmod 755 "$OUT/llama-server"
  echo "$BUILD" > "$OUT/BUILD"
  echo "llama-server $BUILD copied from $d (no download needed)"
  exit 0
done

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "Downloading llama.cpp $BUILD (once; kept in $OUT for later builds) ..."
echo "If GitHub is slow, set a proxy first, e.g.  export https_proxy=http://127.0.0.1:7897"
curl -fL --retry 3 -o "$TMP/llama.tgz" "$URL"
echo "$SHA256  $TMP/llama.tgz" | shasum -a 256 -c - >/dev/null || { echo "Checksum mismatch for $URL" >&2; exit 1; }
tar xzf "$TMP/llama.tgz" -C "$TMP"

rm -rf "$OUT" && mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  cp -L "$TMP/llama-$BUILD/$f" "$OUT/$f"   # -L: copy the file a symlink points to
done
chmod 755 "$OUT/llama-server"
echo "$BUILD" > "$OUT/BUILD"
echo "llama-server $BUILD ready in $OUT ($(du -sh "$OUT" | cut -f1))"
