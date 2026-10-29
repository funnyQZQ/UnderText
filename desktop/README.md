# Building UnderText

Notes for building the app and using the command-line tools. For an overview, see the [main README](../README.md).

## Layout

```
desktop/
├── crates/
│   ├── undertext-core/      pipeline logic, no model or device code
│   ├── undertext-engines/   sherpa-onnx, Opus-MT, llama-server, audio I/O
│   └── undertext-cli/       command-line tools
├── app/
│   ├── src-tauri/           Tauri app: menu bar, windows, commands
│   └── ui/                  overlay.html, settings.html
└── scripts/download_models.sh
```

## Build on macOS

Install the tools:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust, if needed
brew install llama.cpp cmake
```

`llama.cpp` provides `llama-server`, which the app starts by itself. CMake is needed once, to build CTranslate2 from source on the first build (a few minutes).

Download the models into `../models`. Files that already exist are skipped. Set `HF_ENDPOINT` to use a Hugging Face mirror.

```bash
scripts/download_models.sh
```

Build and run:

```bash
cargo run --release -p undertext-app
```

The settings are stored in `~/Library/Application Support/dev.undertext.desktop/config.toml`.

To build a `.app` and `.dmg` (in `target/release/bundle/`):

```bash
cargo install tauri-cli --version "^2"
cd app/src-tauri && cargo tauri build
```

Unsigned builds have to be allowed once in System Settings > Privacy & Security.

To test the `.dmg` the way a new user gets it (no settings, no models, no llama-server), run `scripts/fresh_install.sh reset` before installing and `scripts/fresh_install.sh restore` afterwards. It moves the app's settings folder aside and unlinks Homebrew's llama.cpp; nothing is deleted.

### Build problems

- **sherpa-onnx download fails.** The first build downloads the prebuilt sherpa-onnx library from GitHub (`sherpa-onnx-v2.0.0-osx-arm64-shared-lib.tar.bz2`, about 9 MB). If that fails, download it yourself, put it in `third_party/sherpa-onnx/`, and set `SHERPA_ONNX_ARCHIVE_DIR=$PWD/third_party/sherpa-onnx`.
- **The packaged app quits at launch with `Library not loaded: @rpath/libsherpa-onnx-c-api.dylib` ("different Team IDs").** The app was signed with the hardened runtime, which only loads libraries signed by the same developer; the bundled sherpa-onnx libraries keep their own signature. `tauri.conf.json` turns the hardened runtime off for ad-hoc builds (`"hardenedRuntime": false`). To sign with a Developer ID and notarize, turn it back on and sign the two dylibs in `Contents/Frameworks/` with the same identity.
- **Linker errors about the SDK version.** Point `SDKROOT` at an SDK that matches your linker, e.g. `export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.

On macOS the app links sherpa-onnx dynamically. The ONNX Runtime in the static archive transcribes about 3.5 times slower on Apple Silicon. The two dylibs are copied next to the binaries in `target/release/`, and into `Contents/Frameworks/` in the app bundle.

## Command-line tools

```bash
cargo build --release -p undertext-cli
```

| Command | What it does |
|---|---|
| `check` | Checks that the model files and `llama-server` can be found |
| `devices` | Lists audio input and output devices |
| `run` | Runs the pipeline on a WAV file or a device and prints subtitles in the terminal |
| `report` | Latency per sentence from a `run` log: median and p90 for first Chinese and final Chinese |
| `soak`, `soak-report` | Long real-time run that records how fast each stage is over time |
| `compare` | Puts two runs of the same audio side by side as an HTML page |
| `diarize` | Shows which speaker each part of a recording was given |
| `bench-asr` | Times one speech recognition pass |

Examples:

```bash
./target/release/undertext-cli check --models ../models
./target/release/undertext-cli run --models ../models --wav talk.wav --play --log run.jsonl
./target/release/undertext-cli report run.jsonl
```

### Long runs

`soak` plays a file in real time and logs, every 10 seconds, how long each stage takes and how far processing is behind the audio, plus the macOS thermal state every minute. It plays the audio and labels speakers by default (`--no-play`, `--no-diarize`). Use `-c config.toml` to run with the app's settings and `--minutes 30` to stop early.

```bash
./target/release/undertext-cli soak --models ../models long-talk.wav --log soak.jsonl
./target/release/undertext-cli soak-report soak.jsonl --bucket 5 --csv soak.csv
```

The report shows each stage per 5-minute bucket and compares the start of the run with the end. Stages that got more than 25% slower are marked with `!`.

### Measuring performance

`scripts/perf_report.py` turns `soak` logs into tables of the time per stage and the subtitle latency, plus line charts over time. [docs/performance.md](../docs/performance.md) has the full procedure and how to read the results.

### Comparing translation models

Run `soak` twice on the same audio, the second time with `--commit-model other-model.gguf`, then:

```bash
./target/release/undertext-cli compare a.jsonl b.jsonl --out compare.html
```

The page lines up the sentences of both runs, highlights where the translations differ, flags translations that look too long or too short for the English, and shows GPU time and latency for each run.

## Notes

- The llama-server prompt cache is off by default (`[llama_server] prompt_cache = false`). Qwen3.5 cannot reuse it, and over a long session it grows to several GB. It is worth turning on for full-attention models such as Qwen2.5 or Qwen3.
- System audio needs a virtual device such as BlackHole for now. Native capture with Core Audio process taps (macOS 14.2+) would remove that step.
- `llama-server` is not inside the `.app`. If none is installed, the app downloads the llama.cpp release build on first launch.
- The core and engine crates already build on Windows. A Windows release still needs WASAPI loopback capture and packaging.
