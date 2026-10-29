# Measuring performance

`undertext-cli soak` feeds a recording through the pipeline at real-time speed and logs every stage. `desktop/scripts/perf_report.py` turns the log into tables and line charts.

It measures:

- time per call of speech recognition (Parakeet-TDT), draft translation (Opus-MT), translation (Qwen3.5-2B) and speaker voice prints (TitaNet)
- subtitle latency: English to first Chinese, and end of sentence to final Chinese
- backlog behind the live audio
- macOS thermal state

## Running

From `desktop/`:

```bash
cargo build --release -p undertext-cli
pip install matplotlib

# 60 minutes of speech
ffmpeg -stream_loop -1 -i ../samples/talk.wav -t 3600 -ar 16000 -ac 1 ../samples/long.wav

# default settings, then power saving
./target/release/undertext-cli soak --models ../models ../samples/long.wav --no-play --log ../perf-normal.jsonl
./target/release/undertext-cli soak --models ../models ../samples/long.wav --no-play --power-saving --log ../perf-saving.jsonl

python3 scripts/perf_report.py ../perf-normal.jsonl ../perf-saving.jsonl \
    --labels "Normal" "Power saving" --out ../perf
```

- Connect the power adapter, turn off Low Power Mode and quit other heavy apps.
- Let the Mac cool down for about 15 minutes between runs.
- Repeating one recording keeps the input the same, so a slowdown later in the run comes from the machine, not the speech.
- `--no-play` only mutes the audio; it is still fed in at real-time speed.

The report goes to `../perf/`:

- `summary.md`: tables per run
- `over_time.png`: time per stage, latency, backlog and thermal state per minute
- `buckets.csv`: the numbers behind the charts

## Reading the results

**Time per call.** Mean, p50, p90, p99 and max in ms, plus Busy, the share of the run the stage spent computing.

- Speech recognition is given in ms per second of audio. Divided by 1000 it is the real-time factor and should be well below 1.
- Qwen is called about once per second of speech. A p90 near 1 s means calls start to queue.
- A p99 far above p50 points to occasional stalls.

**Latency.** Use the median and p90. The backlog should stay under about 1 s; if it keeps growing, the Mac can't keep up.

**Start vs end.** Compares the first and last quarter of the run (the first minute, which includes loading the models, is left out). A stage more than 25% slower at the end, together with a fair or serious thermal state, is slowing down from heat.

## Limits

- The Qwen time covers both kinds of call (continuing and finishing a sentence) and includes waiting inside llama-server.
- Silero VAD is not timed separately; its cost is small.
- Report results together with the Mac model, macOS version and whether it was plugged in.
