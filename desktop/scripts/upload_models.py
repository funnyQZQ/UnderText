#!/usr/bin/env python3
"""Upload the models UnderText downloads to the project's own model repositories, so a
download never depends on someone else's repository staying where it is:

    ModelScope:    https://modelscope.cn/models/funnyqzq/undertext-models  (fast in mainland China)
    Hugging Face:  https://huggingface.co/funnyQZQ/undertext-models        (also on hf-mirror.com)

    pip install -U modelscope          # for --to modelscope
    pip install -U huggingface_hub     # for --to huggingface
    python3 desktop/scripts/upload_models.py --to modelscope
    python3 desktop/scripts/upload_models.py --to huggingface

It asks for an access token with write access unless MODELSCOPE_API_TOKEN / HF_TOKEN is
set (ModelScope: https://modelscope.cn/my/myaccesstoken, Hugging Face:
https://huggingface.co/settings/tokens). This script does not write the token to a file;
the ModelScope SDK may remember the login in ~/.modelscope (delete it to log out).

Files are taken from the repository's models/ folder when their size matches what the
app expects. Otherwise (or when missing) the original is downloaded from hf-mirror.com
first, so the copies are byte-for-byte the files the app downloads. Files already in the
target repository with the right size are skipped, so it can be run again after an
interruption. Uploading to Hugging Face from mainland China needs a VPN.
"""

import argparse
import getpass
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

TARGETS = {
    "modelscope": ("funnyqzq/undertext-models", "https://modelscope.cn/models/funnyqzq/undertext-models/files"),
    "huggingface": ("funnyQZQ/undertext-models", "https://huggingface.co/funnyQZQ/undertext-models/tree/main"),
}
HF_MIRROR = "https://hf-mirror.com"

# path in the models folder -> (original Hugging Face repo, file, size in bytes); same as download.rs
FILES = {
    "Qwen3.5-2B-Q4_K_M.gguf": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 1_280_835_840),
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx": ("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "encoder.int8.onnx", 652_184_296),
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx": ("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "decoder.int8.onnx", 7_257_753),
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx": ("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "joiner.int8.onnx", 1_739_080),
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt": ("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "tokens.txt", 9_384),
    "silero_vad.onnx": ("csukuangfj/vad", "silero_vad.onnx", 643_854),
    "nemo_en_titanet_small.onnx": ("csukuangfj/speaker-embedding-models", "nemo_en_titanet_small.onnx", 40_257_283),
    "opus-mt-en-zh-ct2/config.json": ("gaudi/opus-mt-en-zh-ctranslate2", "config.json", 215),
    "opus-mt-en-zh-ct2/model.bin": ("gaudi/opus-mt-en-zh-ctranslate2", "model.bin", 155_502_615),
    "opus-mt-en-zh-ct2/shared_vocabulary.json": ("gaudi/opus-mt-en-zh-ctranslate2", "shared_vocabulary.json", 1_303_887),
    "opus-mt-en-zh-ct2/source.spm": ("gaudi/opus-mt-en-zh-ctranslate2", "source.spm", 806_435),
    "opus-mt-en-zh-ct2/target.spm": ("gaudi/opus-mt-en-zh-ctranslate2", "target.spm", 804_600),
}

ROOT = Path(__file__).resolve().parents[2]  # desktop/scripts/ -> repository root
MODELS = ROOT / "models"
README = Path(__file__).resolve().parent / "modelscope" / "README.md"


def mb(n):
    return f"{n / 1e6:,.1f} MB"


def fetch(repo, name, size, dest):
    """Download one original file from hf-mirror.com and check its size."""
    url = f"{HF_MIRROR}/{repo}/resolve/main/{name}"
    print(f"  downloading the original from {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {mb(done)} / {mb(size)}", end="", flush=True)
    print()
    if dest.stat().st_size != size:
        sys.exit(f"  {name}: got {dest.stat().st_size} bytes, expected {size}. Stopping.")


class ModelScope:
    name = "ModelScope"

    def __init__(self, repo):
        try:
            from modelscope.hub.api import HubApi
        except ImportError:
            sys.exit("Install the ModelScope SDK first:  pip install -U modelscope")
        token = os.environ.get("MODELSCOPE_API_TOKEN") or getpass.getpass("ModelScope access token (not shown): ").strip()
        self.api, self.repo = HubApi(), repo
        self.api.login(token)

    def sizes(self):
        files = self.api.get_model_files(self.repo, recursive=True)
        return {f.get("Path"): f.get("Size") for f in files}

    def upload(self, src, rel, message):
        self.api.upload_file(path_or_fileobj=str(src), path_in_repo=rel, repo_id=self.repo, commit_message=message)


class HuggingFace:
    name = "Hugging Face"

    def __init__(self, repo):
        try:
            from huggingface_hub import HfApi
        except ImportError:
            sys.exit("Install the Hugging Face SDK first:  pip install -U huggingface_hub")
        token = os.environ.get("HF_TOKEN")
        if token:
            print("Using the token in HF_TOKEN (run with HF_TOKEN= in front to type another one)")
        else:
            token = getpass.getpass("Hugging Face access token, with write access (not shown): ").strip()
        self.api, self.repo = HfApi(token=token), repo
        print(f"Logged in to Hugging Face as {self.api.whoami()['name']}")

    def sizes(self):
        return {f.path: getattr(f, "size", None) for f in self.api.list_repo_tree(self.repo, recursive=True)}

    def upload(self, src, rel, message):
        self.api.upload_file(path_or_fileobj=str(src), path_in_repo=rel, repo_id=self.repo, commit_message=message)


def upload(hub, src, rel, message):
    try:
        hub.upload(src, rel, message)
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            sys.exit(f"\n{hub.name} refused the upload (403 Forbidden): the token can log in but has no write access.\n"
                     "Create a token with write access and run again.\n"
                     "令牌没有写入权限。请新建一个有写入权限的令牌（Hugging Face 选 Write），再运行一次。")
        raise


def main():
    ap = argparse.ArgumentParser(description="Upload UnderText's models to ModelScope or Hugging Face.")
    ap.add_argument("--to", choices=sorted(TARGETS), required=True)
    args = ap.parse_args()
    if not README.is_file():
        sys.exit(f"Missing {README}")

    repo, page = TARGETS[args.to]
    hub = (ModelScope if args.to == "modelscope" else HuggingFace)(repo)

    def remote_sizes():
        try:
            return hub.sizes()
        except Exception as e:  # empty repo or listing not available: upload everything
            print(f"(could not list the repo: {e})")
            return {}

    have = remote_sizes()
    tmp = Path(tempfile.mkdtemp(prefix="undertext-upload-"))
    total = sum(s for _, _, s in FILES.values())
    print(f"Uploading {len(FILES)} model files ({mb(total)}) to {hub.name}: {repo}\n")

    for rel, (orig_repo, name, size) in FILES.items():
        print(rel)
        if have.get(rel) == size:
            print(f"  already on {hub.name}, skipped")
            continue
        src = MODELS / rel
        if src.is_file() and src.stat().st_size == size:
            print(f"  using {src}")
        else:
            why = "missing" if not src.is_file() else f"{mb(src.stat().st_size)}, not the expected {mb(size)}"
            print(f"  local copy {why}")
            src = tmp / rel.replace("/", "__")
            fetch(orig_repo, name, size, src)
        upload(hub, src, rel, f"Add {rel}")
        print("  uploaded")

    upload(hub, README, "README.md", "Update README")
    print("README.md uploaded")

    # check what ended up there
    have = remote_sizes()
    if have:
        bad = [rel for rel, (_, _, size) in FILES.items() if have.get(rel) != size]
        if bad:
            sys.exit(f"These files are missing or have the wrong size on {hub.name}:\n  " + "\n  ".join(bad))
        print(f"\nAll {len(FILES)} files are on {hub.name} with the right sizes.")
    print(page)


if __name__ == "__main__":
    main()
