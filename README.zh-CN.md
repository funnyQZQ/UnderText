<p align="center">
  <img src="docs/images/logo.svg" width="300" alt="UnderText">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS on Apple Silicon">
  <a href="https://undertext.leo-draw.xyz/"><img src="https://img.shields.io/badge/website-undertext.leo--draw.xyz-f4b440" alt="官网：undertext.leo-draw.xyz"></a>
</p>

<p align="center"><a href="https://undertext.leo-draw.xyz/"><b>官网</b></a> | <a href="https://undertext.leo-draw.xyz/en/">English website</a> | <a href="../../releases/latest">下载</a></p>

<p align="center"><a href="README.md">English</a> | 简体中文</p>

UnderText 实时显示英文字幕，并在下方给出中文翻译。字幕在一个小窗口里，始终浮在你正在看的内容上面。语音识别、翻译和说话人识别全部在 Mac 本地运行，不会把任何内容发到网上。

<p align="center">
  <img src="docs/images/demo.gif" width="880" alt="UnderText 给一期播客加字幕：左边英文，右边中文，每段标出说话人，最新的几个词还是浅色">
</p>
<p align="center"><sub>在 MacBook Air M4 上录制，左右排列。浅色的字表示还在识别或翻译。</sub></p>

可以给麦克风、WAV 文件加字幕，也可以把声音转到 [BlackHole](https://github.com/ExistentialAudio/BlackHole)，给 Mac 上正在播放的任何内容（Zoom、Teams、YouTube）加字幕。说话人一句话还没说完，中文就出来了。每个说话人有自己的标签（S1、S2…），换人或停顿时自动另起一段。在 MacBook Air 上连续用几个小时的话，打开省电模式能让它保持凉快。

## 工作原理

<p align="center">
  <img src="docs/images/pipeline.svg" width="900" alt="处理流程：音频输入，Silero VAD，Parakeet-TDT，稳定器和分句；Opus-MT 草稿和 Qwen3.5-2B 译文送到字幕板，再显示到字幕窗口或保存成文本；TitaNet 加上说话人标签">
</p>

### 语音转英文

音频按 16 kHz 单声道分块读入。Silero VAD 标出说话开始和结束的位置。静音 0.4 秒后一段结束；如果一直没人停顿，到 20 秒时强制切开。

一段还没结束时，Parakeet-TDT 每 0.5 秒把整段重新识别一遍。如果某一次识别超过 0.5 秒，下一次就等得更久一些，所以识别永远不会落后于音频。一个词在连续两次识别结果里都一样，就变成白色；它后面的词保持灰色，还可能改变。

遇到句号、问号、感叹号，或者一段结束、满 40 个词时，文本被切成句子。

### 英文转中文

每个句子会得到三种翻译：

- **草稿。** Opus-MT 在 CPU 上翻译当前已有的整句（包括灰色的词）。每次新的草稿替换上一次的。
- **续写。** 每新确认 3 个词，或者一个分句以逗号、分号、冒号、破折号结束时，Qwen3.5-2B（llama-server，在 GPU 上）会拿到已确认的英文、前 3 句作为上下文，以及已确认的中文作为回答的开头。它只能在这个开头后面接着写，所以屏幕上已有的字不会被改掉。连续两次回答都一致的中文字被确认，从这时起草稿就隐藏了。
- **定稿。** 句子结束时，Qwen 拿到完整的句子，在已确认的中文后面写完剩下的翻译。

<p align="center">
  <img src="docs/images/translation-timeline.svg" width="900" alt="一个句子随时间的变化：两次 Opus-MT 草稿，然后 Qwen 确认第一个分句，最后是定稿">
</p>

### 说话人与显示

TitaNet 把语音转成声纹，和已经听到过的说话人比较。有人说话时，它每 1.5 秒检查一次最近 1.5 秒的音频，所以已知的说话人在开口约 1.5 秒后就会被标出。一个和谁都不像的声音（相似度低于 0.45），在持续听到约 4.5 秒后会成为新的说话人。一段结束后，会再按 1.5 秒的窗口检查一遍，这样能发现中间没有停顿的换人，必要时修正标签。

换人或停顿 1.2 秒后另起一段。字幕窗口只接收有变化的段落，每秒最多 10 次。**保存**会把上次**清空**之后的全部内容写进一个文本文件，每段都带有真实的时间。

省电模式会改变其中一些数字：Qwen 每 5 个新词调用一次（而不是 3 个），上下文只给 1 句（而不是 3 句），长段落的识别频率也会降低。

## 模型

| 作用 | 模型 | 运行方式 | 大小 | 许可证 |
|---|---|---|---|---|
| 人声检测 | [Silero VAD](https://github.com/snakers4/silero-vad) | sherpa-onnx | 0.6 MB | MIT |
| 语音识别 | [Parakeet-TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)，int8 | sherpa-onnx | 630 MB | CC-BY-4.0 |
| 说话人识别 | [TitaNet-small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small) | sherpa-onnx | 39 MB | NeMo license |
| 草稿翻译 | [Opus-MT en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) | CTranslate2（CPU） | 150 MB | Apache-2.0 |
| 翻译 | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，Q4_K_M | llama-server（GPU） | 1.2 GB | Apache-2.0 |

模型不在这个仓库里。它们存放在本项目自己的 [Hugging Face](https://huggingface.co/funnyQZQ/undertext-models) 和 [ModelScope](https://modelscope.cn/models/funnyqzq/undertext-models) 仓库里。应用第一次启动时会自动下载；从源码编译的话，也可以用下载脚本（见[安装](#安装)）。

## 安装

需要 Apple 芯片的 Mac（M1 及以后）。模型大约占 2.2 GB 磁盘空间。

### 用 .dmg 安装

1. 从 [Releases](../../releases/latest) 页面下载 `.dmg`。
2. 打开它，把 **UnderText** 拖进“应用程序”。
3. 这个应用没有经过 Apple 签名，所以第一次打开会被 macOS 拦下。在“应用程序”里右键点 UnderText，选**打开**，再点一次**打开**。如果还是不行，去**系统设置 → 隐私与安全性**，点**仍要打开**。
4. 第一次启动时会打开设置窗口，列出需要的模型和它们的大小。点**下载**。运行翻译模型的 `llama-server`（来自 llama.cpp）已经包含在应用里，只需要下载模型。应用会测试 Hugging Face 和 ModelScope 的速度，用快的那个；在中国大陆，ModelScope 不需要 VPN。如果两个都连不上，对话框会列出缺少的文件，并提供打开两个下载页和模型文件夹的按钮：手动下载这些文件，按同样的子文件夹放进模型文件夹，再点**重新检查**。
5. 下载完成后，选好要收听的声音来源，点**保存并重新开始**。

模型保存在 `~/Library/Application Support/dev.undertext.desktop/models`。如果下载中断了，在设置窗口里点**下载…**继续；已经下完的文件不会重新下载。

### 从源码编译

需要 [Rust](https://rustup.rs) 和 Homebrew。

```bash
brew install llama.cpp cmake
cd desktop
scripts/download_models.sh        # 保存到 ../models；在国内可以加 MODELS_FROM=modelscope
cargo run --release -p undertext-app
```

自己打包 `.app` 和 `.dmg`（会同时把 llama.cpp 官方编译的 `llama-server` 放进应用里）：

```bash
cargo install tauri-cli --version "^2"
cd app/src-tauri && cargo tauri build      # 输出在 desktop/target/release/bundle/
```

[desktop/README.md](desktop/README.md)（英文）介绍了编译问题的处理，以及 `undertext-cli`：它在终端里运行同样的处理流程，用来测试和长时间运行。

## 性能

在 MacBook Air M4（16 GB，无风扇）上，开着翻译和说话人识别各跑了 30 分钟：一次用默认设置，一次用省电模式。

<p align="center">
  <img src="docs/images/performance.png" width="900" alt="30 分钟内各步骤耗时、字幕延迟、积压和温度状态，默认设置与省电模式对比">
</p>

| | 默认设置 | 省电模式 |
|---|---:|---:|
| Parakeet-TDT，每秒音频耗时 ms | 63 | 40 |
| Opus-MT，每次调用 ms（p50 / p90） | 55 / 121 | 34 / 60 |
| Qwen3.5-2B，每次调用 ms（p50 / p90） | 730 / 1070 | 313 / 494 |
| TitaNet，每次调用 ms（p50 / p90） | 25 / 38 | 11 / 17 |
| 句子结束到最终中文，秒（p50 / p90） | 0.92 / 1.89 | 0.37 / 0.76 |
| 落后实时音频的积压，秒（p50 / p90） | 0.94 / 1.89 | 0.43 / 0.73 |

两次运行都在 3 分钟内进入“fair”温度状态。两种设置下，Opus-MT 草稿都和英文一起出现（中位数不到 0.1 秒）。默认设置下 Qwen 有 69% 的时间在忙，调用耗时在运行过程中变慢约 40%，积压从 1.5 秒涨到 2 秒。省电模式发送更短的提示词、调用更少（Qwen 忙碌 25%），整个过程几乎保持平稳，延迟减半，所以在 MacBook Air 上长时间使用更推荐省电模式。测试方法见 [docs/performance.md](docs/performance.md)（英文）。

## 致谢

UnderText 基于这些项目构建：

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 和 [ONNX Runtime](https://github.com/microsoft/onnxruntime)：语音识别、VAD 和说话人声纹
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：运行翻译模型
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)、[ct2rs](https://github.com/jkawamoto/ctranslate2-rs) 和 [sentencepiece-rust](https://crates.io/crates/sentencepiece-rust)：运行 Opus-MT
- [Tauri](https://tauri.app)：桌面应用
- [cpal](https://github.com/RustAudio/cpal) 和 [hound](https://github.com/ruuda/hound)：音频

模型来自 [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)、[Qwen](https://github.com/QwenLM)、[Helsinki-NLP](https://github.com/Helsinki-NLP/Opus-MT) 和 [Silero](https://github.com/snakers4/silero-vad)。

## 许可证

代码使用 [MIT 许可证](LICENSE)。模型各有自己的许可证，见上表。
