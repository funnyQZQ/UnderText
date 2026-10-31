# UnderText models

[UnderText](https://github.com/funnyQZQ/UnderText) 是一个在 Mac 本地运行的实时英文字幕和中文翻译工具。这个仓库存放它用到的五个模型的副本，这样下载不依赖原作者的仓库一直可用。同样的文件在 [ModelScope](https://modelscope.cn/models/funnyqzq/undertext-models)（中国大陆下载快）和 [Hugging Face](https://huggingface.co/funnyQZQ/undertext-models) 上各有一份。文件和原版完全一样，没有做任何修改。

Copies of the five models used by [UnderText](https://github.com/funnyQZQ/UnderText), a local real-time English subtitle and Chinese translation app for macOS. Keeping them here means downloads don't depend on the original repositories staying available. The same files are on [ModelScope](https://modelscope.cn/models/funnyqzq/undertext-models) (fast in mainland China) and [Hugging Face](https://huggingface.co/funnyQZQ/undertext-models). They are unchanged from the original releases.

## 文件 / Files

| 文件 / File | 用途 / Role | 来源 / Source | 原作者 / Original authors | 许可证 / License |
|---|---|---|---|---|
| `Qwen3.5-2B-Q4_K_M.gguf` | 翻译 / Translation | [unsloth/Qwen3.5-2B-GGUF](https://modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF) | Qwen Team, Alibaba ([Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)); GGUF quantization by Unsloth | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/` | 语音识别 / Speech recognition | [csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8) | NVIDIA ([nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)); ONNX export by the sherpa-onnx project | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| `nemo_en_titanet_small.onnx` | 说话人识别 / Speaker labels | [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) | NVIDIA ([TitaNet-small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small)); ONNX export by the sherpa-onnx project | NeMo Toolkit license ([Apache-2.0](https://github.com/NVIDIA/NeMo/blob/main/LICENSE)) |
| `silero_vad.onnx` | 人声检测 / Voice activity detection | [csukuangfj/vad](https://huggingface.co/csukuangfj/vad) | [Silero Team](https://github.com/snakers4/silero-vad) | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) |
| `opus-mt-en-zh-ct2/` | 草稿翻译 / Draft translation | [gaudi/opus-mt-en-zh-ctranslate2](https://huggingface.co/gaudi/opus-mt-en-zh-ctranslate2) | [Helsinki-NLP](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) (Jörg Tiedemann et al.); CTranslate2 conversion by gaudi | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

每个模型都按它自己的许可证使用和分发，这个仓库没有为它们设定统一的许可证。Parakeet-TDT（CC-BY-4.0）在使用时需要注明 NVIDIA 为原作者。

Each model is used and redistributed under its own license; this repository does not apply a single license to them. Parakeet-TDT (CC-BY-4.0) requires attribution to NVIDIA.

## 目录结构 / Layout

和 UnderText 的模型文件夹一致，下载后可以直接使用：

Same layout as UnderText's models folder:

```
Qwen3.5-2B-Q4_K_M.gguf
sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/
  encoder.int8.onnx
  decoder.int8.onnx
  joiner.int8.onnx
  tokens.txt
nemo_en_titanet_small.onnx
silero_vad.onnx
opus-mt-en-zh-ct2/
  config.json
  model.bin
  shared_vocabulary.json
  source.spm
  target.spm
```

## 引用 / Citations

- Parakeet-TDT: Xu et al., "Efficient Sequence Transduction by Jointly Predicting Tokens and Durations", ICML 2023.
- TitaNet: Koluguri et al., "TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable Convolutions and Global Context", ICASSP 2022.
- Opus-MT: Tiedemann and Thottingal, "OPUS-MT — Building open translation services for the World", EAMT 2020.
