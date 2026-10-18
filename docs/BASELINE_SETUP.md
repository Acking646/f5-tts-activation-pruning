# F5-TTS 基线跑通指南（Windows + RTX 4060 Laptop 8GB）

本指南针对当前机器：NVIDIA 驱动 566.24、CUDA Driver API 12.7、8GB 显存。当前系统 Python 3.7 不满足 F5-TTS 要求，且尚未安装 FFmpeg，因此必须建立独立环境。第一阶段只跑官方预训练推理，不需要下载训练数据集。

## 0. 三类文件不要混淆

| 文件 | 用途 | 跑推理基线是否需要 |
|---|---|---|
| 官方 checkpoint + vocab + Vocos | F5-TTS 模型与声码器 | 需要；首次运行可自动下载 |
| 参考音频 + 精确转写 | 告诉 zero-shot 模型目标音色 | 需要；官方示例已附英文参考音频，也可换成自己的中文录音 |
| 音频—文本训练集 | 参数微调 | 不需要；完成基线后再准备 |

## 1. 安装独立环境

在 PowerShell 执行：

```powershell
winget install --id CondaForge.Miniforge3 -e
```

关闭并重新打开 PowerShell，然后：

```powershell
conda create -n f5tts python=3.11 -y
conda activate f5tts
conda install -c conda-forge ffmpeg -y
python -m pip install --upgrade pip setuptools wheel
```

当前 566.24 驱动不建议直接使用官方示例中的 CUDA 12.8 wheel。先使用 F5-TTS README 同样列出的 CUDA 12.4 路线：

```powershell
python -m pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 `
  --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip install --force-reinstall numpy==1.26.4
cd C:\Users\LENOVO\Desktop\tts\code
python -m pip install -e ".[tts]"
```

PyTorch 2.4.0 的 Windows wheel 使用 NumPy 1.x ABI；若环境中已有 NumPy 2.x，会出现 `_ARRAY_API not found`。这里显式固定 1.26.4，避免导入警告和潜在崩溃。

验证环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
ffmpeg -version
f5-tts_infer-cli --help
```

预期 `torch.cuda.is_available()` 为 `True`，设备名包含 `RTX 4060`。

## 2. 下载模型并跑官方最小基线

推荐先让 F5-TTS 自动管理模型缓存：

```powershell
f5-tts_infer-cli --model F5TTS_v1_Base `
  --output_dir outputs\upstream `
  --output_file baseline_en.wav
```

该命令使用 F5-TTS 自带英文参考音频和文本。第一次运行会下载：

- `F5TTS_v1_Base/model_1250000.safetensors`，约 1.35GB；
- `F5TTS_v1_Base/vocab.txt`；
- Vocos 24kHz 声码器权重。

默认缓存通常位于 `%USERPROFILE%\.cache\huggingface\hub`。首次下载后不要移动缓存；确认 `outputs\upstream\baseline_en.wav` 能正常播放，再进入中文和评测阶段。

如需显式下载到固定目录：

```powershell
python -m pip install -U huggingface_hub
hf download SWivid/F5-TTS `
  F5TTS_v1_Base/model_1250000.safetensors `
  F5TTS_v1_Base/vocab.txt `
  --local-dir D:\models\F5-TTS
```

调用固定 checkpoint：

```powershell
f5-tts_infer-cli --model F5TTS_v1_Base `
  --ckpt_file D:\models\F5-TTS\F5TTS_v1_Base\model_1250000.safetensors `
  --vocab_file D:\models\F5-TTS\F5TTS_v1_Base\vocab.txt `
  --output_dir outputs\upstream --output_file baseline_local.wav
```

模型权重页面标注为 CC BY-NC 4.0；用于简历研究项目没有问题，但不要默认用于商业产品。

## 3. 准备中文参考音频

zero-shot 基线只需一段参考音频，不是训练集。推荐：

- 单人、无背景音乐、无混响和爆音；
- 5–10 秒，包含完整自然句子；
- 参考转写必须逐字准确，包括数字实际读法；
- 统一成 mono、24kHz、PCM 16-bit WAV；
- 不使用没有授权的他人声音。

转换示例：

```powershell
New-Item -ItemType Directory -Force data\private | Out-Null
ffmpeg -i D:\recordings\reference_raw.m4a -ac 1 -ar 24000 -sample_fmt s16 `
  data\private\reference.wav
```

中文推理：

```powershell
f5-tts_infer-cli --model F5TTS_v1_Base `
  --ref_audio data\private\reference.wav `
  --ref_text "这是用于克隆音色的参考语音。" `
  --gen_text "前方道路施工，请减速慢行。" `
  --output_dir outputs\zh --output_file zh_baseline.wav
```

不要把 `ref_text` 留空；留空会额外加载 ASR，在 8GB 显存上增加资源占用，也会引入转写误差。

## 4. 跑本仓库的固定测试集与 RTF

将 `data/eval_texts.jsonl` 中所有 `ref_audio` 和 `ref_text` 改为真实参考文件与转写，然后：

```powershell
car-tts benchmark data\eval_texts.jsonl outputs\baseline `
  --warmup 1 --repeats 3 --nfe-step 32 --seed 42
```

检查：

- `outputs/baseline/benchmark.json`；
- 每条固定文本的全部 WAV，而不是只挑最好的一条；
- 环境信息、逐条 latency、输出时长和 aggregate RTF。

## 5. 微调数据怎么准备

### 推荐路线：自录或明确授权的单说话人数据

只为跑通流程，可先准备 30–60 分钟；要形成可信的个人项目，建议逐步扩展到 2–5 小时干净、覆盖充分的数据。这是工程建议，不是效果保证。每条音频：

- 1–30 秒，3–12 秒最容易管理；
- mono、24kHz、PCM 16-bit WAV；
- 文本与语音严格一致；
- 录音设备、距离、房间声学尽量一致；
- 保留数字、英文缩写、标点、长短句等真实车载播报场景；
- 按原始录音会话分组划分，避免相邻切片泄漏。

目录示例：

```text
data/private/
  wavs/
    000001.wav
    000002.wav
  manifest.jsonl
```

`manifest.jsonl` 每行：

```json
{"id":"speaker01_000001","audio":"wavs/000001.wav","text":"前方两百米，请在路口右转。","speaker":"speaker01","source_recording":"session01"}
```

运行本项目清洗和分组划分：

```powershell
car-tts prepare data\private\manifest.jsonl artifacts\dataset --seed 42
```

输出：

- `train.jsonl`、`valid.jsonl`、`test.jsonl`；
- `rejected.jsonl`，保存被拒绝样本及原因；
- `metadata.csv`，只包含训练集，格式为官方要求的 `audio_file|text`，音频使用绝对路径。

然后才运行：

```powershell
.\scripts\finetune.ps1 `
  -DatasetName car_voice_pinyin `
  -MetadataCsv (Resolve-Path artifacts\dataset\metadata.csv) `
  -LearningRate 0.00001 -Epochs 10 -BatchFrames 400
```

8GB 显存先从 `BatchFrames=400` 或更低开始，发生 OOM 就继续减小，并使用梯度累积。完整微调比推理占用大得多；如果仍无法稳定运行，换 Linux/WSL2 或 24GB 云端 GPU，不要为了“跑起来”牺牲数据划分和评测。

## 6. 是否要下载 Emilia

不建议为这个项目下载完整 Emilia。官方数据超过 101k 小时、需要先接受访问条款，且原始版本为 CC BY-NC 4.0；它适合预训练研究，不适合 8GB 笔记本上的个人微调。F5-TTS 官方权重已经在 Emilia 中完成预训练。

如果只想验证公开训练管线：

- 英文单说话人：LJSpeech，约 24 小时，F5-TTS 提供 `prepare_ljspeech.py`；
- 多说话人英文：LibriTTS，F5-TTS 提供 `prepare_libritts.py`；
- 中文个人项目：优先用自录/明确授权的小规模单说话人数据，并保留授权与清洗记录。

公开数据不应直接全部参与训练后再把同源片段当作“未见测试集”。

## 7. 建议验收顺序

1. 官方英文示例成功生成 WAV。
2. 自己的中文参考音频成功 zero-shot 合成。
3. 固定五类测试文本完成端到端 RTF。
4. 数据清洗和 train/valid/test 划分可复核。
5. 再进行微调，并与完全相同的 zero-shot 基线比较。
6. 最后运行专家 profile、global/equal 裁剪和盲听评测。

不要一开始就下载完整 Emilia，也不要在基线未跑通时同时排查微调、裁剪和 TensorRT。
