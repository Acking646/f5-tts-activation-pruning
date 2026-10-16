# F5-TTS Activation Pruning

基于 **F5-TTS v1** 的真实语音复现、微调、推理评测与激活引导结构化裁剪项目。它把 EMNLP 2025 Findings 论文 *Unveiling Multimodal Processing: Exploring Activation Patterns in Multimodal LLMs for Interpretability and Efficiency* 中“稠密 FFN → 细粒度专家 → 激活频率 → 全局非均匀裁剪”的思路迁移到 TTS DiT，并把结果约束在可复核的 RTF、质量和盲听协议上。

> 当前仓库已经完成工程实现和无权重单元测试；尚未在本机完成 GPU 微调、正式 RTF 或人工 MOS，因此不提供虚构结果。真实数值应由固定数据、权重和硬件运行后提交到 `docs/RESULTS.md`。

## 为什么选 F5-TTS

检索与对比了当前开源中文/多语种方案：

| 项目 | 公开时间/定位 | 本项目取舍 |
|---|---|---|
| [F5-TTS](https://github.com/SWivid/F5-TTS) | ACL 2025，非自回归 flow matching；官方含微调、评测和 TensorRT-LLM 路径 | **采用**。DiT 中有规则的 FFN，适合把论文的专家激活分析迁移过来；官方 v1 也便于复现 |
| [CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice) | 2025，多语种 zero-shot、流式与 post-training | 质量和工程能力强，但链路更复杂，不利于先把个人改动和裁剪因果讲清楚 |
| [IndexTTS2](https://github.com/index-tts/index-tts) | 2025，自回归、情感与时长可控 | 适合情感/时长控制项目，不是当前 FFN 稀疏化实验的最小可复现基线 |
| [VITS](https://github.com/jaywalnut310/vits) | 经典单阶段 TTS | 适合学习 VAE、flow、对齐与 GAN；但不是“最新项目”，也不能再写成“VITS 外接 HiFi-GAN” |

F5-TTS 论文报告的是论文设置结果，官方仓库也列出 PyTorch 与 TensorRT 路径的示例 RTF；它们都不是本仓库的实测结果。本项目要求在自己的硬件上重新运行并提交逐条记录。

## 做了什么

- 真实语音数据入口：检查 WAV、时长、重复 ID/文本，以 `speaker/source_recording` 分组划分，降低相邻切片泄漏。
- 官方 F5-TTS v1 推理封装：固定参考音频、参考文本、seed、NFE 和 speed。
- 严谨 RTF：预热、CUDA 同步、逐条 latency/音频时长、总耗时除以总音频时长、软硬件环境快照。
- 可复核 MOS：生成匿名随机 trial，校验 1–5 自然度评分，汇总评分数、听者数、均值和标准差。
- 论文迁移：对 F5-TTS DiT FFN 第一投影权重做平衡 K-means，记录 GELU 后专家激活并进行层内 Z-score、跨层 top-K。
- 两个裁剪对照：`global` 跨层非均匀保留和 `equal` 每层等比例保留。
- 结构化裁剪：同时缩小 FFN 的两层线性矩阵，实际减少参数和矩阵乘；不把零掩码或“活跃参数下降”冒充真实加速。
- 完整产物：profile、mask、逐条 benchmark、固定测试文本、盲听原始记录和结果表模板。

## 项目结构

```text
src/car_tts/
  backend.py       # F5-TTS/CI mock 后端
  dataset.py       # 清洗、分组划分、F5 metadata.csv
  benchmark.py     # 端到端 RTF
  experts.py       # 平衡聚类、激活统计、global/equal 结构化裁剪
  mos.py           # 盲听 trial 与汇总
scripts/
  finetune.ps1
  profile_experts.py
  prune_benchmark.py
data/
  manifest.example.jsonl
  eval_texts.jsonl
docs/
  PROTOCOL.md
  RESULTS.md
```

根目录保留的 `llava-*` 与 `Omni-*` 脚本是论文早期 MLLM 实验材料，不属于 F5-TTS 运行链路；新的迁移实现集中在 `src/car_tts/experts.py`，以免把多模态理解代码误称为 TTS 代码。

## 1. 安装

建议 Linux + Python 3.10/3.11 + NVIDIA GPU；先按自己的 CUDA 版本安装 PyTorch，再安装本项目。

```bash
git clone https://github.com/Acking646/f5-tts-activation-pruning.git
cd f5-tts-activation-pruning
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[tts,test]"
pytest -q
```

Windows PowerShell 激活命令为 `.\.venv\Scripts\Activate.ps1`。第一次真实推理会由 F5-TTS 下载官方权重；模型许可证、语料许可证和生成语音使用边界以其上游说明为准。

## 2. 准备真实语音数据

不要把私人录音或无再分发许可的数据提交到 Git。将音频放进已忽略的 `data/private/`，按 `data/manifest.example.jsonl` 建清单：

```json
{"id":"speaker01_0001","audio":"private/wavs/0001.wav","text":"今天天气很好。","speaker":"speaker01","source_recording":"session01"}
```

运行：

```bash
car-tts prepare data/my_manifest.jsonl artifacts/dataset --seed 42
```

产物包括 `train.jsonl`、`valid.jsonl`、`test.jsonl`、`rejected.jsonl` 和供官方 F5-TTS 预处理器使用的 `metadata.csv`。必须人工检查转写、长静音、爆音、截断、采样率和授权；自动脚本不能替代听检。

## 3. 复现官方推理基线

准备 3–10 秒干净参考音频，修改 `data/eval_texts.jsonl` 中的 `ref_audio/ref_text`：

```bash
car-tts infer \
  --ref-audio data/private/reference.wav \
  --ref-text "这是用于克隆音色的参考语音。" \
  --text "前方道路施工，请减速慢行。" \
  --output outputs/demo.wav

car-tts benchmark data/eval_texts.jsonl outputs/baseline \
  --warmup 1 --repeats 3 --nfe-step 32 --seed 42
```

主指标包含 WAV 写盘，是端到端 RTF；模型加载不计入。若要报告模型-only RTF，需要另设清晰计时边界，不能与本结果混写。

## 4. 微调

官方预处理入口要求 `audio_file|text` 两列。先运行第 2 步，再在安装了 F5-TTS 的环境执行：

```powershell
.\scripts\finetune.ps1 `
  -DatasetName car_voice_pinyin `
  -MetadataCsv (Resolve-Path artifacts/dataset/metadata.csv) `
  -LearningRate 0.00001 -Epochs 10 -BatchFrames 1600
```

`BatchFrames` 必须按显存调整。训练前记录上游 F5-TTS 版本、官方 checkpoint 来源和哈希；训练后保留 TensorBoard、每个 split 的条数/时长、超参数、最佳 checkpoint 的选择依据。小数据说话人适配效果取决于录音质量和覆盖度，仓库不预先承诺提升。

## 5. 从论文迁移专家激活分析

先在固定、覆盖短句/长句/数字/中英混合/标点的清单上采集 profile：

```bash
python scripts/profile_experts.py data/eval_texts.jsonl artifacts/expert_profile.json \
  --experts 64 --top-fraction 0.20 --nfe-step 32 --seed 42
```

然后在同一清单、硬件、精度和参数下做结构化裁剪：

```bash
python scripts/prune_benchmark.py \
  artifacts/expert_profile.json data/eval_texts.jsonl outputs/pruned-global \
  --retain 0.80 --strategy global --warmup 1 --repeats 3

python scripts/prune_benchmark.py \
  artifacts/expert_profile.json data/eval_texts.jsonl outputs/pruned-equal \
  --retain 0.80 --strategy equal --warmup 1 --repeats 3
```

必须再与未裁剪基线比较 CER/WER、说话人相似度和盲听。若 RTF 未下降，应如实分析 attention、ODE、声码器、kernel shape 与 I/O 占比；不能只依据参数量得出“推理加速”。结构裁剪后的 `state_dict` 需要配套裁剪结构才能重新加载，不能直接塞回官方原始配置。

## 6. 小规模自然度盲听

先把各系统输出整理为 JSONL，每行至少有同一 `id` 和 `audio`：

```bash
car-tts mos-create \
  --system baseline=artifacts/mos/baseline.jsonl \
  --system finetuned=artifacts/mos/finetuned.jsonl \
  --system pruned=artifacts/mos/pruned.jsonl \
  --output artifacts/mos/trials.jsonl --seed 42
```

隐藏 `system` 字段后把 `trial_id/audio` 提供给听者，收集 `listener_id,trial_id,naturalness`，然后：

```bash
car-tts mos-aggregate artifacts/mos/ratings.csv artifacts/mos/trials.jsonl
```

本工具是小规模实验的数据骨架，不自动使实验满足正式 MOS 规范。完整口径见 `docs/PROTOCOL.md`，主观语音质量测试可参考 [ITU-T P.800](https://www.itu.int/rec/T-REC-P.800/en)。

## 结果与可写进简历的边界

只有在实际跑完相应阶段后再写：

- 跑通官方权重：`完成 F5-TTS v1 预训练模型推理复现与固定测试集评测`。
- 有参数更新、日志和 checkpoint：`在自建/授权的 X 小时语料上完成说话人微调`。
- 有同机基线和逐条记录：`端到端 RTF 从 X 降至 Y`。
- 有匿名听测原始记录：`N 名听者、M 条语音的小规模自然度 MOS 为 X±Y`。

目前准确表述是：`实现 F5-TTS 数据清洗、微调入口、RTF/MOS 评测与基于 FFN 激活频率的结构化裁剪实验框架；4 项自动化测试通过。` 不应写 TensorFlow、QLoRA、DPO/GRPO、ONNX 或 TensorRT，除非后续确实实现并评测。

## 论文与来源

- Chuan Wu et al. *Unveiling Multimodal Processing: Exploring Activation Patterns in Multimodal LLMs for Interpretability and Efficiency*. Findings of EMNLP 2025.
- Yushen Chen et al. [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://aclanthology.org/2025.acl-long.313/). ACL 2025.
- Jaehyeon Kim et al. [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://proceedings.mlr.press/v139/kim21f.html). ICML 2021.
- Jungil Kong et al. [HiFi-GAN](https://proceedings.neurips.cc/paper/2020/hash/c5d736809766d46260d816d8dbc9eb44-Abstract.html). NeurIPS 2020.

本仓库没有复制 F5-TTS 权重或源码；运行时通过其公开 Python API 调用，并保留清楚的上游归属。
