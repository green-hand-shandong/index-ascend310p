# Index-Ascend310P：IndexTeam 模型 × 华为昇腾 Atlas 300I Duo（Ascend 310P3）推理适配合集

**B站 IndexTeam 全家桶在华为昇腾 Atlas 300I Duo（Ascend 310P3，单芯片 44GB）上的推理适配——5 个模型全部实测跑通，含 2 个内核缺陷的定位与修复、70 算子真机支持度实测。**

> English summary at the bottom. / 英文摘要见文末。

---

## 🎯 一句话

在一颗 **Ascend 310P3** 芯片上，把 B站 IndexTeam 的 **IndexTTS-2.5（零样本音色克隆 TTS）、Index-Translate-2B/9B（150 语向翻译）、Index-Homura-2B（音节控制翻译）、Index-1.9B-Chat（对话 LLM）** 全部跑通：语音合成无削波且 ASR 回读可懂（RTF 5.6），翻译/对话输出正确（2~11 tok/s）。

## 🖥️ 硬件与软件环境

| 项 | 配置 |
|---|---|
| 硬件 | Atlas 300I Duo 服务器，**Ascend 310P3**（soc=202），实测使用单芯片（44GB 可用显存） |
| 驱动 | npu-smi 25.5.1 |
| 容器镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-310p`（CANN 9.2.0，Python 3.12.13） |
| 推理框架 | torch 2.10.0 + torch_npu 2.10.0.post7；vLLM 0.30.0（vllm-ascend）；transformers 双栈 5.21 / 4.52.1 隔离共存 |
| 主机 | aarch64 Linux，Docker 29 |

## 📊 成果总表（模型名 × 实测状态）

| 模型 | 用途 | 适配路线 | 实测速度 | 状态 |
|---|---|---|---|---|
| **IndexTTS-2.5** | 零样本音色克隆 TTS（中英日西阿） | torch_npu 全栈（热路径 NPU + 前端 CPU） | RTF 5.6（6.14s 语音 34.6s 合成） | ✅ 跑通，无削波，ASR 回读通过 |
| **Index-Translate-2B** | 150 语向文本/图片翻译底座 | transformers 5.21 + eager + fp16 | 6~8 tok/s，加载 5.9s | ✅ 三向翻译全对 |
| **Index-Translate-9B** | 同上（思考型） | 同上 + `enable_thinking=False` | 2.1~2.3 tok/s，加载 12.5s | ✅ 三向翻译直接输出 |
| **Index-Homura-2B** | 音节控制翻译（配音对口型） | 与 Translate-2B 完全同配方 | 5.8~7.3 tok/s | ✅ **零改动复用** |
| **Index-1.9B-Chat** | 轻量对话 LLM（自研架构） | transformers 4.52 隔离栈 + trust_remote_code | 8~11 tok/s，加载 10.8s | ✅ 对话/问答/翻译正常 |
| Index-Translate-35B-A3B | MoE 翻译 | 同 Qwen3.5 配方，fp16 需 ~70GB → TP2 双芯片 | — | ⏸ 架构路径一致，未实测（磁盘受限） |
| Index-Nailong-2B/9B | 长文档翻译（229K 上下文） | 同 Qwen3.5 配方 | — | ⏸ 预期可用；长上下文受 310P 掩码物化限制（实用 ≤32K） |
| Index-Echo S2TT/S2ST | 语音翻译 | 需 Qwen3-Omni 音频塔 + CosyVoice3 联合适配 | — | ❌ 未做，建议 910 级硬件 |
| Index-anisora | 动漫视频生成 | CogVideoX/Wan2.1 扩散 | — | ❌ 310P 不可行 |
| IndexTTS 1.x / 2 | 早期 TTS | — | — | ⏸ 2.5 已覆盖，无需重复 |

## 🔊 模型实测详情

### IndexTTS-2.5（语音合成）

- **结果**：22.05kHz 语音合成，稳态 **RTF 5.6**（首跑 11.4，含初始化）；峰值 0.49~0.54、RMS 0.075、**零削波**、无 NaN。
- **质量验证**：用 paraformer ASR 回读合成音频，原文「欢迎来到昇腾三一零P语音合成测试，这段语音完全在国产算力上生成」第二遍 100% 还原（第一遍仅“昇腾”被 ASR 识别为同音字，属 ASR 生僻字问题非发音问题）。
- **耗时分解**：GPT 自回归 8.0s + s2mel DiT(25步) 4.8s + BigVGAN 声码 1.6s；模型加载 53~64s。
- **关键工作**：
  1. 70 算子真机探针 → FFT 全家（stft/istft/fft/复数）内核级不支持 → 参考音频前端整体 CPU 化（每请求仅一次，开销可忽略）；
  2. 生成热路径（GPT/DiT/BigVGAN）全 NPU：`transfer_to_npu` 垫片 + 三开关；
  3. **修复 310P op-plugin 内核缺陷：`conv_transpose1d(groups>64)` 静默输出错误**——BigVGAN 抗混叠上采样（C=384/768）命中，导致 87% 采样点饱和。零精度损失等效实现修复（与 CPU 输出 max|d|=0，且比原生内核快 24%）；
  4. 上游仅 3 个文件 ~10 行补丁（wetext 正则化、eager 注意力、前端链 CPU 化），其余全部由 `npu_compat.py` 运行时注入；
  5. 全部辅助模型（w2v-bert-2.0 / BigVGAN / MaskGCT / CAMPPlus）预下载本地化，推理不联网。

### Index-Translate-2B / 9B / Index-Homura-2B（Qwen3.5 底座三兄弟）

- **结果**（三句固定语向抽查，中→英 / 英→中 / 中→日 全对）：

| 模型 | 中→英示例 | 速度 |
|---|---|---|
| Translate-2B | "The Ascend 310 processor is Huawei's AI acceleration card for edge inference." | 6~8 tok/s |
| Translate-9B | "The Ascend 310 processor is an AI acceleration card from Huawei designed for edge inference." | 2.1~2.3 tok/s |
| Homura-2B | 同 2B 水平 | 5.8~7.3 tok/s |

- **要点**：
  1. vLLM 路线当前走不通：`flash_attention_prefill_24` 内核对 head_dim=256 形状 **aicore timeout**（EZ9999，权重加载与多模态 warmup 均正常，2048/4096 上下文均复现），详见 [docs/llm-models-310p.md](docs/llm-models-310p.md)；
  2. transformers eager 路线三件套即可：`allow_internal_format=False` + `jit_compile=False` + `attn_implementation="eager"`，fp16 加载；
  3. **9B 必须传 `enable_thinking=False`**，否则先输出一大段推理再翻译；
  4. **Homura-2B 零改动复用** Translate 配方——同底座微调模型无任何额外工作。

### Index-1.9B-Chat（自研架构）

- **结果**：自我介绍/知识问答/翻译正常，8~11 tok/s。
  - 「你好，请用一句话介绍你自己」→「我是一个由Bilibili研发的大语言模型……」
  - 「把这句话翻译成英文：今天天气真不错」→「The weather is really nice today.」
- **要点**：模型代码零改动（trust_remote_code），但其自定义 modeling 按 2023 年 transformers API 编写——transformers ≥5.x 的 `rope_scaling` 新结构会直接 KeyError。方案：隔离栈 `/opt/tf452`（transformers 4.52.1 + huggingface_hub 0.36.2 + tokenizers 0.21，`pip --target` 纯 Python 包注入），与系统栈 5.21 互不干扰。

## 🐛 发现并修复的 310P 内核缺陷（对本仓库外同样有复用价值）

| 缺陷 | 现象 | 影响 | 修复 |
|---|---|---|---|
| **`conv_transpose1d(groups=C>64)` 输出错误**（op-plugin） | C=8/64 正常，C=384/768 静默错误，无报无警 | BigVGAN 声码输出 87% 饱和爆音 | matmul 等效实现，max\|d\|=0，反超原生内核速度 → [docs/conv-transpose1d-groups-bug.md](docs/conv-transpose1d-groups-bug.md) |
| **`flash_attention_prefill` head_dim=256 aicore timeout**（vllm-ascend 310P 路径） | EZ9999 retCode=0x25，稳定复现 | vLLM 跑 Qwen3.5 被阻塞 | 绕行：transformers eager 路线 → [docs/llm-models-310p.md](docs/llm-models-310p.md) |
| FFT/STFT/复数算子内核缺失 | aclnn 内核直接报错 | 含频域计算的模型 | CPU 化（小张量低频调用可忽略） |
| SDPA 4D+mask ERR01007 | 格式转换错误 | 所有 sdpa 注意力 | `allow_internal_format=False`（causal 场景可用）或 eager |

通用三开关（310P torch_npu 必备）：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu     # .cuda() 代码零改动映射到 npu
torch.npu.config.allow_internal_format = False    # 修复 SDPA ERR01007
torch.npu.set_compile_mode(jit_compile=False)     # 修复 polar/bincount/复数算子调度
```

## 🤔 什么都没改就能跑吗？

严格说**没有**——310P 的硬件特性要求每个模型至少做环境级配置（三开关 + eager 注意力 + fp16）。但**边际成本可以为零**：Qwen3.5 底座一旦跑通一个（Translate-2B），其余同底座模型（Homura-2B、Nailong）**一行代码不用改**，只换模型路径；Translate-9B 也只多一个 `enable_thinking=False` 参数。

## 🧪 测试方法与局限（如实说明）

- **IndexTTS-2.5 测得最完整**：客观指标（峰值/削波率/NaN/过零率）+ **ASR 回读闭环验证可懂度** + 双次合成计时。
- **LLM 为冒烟级抽查**：三句固定语向人工核对译文 + tok/s 计时；**未跑** FloRES/COMET 标准翻译评测、未跑 MMLU/C-Eval、未测长文本稳定性与并发。速度为单请求贪心解码，未做批处理优化。
- 未适配项（Echo/anisora/35B-A3B/Nailong）的结论基于架构分析而非实测，已在总表中注明。

## 🚀 快速开始

环境要求：Atlas 300I Duo（或 200I Pro），Docker，磁盘 ~15GB。

```bash
# 1. 部署（克隆上游 index-tts + 打补丁 + 下载全部模型，国内自动走 ModelScope/hf-mirror）
git clone https://github.com/green-hand-shandong/index-ascend310p.git
cd index-ascend310p
bash scripts/setup_all.sh workspace

# 2. 启动 NPU 容器
docker run -d --name indextts-310p \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $PWD:/workspace -w /workspace \
  quay.io/ascend/vllm-ascend:v0.30.0rc2-310p sleep infinity

# 3. IndexTTS-2.5 依赖 + 语音合成（REF_WAV = 3~15s 干净人声）
bash scripts/docker_exec.sh indextts-310p \
  'pip install transformers==4.52.1 tokenizers==0.21.0 ... -i https://pypi.tuna.tsinghua.edu.cn/simple'   # 完整列表见 scripts/setup_all.sh 末尾
bash scripts/docker_exec.sh indextts-310p \
  'REF_WAV=/workspace/your_voice.wav python tests/run_test.py out.wav "你好，昇腾！"'

# 4. LLM 系（Qwen3.5 底座，系统栈直接跑）
bash scripts/docker_exec_vllm.sh indextts-310p /workspace \
  'python scripts/test_translate_hf.py /workspace/Index-Translate-2B'
```

各模型权重均从 ModelScope / hf-mirror 下载：`IndexTeam/IndexTTS-2.5`（~6GB）、`IndexTeam/Index-Translate-2B`（4.6GB）、`IndexTeam/Index-Translate-9B`（19GB，4 分片）、`IndexTeam/Index-Homura-2B`（4.6GB）、`IndexTeam/Index-1.9B-Chat`（4.4GB）。

## 📁 仓库结构

```
├── npu_compat.py        # NPU 兼容层（三开关 + FFT CPU 化 + convT 缺陷修复，import 即用）
├── patches/             # 对上游 index-tts 的最小补丁（3 文件 ~10 行）
├── scripts/             # 部署脚本 + 各模型测试脚本（TTS / Translate / Homura / 1.9B）
├── tests/               # 端到端测试与音频验证（含 ASR 回读）
├── probes/              # 7 个真机算子探针（复现内核缺陷的完整链路）
└── docs/                # 310P 算子支持度报告 / convT 缺陷分析 / LLM 适配结论
```

## 🗺️ 后续方向

- vLLM 服务化：等 vllm-ascend/CANN 修复 FA 内核，或试 CANN 26 POC 镜像 + W8A8 量化（RFC #7394 路径）
- IndexTTS 性能：图模式 / DiT 蒸馏 / convT 等效实现用 Ascend C 重写
- 35B-A3B 双芯片 TP2 实测、Nailong ≤32K 长文档实测
- 补标准评测（FloRES/COMET、TTS MOS）

## 📄 许可与致谢

- 本仓库（补丁、兼容层、脚本、探针、文档）：Apache-2.0（见 [LICENSE](LICENSE) / [NOTICE](NOTICE)）
- 上游模型与代码：[index-tts](https://github.com/index-tts/index-tts)（Bilibili Model Use License），本仓库仅以 diff 分发修改；权重见各 ModelScope/HF 仓库
- 依赖权重：[nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)、[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)、[amphion/MaskGCT](https://huggingface.co/amphion/MaskGCT)、[funasr/campplus](https://huggingface.co/funasr/campplus)

> 注：GitCode 仓库 URL path 因平台限制仍为 `indextts-2.5-ascend310p`（显示名已更新），GitHub 为本仓库主地址且旧链接自动重定向。

---

## English Summary

**All five flagship models from Bilibili's IndexTeam run end-to-end on a single Huawei Ascend 310P3 chip (Atlas 300I Duo, 44GB)**:

| Model | Task | Result |
|---|---|---|
| IndexTTS-2.5 | Zero-shot voice-cloning TTS | RTF 5.6, zero clipping, ASR-roundtrip verified |
| Index-Translate-2B / 9B | 150-language translation | 6~8 / 2.1~2.3 tok/s, all translation directions correct |
| Index-Homura-2B | Syllable-controlled translation (dubbing) | 5.8~7.3 tok/s, **zero-change reuse** of the 2B recipe |
| Index-1.9B-Chat | Conversational LLM | 8~11 tok/s, via isolated transformers 4.52 stack |

Highlights: on-device 70-operator support matrix; **discovered & fixed an op-plugin kernel bug where `conv_transpose1d(groups>64)` silently returns wrong results** (zero-loss workaround, faster than native); documented the vLLM-Ascend `flash_attention_prefill` head_dim=256 aicore-timeout that currently blocks Qwen3.5 serving on 310P; everything works through a thin runtime compat layer plus ~10 lines of upstream patches. Scripts, probes and docs in this repo are fully reproducible.
