# IndexTTS-2.5-Ascend310P

**B站 [IndexTTS-2.5](https://www.modelscope.cn/models/IndexTeam/IndexTTS-2.5) 零样本音色克隆语音合成模型在华为昇腾 Atlas 300I Duo（Ascend 310P3）上的推理适配**

> English summary at the bottom. / 英文摘要见文末。

---

## 🎯 一句话

在**一颗 310P3 芯片**（Atlas 300I Duo，~44GB）上把 IndexTTS-2.5 的**全流程**跑起来：
参考音频 + 文本进 → 克隆音色的语音出，fp32 无削波、ASR 回读可懂，稳态 RTF ≈ 5.6。
**首个公开的 310P 跑通方案**，并在过程中实锤了一个 op-plugin 内核缺陷（分组 convT）。

## ✅ 成果一览

| 项 | 结果 |
|---|---|
| 端到端合成 | ✅ GPT 自回归 + s2mel DiT + BigVGAN 声码器全流程 NPU 推理，输出 22.05kHz 语音 |
| 音频质量 | ✅ rms 0.075 / peak 0.54 / 零削波 / 无 NaN；paraformer ASR 回读文字与输入一致（"昇腾"识别为同音"张腾"，属 ASR 生僻字问题） |
| 性能 | 稳态合成 6.1s 语音耗时 34.6s（GPT 8.0s + DiT 4.8s + BigVGAN 1.6s，RTF 5.6）；加载 ~55s |
| 310P 算子普查 | ✅ 真机探针实测 70 算子：55 原生通过、5 个开关后通过、FFT 家族内核级不支持（CPU 化） |
| 🐛 内核缺陷定位 | ✅ **`conv_transpose1d(groups>64)` 输出错误**（无报错、静默数值错误）——逐层二分定位 + 零损失等效修复 |
| 适配方式 | 对上游代码仅 **3 处共 ~10 行改动**（补丁形式分发），其余全部通过 `npu_compat.py` 运行时注入 |

## 📖 模型背景

IndexTTS-2.5 是 B站 IndexTeam 2026 年开源的零样本 TTS：单条参考音频克隆音色，支持
中/英/日/西/阿五语与跨语言音色迁移，情感与音色解耦可控制。约 0.8B GPT 骨干（自回归 mel token）
+ flow-matching（s2mel DiT，25 步 ODE）+ BigVGAN 声码器，外加 w2v-bert / MaskGCT 语义编解码 /
CAMPPlus 说话人嵌入等辅助模型，组件多、算子面广，对推理硬件兼容性是块硬骨头。

## 🧩 适配架构

```
                     ┌────────────── NPU（生成热路径）──────────────┐
文本前端(CPU/wetext) →│ GPT2 骨干(eager注意力,fp32) → s2mel DiT     │→ 22.05kHz wav
                     │        ↑ BigVGAN(convT缺陷已修复)            │
                     └──────────────────────────────────────────────┘
                     ┌────── CPU（参考音频前端，每请求一次）──────────┐
参考音频 ──────────→ │ w2v-bert → MaskGCT语义codec → CAMPPlus/mel  │→ 条件向量
                     │ （含全部 FFT/STFT/复数算子）                  │
                     └──────────────────────────────────────────────┘
```

三个关键开关（310P 通用，详见 [docs/ascend310p-op-support.md](docs/ascend310p-op-support.md)）：

```python
from torch_npu.contrib import transfer_to_npu          # .cuda() 全局映射到 npu，上游代码零改动
torch.npu.config.allow_internal_format = False         # 修复 SDPA ERR01007
torch.npu.set_compile_mode(jit_compile=False)          # 修复 polar/bincount/复数算子调度
```

## 🐛 发现的内核缺陷（本仓库最有复用价值的部分）

`F.conv_transpose1d(x, w, groups=C)` 在 **C > 64 时输出静默错误**（C=64 正常）。
BigVGAN 的抗混叠上采样（C=384/768）命中此缺陷，导致声码输出 87% 采样点饱和。
逐层二分定位过程与最小复现见 [docs/conv-transpose1d-groups-bug.md](docs/conv-transpose1d-groups-bug.md)，
修复为数学等效实现（与 CPU 原生输出 max|d|=0，且比原生内核更快）。**其他在 310P 上跑
GAN 声码器/扩散上采样头的项目若遇到"无报错但输出全是爆音"，大概率是同一个坑。**

## 🚀 快速开始

环境要求：Atlas 300I Duo（或 200I Pro），Docker，约 10GB 磁盘（模型 ~6GB）。

```bash
# 1. 克隆本仓库并部署（克隆上游代码 + 打补丁 + 下载全部模型）
git clone https://github.com/green-hand-shandong/indextts-2.5-ascend310p.git
cd indextts-2.5-ascend310p
bash scripts/setup_all.sh workspace

# 2. 启动 NPU 容器（vllm-ascend 310P 镜像，只借它的 CANN/torch_npu 环境）
docker run -d --name indextts-310p \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $PWD:/workspace -w /workspace \
  quay.io/ascend/vllm-ascend:v0.30.0rc2-310p sleep infinity

# 3. 装依赖（国内走清华镜像）
bash scripts/docker_exec.sh indextts-310p \
  'pip install transformers==4.52.1 tokenizers==0.21.0 accelerate==1.8.1 \
      torchaudio==2.10.0 librosa soundfile scipy einops omegaconf munch \
      jieba jieba_fast cn2an g2p-en json5 sentencepiece wetext whisper \
      fugashi unidic-lite modelscope addict simplejson sortedcontainers \
      "huggingface_hub==0.36.2" "datasets==3.6.0" pyarrow pandas descript-audiotools \
      -i https://pypi.tuna.tsinghua.edu.cn/simple'

# 4. 合成（REF_WAV 用任意 3~15 秒干净人声）
bash scripts/docker_exec.sh indextts-310p \
  'REF_WAV=/workspace/your_voice.wav python tests/run_test.py out.wav "你好，昇腾！"'
```

> 提示：`setup_all.sh` 已把所有运行期需要的模型（含 w2v-bert-2.0、BigVGAN、MaskGCT、
> CAMPPlus）预下载为本地文件，推理全程不联网。国内网络自动走 ModelScope / hf-mirror 双源。

## 🧪 验证

```bash
# 音频统计 + ASR 回读（funasr paraformer）
bash scripts/docker_exec.sh indextts-310p 'pip install funasr -i https://pypi.tuna.tsinghua.edu.cn/simple'
bash scripts/docker_exec.sh indextts-310p 'python tests/validate_audio.py out.wav out_2.wav'
```

实测输出示例：

```
out_2.wav: sr=22050 dur=6.14s clip%=0.000 rms=0.075 peak=0.541 nan=False
out_2.wav -> 欢 迎 来 到 [昇腾] 三 一 零 p 语 音 合 成 测 试 这 段 语 音 完 全 在 国 产 算 力 上 生 成
```

## 📊 310P 算子支持度速查（IndexTTS 视角）

| 状态 | 算子 | 处理 |
|---|---|---|
| ✅ 原生 | conv1d / matmul / eager注意力 / multinomial采样 / 全部逐元素与归约 | 直接用 |
| ✅ 开关后 | SDPA 4D(causal)、polar、bincount | `allow_internal_format=False` + `jit_compile=False` |
| ❌ 内核缺失 | stft / istft / fft 家族 / 复数abs | 参考音频前处理整体 CPU 化（每请求一次，可忽略） |
| 🐛 内核缺陷 | conv_transpose1d(groups>64) | `npu_compat.py` 自动替换为等效实现（零精度损失） |

完整 70 算子实测矩阵与复现探针见 [docs/](docs/) 与 [probes/](probes/)。

## 🗺️ 后续优化方向

- GPT 自回归占 8s 大头：可试 torch_npu 图模式 / 迁移到 MindIE 部署
- DiT 25 步 ODE：蒸馏/少步采样
- BigVGAN 的 convT 等效实现可用 Ascend C 算子重写
- bf16：新栈已支持 bf16 matmul（实测通过），GPT/DiT 转 bf16 预计可再省一半显存带宽

## 📄 许可与致谢

- 本仓库（补丁、npu_compat、脚本、文档）：Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)
- 上游模型与代码：[index-tts](https://github.com/index-tts/index-tts)（Bilibili，bilibili Model Use License），本仓库仅以 diff 形式分发对其源码的修改
- 依赖的开源权重：[nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)、[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)、[amphion/MaskGCT](https://huggingface.co/amphion/MaskGCT)、[funasr/campplus](https://huggingface.co/funasr/campplus)

---

## English Summary

**IndexTTS-2.5 (Bilibili's zero-shot voice-cloning TTS) fully ported to a single Ascend 310P3 chip (Atlas 300I Duo)** — GPT backbone + flow-matching DiT + BigVGAN vocoder all run on NPU; reference-audio frontend (w2v-bert / MaskGCT / CAMPPlus + all FFT ops) on CPU. First publicly documented end-to-end run on 310P: clean 22.05 kHz speech (zero clipping), ASR-roundtrip verified, steady-state RTF ≈ 5.6. Includes a 70-operator on-device support matrix for 310P and the discovery + zero-loss workaround of an **op-plugin kernel bug where `conv_transpose1d(groups>64)` silently returns wrong results**. Only 3 minimal patches (~10 lines) to upstream; everything else is injected at runtime via `npu_compat.py`.
