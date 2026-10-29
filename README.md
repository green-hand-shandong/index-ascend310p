# Index-Ascend310P: Bilibili IndexTeam Models on Huawei Atlas 300I Duo (Ascend 310P3)

**All five flagship IndexTeam models fully ported and verified on a single Huawei Ascend 310P3 chip (Atlas 300I Duo, 44GB) — including localization & fixes for 2 vendor kernel bugs and an on-device 70-operator support matrix.**

> **English** | [简体中文](README_zh.md)

---

## 🎯 TL;DR

On one **Ascend 310P3** chip we run end-to-end: **IndexTTS-2.5** (zero-shot voice-cloning TTS), **Index-Translate-2B/9B** (150-language translation), **Index-Homura-2B** (syllable-controlled translation for dubbing) and **Index-1.9B-Chat** (conversational LLM) — speech with zero clipping verified by ASR round-trip (RTF 5.6), translation/dialogue outputs correct at 2~11 tok/s.

## 🖥️ Hardware & Software

| Item | Config |
|---|---|
| Hardware | Atlas 300I Duo server, **Ascend 310P3** (soc=202), single chip (44GB usable) |
| Driver | npu-smi 25.5.1 |
| Container image | `quay.io/ascend/vllm-ascend:v0.30.0rc2-310p` (CANN 9.2.0, Python 3.12.13) |
| Frameworks | torch 2.10.0 + torch_npu 2.10.0.post7; vLLM 0.30.0 (vllm-ascend); transformers dual stack 5.21 / 4.52.1 isolated |
| Host | aarch64 Linux, Docker 29 |

## 📊 Results Matrix (model × status)

| Model | Task | Route | Measured speed | Status |
|---|---|---|---|---|
| **IndexTTS-2.5** | Zero-shot voice-cloning TTS (zh/en/ja/es/ar) | torch_npu full stack (hot path NPU + frontend CPU) | RTF 5.6 (6.14s audio in 34.6s) | ✅ Works, zero clipping, ASR round-trip passed |
| **Index-Translate-2B** | 150-language text/image translation | transformers 5.21 + eager + fp16 | 6~8 tok/s, 5.9s load | ✅ All 3 translation directions correct |
| **Index-Translate-9B** | Same (thinking-mode) | Same + `enable_thinking=False` | 2.1~2.3 tok/s, 12.5s load | ✅ Direct translations |
| **Index-Homura-2B** | Syllable-controlled translation (dubbing) | Identical to Translate-2B recipe | 5.8~7.3 tok/s | ✅ **Zero-change reuse** |
| **Index-1.9B-Chat** | Lightweight chat LLM (custom arch) | transformers 4.52 isolated stack + trust_remote_code | 8~11 tok/s, 10.8s load | ✅ Dialog/QA/translation OK |
| Index-Translate-35B-A3B | MoE translation | Same Qwen3.5 recipe; fp16 ~70GB → TP2 dual chip | — | ⏸ Not run (disk-limited); path identical |
| Index-Nailong-2B/9B | Long-doc translation (229K ctx) | Same Qwen3.5 recipe | — | ⏸ Expected OK; long context capped by mask materialization (≤32K practical) |
| Index-Echo S2TT/S2ST | Speech translation | Needs Qwen3-Omni audio tower + CosyVoice3 | — | ❌ Not attempted; recommend 910-class HW |
| Index-anisora | Anime video generation | CogVideoX/Wan2.1 diffusion | — | ❌ Not feasible on 310P |
| IndexTTS 1.x / 2 | Legacy TTS | — | — | ⏸ Superseded by 2.5 |

## 🔊 Per-Model Details

### IndexTTS-2.5 (speech synthesis)

- **Result**: 22.05 kHz speech, steady-state **RTF 5.6** (first run 11.4 incl. warmup); peak 0.49~0.54, RMS 0.075, **zero clipping**, no NaN.
- **Quality gate**: paraformer ASR round-trip on the synthesized audio — the sentence "欢迎来到昇腾三一零P语音合成测试，这段语音完全在国产算力上生成" was 100% recovered on the second run (first run: only "昇腾" homophone-recognized by ASR, an ASR-side artifact).
- **Breakdown**: GPT autoregression 8.0s + s2mel DiT (25 steps) 4.8s + BigVGAN vocoder 1.6s; model load 53~64s.
- **Key work**:
  1. 70-operator on-device probe → FFT family (stft/istft/fft/complex) kernel-level unsupported → entire reference-audio frontend pinned to CPU (once per request, negligible);
  2. Generation hot path (GPT/DiT/BigVGAN) fully on NPU: `transfer_to_npu` shim + three switches;
  3. **Fixed a 310P op-plugin kernel bug: `conv_transpose1d(groups>64)` silently returns wrong results** — BigVGAN anti-aliased upsampling (C=384/768) hit it, saturating 87% of samples. Zero-loss equivalent implementation (max|d|=0 vs CPU, and faster than the native kernel);
  4. Only 3 upstream files ~10 lines patched (wetext normalization, eager attention, frontend CPU pinning); everything else injected at runtime by `npu_compat.py`;
  5. All auxiliary models (w2v-bert-2.0 / BigVGAN / MaskGCT / CAMPPlus) pre-downloaded for offline inference.

### Index-Translate-2B / 9B / Index-Homura-2B (Qwen3.5-based trio)

- **Results** (three fixed sentences, zh→en / en→zh / zh→ja, all correct):

| Model | zh→en sample | Speed |
|---|---|---|
| Translate-2B | "The Ascend 310 processor is Huawei's AI acceleration card for edge inference." | 6~8 tok/s |
| Translate-9B | "The Ascend 310 processor is an AI acceleration card from Huawei designed for edge inference." | 2.1~2.3 tok/s |
| Homura-2B | Same quality as 2B | 5.8~7.3 tok/s |

- **Notes**:
  1. vLLM route currently blocked: the `flash_attention_prefill_24` kernel **aicore-timeouts** for head_dim=256 shapes (EZ9999; weights load and multimodal warmup both pass; reproduces at 2048/4096 ctx) — see [docs/llm-models-310p.md](docs/llm-models-310p.md) and the "unfinished attempts" section below;
  2. The transformers eager recipe is just three switches: `allow_internal_format=False` + `jit_compile=False` + `attn_implementation="eager"`, loaded in fp16;
  3. **9B requires `enable_thinking=False`**, otherwise it produces a long reasoning preamble before translating;
  4. **Homura-2B reuses the Translate recipe with zero changes** — same-base fine-tunes need no extra work.

### Index-1.9B-Chat (custom architecture)

- **Result**: self-intro / knowledge QA / translation all normal, 8~11 tok/s.
  - "你好，请用一句话介绍你自己" → "我是一个由Bilibili研发的大语言模型……"
  - "把这句话翻译成英文：今天天气真不错" → "The weather is really nice today."
- **Notes**: zero model-code changes (trust_remote_code), but the upstream custom modeling was written against the 2023-era transformers API — transformers ≥5.x normalizes `rope_scaling` into a new structure and the old code raises KeyError. Solution: isolated stack `/opt/tf452` (transformers 4.52.1 + huggingface_hub 0.36.2 + tokenizers 0.21 via `pip --target`), coexisting with the system 5.21 stack.

## 🐛 310P Kernel Bugs Found & Fixed (reusable beyond this repo)

| Bug | Symptom | Impact | Fix |
|---|---|---|---|
| **`conv_transpose1d(groups=C>64)` wrong results** (op-plugin) | C=8/64 fine, C=384/768 silently wrong, no warning | BigVGAN output 87% saturated noise | matmul-equivalent impl, max\|d\|=0, faster than native → [docs/conv-transpose1d-groups-bug.md](docs/conv-transpose1d-groups-bug.md) |
| **`flash_attention_prefill` head_dim=256 aicore timeout** (vllm-ascend 310P path) | EZ9999 retCode=0x25, stable repro | vLLM serving of Qwen3.5 blocked | Workaround: transformers eager route → [docs/llm-models-310p.md](docs/llm-models-310p.md) |
| FFT/STFT/complex ops missing | aclnn kernel errors directly | Models with spectral paths | CPU-ize (small tensors, low frequency) |
| SDPA 4D+mask ERR01007 | Format-cast error | All sdpa attention | `allow_internal_format=False` (causal OK) or eager |

The three switches every 310P torch_npu program needs:

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu     # maps .cuda() code to npu, zero changes
torch.npu.config.allow_internal_format = False    # fixes SDPA ERR01007
torch.npu.set_compile_mode(jit_compile=False)     # fixes polar/bincount/complex dispatch
```

## 🤔 Did anything run with zero changes?

Strictly **no** — 310P hardware quirks require at least environment-level config (three switches + eager attention + fp16). But the **marginal cost can be zero**: once one Qwen3.5-base model works (Translate-2B), sibling fine-tunes (Homura-2B, Nailong) run by only changing the model path; Translate-9B adds a single `enable_thinking=False` flag.

## 🧪 Test Methodology & Limitations (honest)

- **IndexTTS-2.5 got the deepest testing**: objective metrics (peak/clipping/NaN/ZCR) + **ASR round-trip for intelligibility** + double-run timing.
- **LLMs are smoke-level**: three fixed sentences manually checked + tok/s timing; **no** FloRES/COMET, no MMLU/C-Eval, no long-text or concurrency testing. Speeds are single-request greedy decode, no batching.
- Conclusions for unadapted models (Echo/anisora/35B-A3B/Nailong) are architecture-based analysis, not measurements — flagged in the matrix.

## 🚧 Unfinished Attempts & Lessons Learned

> Failures and non-verified items are part of the deliverable: for each, we record the blocker, what was ruled out, and reusable experience — so nobody steps on the same rake twice.

### 1. vLLM-Ascend serving of Qwen3.5 (not cleared, worked around)

- **What works**: vLLM 0.30.0rc2 registers `Qwen3_5ForConditionalGeneration`; weights load fine (2B 6.1GB / 9B 18GB fp16); multimodal warmup passes (42s); the 310P attention backend selects normally (CUSTOM backend + LBNHC KV layout).
- **Blocker**: during LM memory profiling the `flash_attention_prefill_24` kernel **aicore-timeouts** (EZ9999 retCode=0x25); reproduces at max_model_len 2048/4096 and on idle chips.
- **What we tried**: `ASCEND_LAUNCH_BLOCKING=1` to localize the faulting kernel; ruled out chip occupancy and transient faults; attempted to probe `torch_npu._npu_flash_attention` directly — failed due to its strict calling conventions.
- **Lessons**:
  1. Low-level PTA private APIs have strict layout/mask conventions — **the reliable way to localize kernel faults is the upper engine + LAUNCH_BLOCKING**, not hand-rolled kernel calls;
  2. The kernel is suspected to be shape-related (head_dim=256) — vllm-ascend RFC #7394 validated Qwen3.5 on 310P with the **CANN 26.0 POC image**, we used CANN 9.2. **Try the POC image first** when reproducing;
  3. The RFC's W8A8 + msmodelslim quantization path is untried (needs the quant toolchain);
  4. For offline use the transformers eager recipe is fully sufficient (this repo); for serving, wait for the kernel fix or use the POC image.

### 2. Index-Translate-35B-A3B (MoE, not run)

- **Blocker**: fp16 weights ~70GB > 44GB single chip; disk had <10GB left during adaptation so weights weren't downloaded.
- **Lessons**: architecture (`Qwen3_5MoeForConditionalGeneration`) is same-family as 2B/9B and registered everywhere; the transformers eager recipe is expected to work as-is; community consensus is that MoE (3B active) is actually the *best-performing* tier on 310P. Feasible paths: **TP2 dual-chip fp16**, or W8A8 then single chip.

### 3. Index-Nailong (229K long-doc translation, not run)

- **Blocker**: 310P attention has no compressed-mask path; materializing the `[L, L]` fp16 mask at 229K context needs 100GB+ — physically impossible.
- **Lessons**: the model itself matches the Translate recipe (expected to run), but its "whole-document in one shot" selling point dies on 310P; practical cap ~32K. The vllm-ascend 310P backend has a `support_compressed_mask` switch (`_310p/attention/attention_mask.py`) — **if verified working, this limit could be lifted significantly — unverified**.

### 4. Index-Echo S2TT/S2ST (speech translation, not started)

- **Blocker**: three-component joint adaptation (Qwen3-Omni AuT audio tower + Index-Translate decoder + CosyVoice3 generation) — ~3× the work of a single model; Omni multimodality adds more uncertainty on the 310P attention path.
- **Lessons**: if transformers 5.x natively supports the AuT audio tower, the eager recipe should carry over; the omnirt runtime claims an ascend backend for IndexTTS, worth investigating; **recommend 910-class hardware**.

### 5. Index-anisora (anime video generation, ruled out)

- **Blocker**: 14B-class video diffusion (CogVideoX-5B / Wan2.1-14B base) — compute and memory both exceed the 310P product tier.
- **Lessons**: only small diffusion models have 310P success cases; don't attempt video generation here.

### 6. Engineering misc (network / deps)

- Gitee mirrors' **LFS is a paid feature** — repo sample wavs couldn't be pulled → built reference audio from a public hf-mirror dataset (librispeech dummy);
- raw.githubusercontent.com / jsDelivr unreachable from this host; **`gh-proxy.com` prefix proxy works** (used to fetch latest upstream);
- ModelScope single-file API: `/api/v1/models/{org}/{repo}/repo?FilePath=x&Revision=master`, resumable via `curl -C -`; **watch out for sharded weights** (9B = 4 shards + `model.safetensors.index.json`; missing index fails with "no file named model.safetensors");
- pypi.org unreachable; Tsinghua/Aliyun mirrors work; hf-mirror.com works (set `HF_ENDPOINT`).

## 🚀 Quick Start

Requirements: Atlas 300I Duo (or 200I Pro), Docker, ~15GB disk.

```bash
# 1. Deploy (clone upstream index-tts + patch + download all models; CN mirrors auto-selected)
git clone https://github.com/green-hand-shandong/index-ascend310p.git
cd index-ascend310p
bash scripts/setup_all.sh workspace

# 2. Start the NPU container
docker run -d --name indextts-310p \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $PWD:/workspace -w /workspace \
  quay.io/ascend/vllm-ascend:v0.30.0rc2-310p sleep infinity

# 3. IndexTTS-2.5 deps + speech synthesis (REF_WAV = 3~15s clean speech)
bash scripts/docker_exec.sh indextts-310p \
  'pip install transformers==4.52.1 tokenizers==0.21.0 ... -i https://pypi.tuna.tsinghua.edu.cn/simple'   # full list at the end of scripts/setup_all.sh
bash scripts/docker_exec.sh indextts-310p \
  'REF_WAV=/workspace/your_voice.wav python tests/run_test.py out.wav "你好，昇腾！"'

# 4. LLM family (Qwen3.5-based, system stack)
bash scripts/docker_exec_vllm.sh indextts-310p /workspace \
  'python scripts/test_translate_hf.py /workspace/Index-Translate-2B'
```

Weights are pulled from ModelScope / hf-mirror: `IndexTeam/IndexTTS-2.5` (~6GB), `IndexTeam/Index-Translate-2B` (4.6GB), `IndexTeam/Index-Translate-9B` (19GB, 4 shards), `IndexTeam/Index-Homura-2B` (4.6GB), `IndexTeam/Index-1.9B-Chat` (4.4GB).

## 📁 Repository Layout

```
├── npu_compat.py        # NPU compat layer (three switches + FFT CPU-ization + convT fix, import & use)
├── patches/             # minimal upstream index-tts patches (3 files, ~10 lines)
├── scripts/             # deploy scripts + per-model test scripts (TTS / Translate / Homura / 1.9B)
├── tests/               # e2e tests & audio validation (incl. ASR round-trip)
├── probes/              # 7 on-device operator probes (full kernel-bug reproduction chain)
└── docs/                # 310P operator support report / convT bug analysis / LLM adaptation notes
```

## 🗺️ Roadmap

- vLLM serving: retry on CANN 26 POC image + W8A8 (RFC #7394 path) or wait for the FA kernel fix
- IndexTTS perf: graph mode / DiT distillation / Ascend C rewrite of the convT workaround
- 35B-A3B dual-chip TP2 run; Nailong ≤32K long-doc run
- Proper benchmarks (FloRES/COMET, TTS MOS)

## 📄 License & Acknowledgements

- This repo (patches, compat layer, scripts, probes, docs): Apache-2.0 (see [LICENSE](LICENSE) / [NOTICE](NOTICE))
- Upstream models & code: [index-tts](https://github.com/index-tts/index-tts) (Bilibili Model Use License); this repo distributes modifications as diffs only
- Dependent weights: [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x), [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), [amphion/MaskGCT](https://huggingface.co/amphion/MaskGCT), [funasr/campplus](https://huggingface.co/funasr/campplus)

> Note: the GitCode mirror's URL path remains `indextts-2.5-ascend310p` due to platform limits (display name updated); GitHub is the canonical home with automatic redirects from old links.

---

## 简体中文摘要

**B站 IndexTeam 的 5 个旗舰模型已全部在一颗华为昇腾 310P3（Atlas 300I Duo，44GB）上端到端跑通**：IndexTTS-2.5 语音合成（RTF 5.6、零削波、ASR 回读验证）、Index-Translate-2B/9B 翻译（6~8 / 2.1~2.3 tok/s，三个语向全部正确）、Index-Homura-2B（5.8~7.3 tok/s，零改动复用）、Index-1.9B-Chat（8~11 tok/s，隔离栈方案）。含 70 算子真机支持度实测、2 个厂商内核缺陷的定位与修复（`conv_transpose1d(groups>64)` 静默错误、`flash_attention_prefill` head_dim=256 超时）、以及未完成项的完整经验记录。**完整中文文档见 [README_zh.md](README_zh.md)**。
