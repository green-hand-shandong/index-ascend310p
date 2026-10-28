# 昇腾 310P（Ascend 310P3 / Atlas 300I Duo）PyTorch 算子支持度实测报告

> 环境：Atlas 300I Duo / 310P3（soc=202），CANN 9.2，torch 2.10.0+cpu，torch_npu 2.10.0.post7，
> 镜像 `quay.io/ascend/vllm-ascend:v0.30.0rc2-310p`。
> 方法：真机逐算子执行探针（见 `probes/`），`probes/probe1_basic_ops.py` 覆盖 IndexTTS-2.5
> 推理路径全部约 90 个算子。

## 结论速览

| 类别 | 数量 | 说明 |
|---|---|---|
| ✅ 原生支持 | 55/70 | 全部核心 transformer/卷积/采样/逐元素算子 |
| ✅ 开关后支持 | 5 | SDPA 4D、polar、bincount 等 |
| ❌ 内核级不支持 | 8 | FFT 家族、kaiser_window（有 CPU 等效方案） |
| 🐛 内核缺陷 | 1 | `conv_transpose1d(groups=C>64)` 输出错误（本仓库重点，见另一文档） |

## 两个关键开关（310P 适配必备）

```python
import torch_npu
torch.npu.config.allow_internal_format = False   # 修复 SDPA "ERR01007 can not cast format"
torch.npu.set_compile_mode(jit_compile=False)    # 修复 polar/bincount/复数算子调度失败
```

开启后：**4D causal SDPA 实测可用**（此前报 ERR01007）、polar/bincount(int32) 通过。

## ✅ 原生通过（部分列表）

layer_norm、gelu（含 tanh 近似）、silu、softmax/log_softmax、cross_entropy、mse_loss、
smooth_l1_loss、embedding、linear、**conv1d**、**conv_transpose1d（groups≤64）**、dropout、
one_hot、interpolate(linear)、pad(constant/reflect)、fold/unfold、matmul/bmm/einsum、
norm、rsqrt、cumsum、logsumexp、argmax、topk、sort、**multinomial**（GPT 采样）、
gather/scatter_add/index_select、tanh/sigmoid/pow/exp/log/sin/cos/flip/tril/outer/sign/
atan2/round/abs/where/clamp、hann_window、linspace、randn/randint/randperm、
view_as_complex/view_as_real、**bf16 matmul**（新栈已支持，旧资料称 310P 无 bf16 已过时）、
autocast(npu, fp16)、eager 手写注意力（matmul+softmax+masked_fill）。

另：HF GPT2（eager, fp16）在 NPU 上 forward+generate 实测通过（`probes/probe4_gpt2_transfer.py`）。

## ❌ 不支持与绕行方案

| 算子 | 现象 | 绕行方案 |
|---|---|---|
| `torch.stft` / `torch.istft` | aclnn 内核 `NPU function error`（jit_compile=False 后仍失败） | CPU 计算（仅参考音频前处理用，每请求一次，开销可忽略） |
| `torch.fft.fft/rfft/irfft/ifft` | 同上（`FFTc2cKernelNpuOpApi` 报错） | CPU 计算 |
| 复数 `abs/conj` | `StructKernelNpuOpApi` 报错 | 复数张量全程留 CPU |
| `torch.kaiser_window` | PyTorch DispatchStub INTERNAL ASSERT | CPU 创建后 `.to(npu)`（BigVGAN 滤波器本就在 CPU 构建后再搬移，无需改动） |
| SDPA 4D + attn_mask | ERR01007 | eager 注意力或仅用 is_causal |

## 复现

```bash
docker run --rm <挂载见 scripts/docker_exec.sh 注释> \
  quay.io/ascend/vllm-ascend:v0.30.0rc2-310p python probes/probe1_basic_ops.py
```

探针清单：
- `probe1_basic_ops.py` — 70 算子基线扫描
- `probe2_sdpa_fft_workarounds.py` — SDPA/FFT 绕行方案验证
- `probe3_jit_compile.py` — jit_compile=False 开关效果
- `probe4_gpt2_transfer.py` — transfer_to_npu 垫片 + GPT2 真模型 forward/generate
- `probe5_conv_accuracy.py` 起 — conv 精度与 convT 缺陷定位（见 convt 缺陷文档）
