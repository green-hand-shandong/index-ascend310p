# IndexTeam LLM 系列在 310P 上的适配结论

> 环境：Atlas 300I Duo / 310P3，CANN 9.2，torch_npu 2.10.0.post7，vllm-ascend 0.30.0rc2。
> 测试脚本见 `scripts/test_translate_hf.py`、`scripts/test_19b_chat.py`。

## 实测结果汇总

| 模型 | 底座 | 路线 | 实测结果 |
|---|---|---|---|
| **Index-Translate-2B** | Qwen3.5 混合架构 (linear attn + full attn, head_dim=256) | transformers 5.21 + eager + fp16 | ✅ 三向翻译全部正确，6~8 tok/s，加载 5.9s |
| **Index-Translate-9B** | 同上 | 同上 + `enable_thinking=False` | ✅ 三向翻译直接输出，2.1~2.3 tok/s，加载 12.5s |
| **Index-Homura-2B** | 同上（音节控制微调） | 同 2B 配方 | ✅ 三向翻译正确，5.8~7.3 tok/s |
| **Index-1.9B-Chat** | 自研 LLaMA 类 (trust_remote_code) | transformers 4.52.1 隔离栈 + eager + fp16 | ✅ 对话/问答/翻译正常，8~11 tok/s，加载 10.8s |
| Index-Nailong-2B/9B | 同 Qwen3.5 底座 | 同 2B 配方 | 预期可用；229K 长上下文受掩码物化限制（≤32K 实用） |
| Index-Translate-35B-A3B | Qwen3.5 MoE | 同配方，fp16 需 ~70GB → TP2 双芯片 | 未实测（磁盘受限），架构路径与 2B 一致 |
| IndexTTS-2.5 | 见 README 主文档 | torch_npu 全栈 | ✅ RTF 5.6 |

## 通用配方（Qwen3.5 底座）

```python
import torch, torch_npu
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False   # SDPA ERR01007 规避
torch.npu.set_compile_mode(jit_compile=False)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16,
    attn_implementation="eager").npu().eval()
```

要点：
1. **必须 eager 注意力**（SDPA 4D mask 在 310P 报 ERR01007）。
2. **fp16 而非 bf16**（权重原生 bf16，加载时用 dtype=float16 转换；新版栈虽支持 bf16 matmul，
   但混合层未必全部支持，fp16 实测更稳）。
3. 单芯片 (44GB) 可容纳：2B (4.5GB)、9B (18GB)、35B-A3B (fp16 需 TP2)；Nailong 的 229K
   长上下文受掩码物化限制，实际可用上下文建议 ≤32K。

## ⚠️ vLLM-Ascend 在 310P 上跑 Qwen3.5 的现状

vllm-ascend 0.30.0rc2 已注册 `Qwen3_5ForConditionalGeneration`，权重加载与多模态 warmup
均正常，但 **LM memory profiling 阶段 `flash_attention_prefill_24` 内核 aicore timeout**
（`EZ9999 retCode=0x25`，`max_model_len=2048/4096` 均复现）。该内核经由
`torch_npu._npu_flash_attention` 调用，怀疑与 head_dim=256 的形状组合在 310P3 上存在
内核缺陷（vllm-ascend RFC #7394 验证 Qwen3.5 用的是 CANN 26.0 POC 镜像）。

绕行方案即上文的 transformers eager 路线；若必须 vLLM 服务化，可尝试：
- CANN 26.x POC 镜像 + msmodelslim W8A8 量化（RFC #7394 的验证路径）；
- 或 patch `_310p/attention/attention_v1.py` 把 prefill 换成掩码物化的 torch eager 实现。

## Index-1.9B 的版本坑

上游 `modeling_index.py`（trust_remote_code）按 transformers 4.3x 时代 API 编写：
- transformers ≥5.x 会把 `rope_scaling` 规范化为新结构，老代码 `rope_scaling["type"]` 直接 KeyError；
- 4.52.1 + `huggingface_hub<1.0` + `tokenizers==0.21` 组合实测可用（本仓库
  `scripts/docker_exec.sh` 的 `/opt/tf452` 隔离栈即此用途）。
