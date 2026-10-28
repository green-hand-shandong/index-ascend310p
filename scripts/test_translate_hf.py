"""Index-Translate (2B/9B/Homura/Nailong 等任意 Qwen3.5 底座模型) 在昇腾 310P 上的
transformers 直接推理测试。

用法: python test_translate_hf.py <模型目录>
说明: vLLM-Ascend 的 310P FA 内核 (flash_attention_prefill) 对该架构 head_dim=256
会 aicore timeout，故走 transformers + eager 注意力（见 docs/ascend310p-op-support.md）。
"""
import sys
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./Index-Translate-2B"

torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False   # SDPA ERR01007 规避
torch.npu.set_compile_mode(jit_compile=False)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="eager").npu().eval()
print(f">> 模型加载耗时: {time.time()-t0:.1f}s")

prompts = [
    "将下面的文本翻译成英文：\n昇腾310处理器是华为面向边缘推理的AI加速卡。",
    "Translate the following text into Chinese:\nThe weather is nice today, let's go for a walk in the park.",
    "将下面的文本翻译成日文：\n今天我们要测试国产芯片上的大模型翻译能力。",
]
for p in prompts:
    enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True,
                                  enable_thinking=False).to("npu:0")
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(enc.input_ids, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    n_new, dt = out.shape[1] - enc.input_ids.shape[1], time.time() - t1
    print(f">> [{n_new} tok / {dt:.1f}s = {n_new/dt:.1f} tok/s] {p[:40]}")
    print("   ->", tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()[:200])
print("HF_TRANSLATE_TEST_OK")
