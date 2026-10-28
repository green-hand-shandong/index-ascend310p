"""Index-1.9B-Chat (自研架构, trust_remote_code) 在昇腾 310P 上的推理测试。

需在 transformers 4.5x 隔离栈下运行（新版 transformers 5.x 的 rope_scaling
字典结构变化会使上游自定义 modeling 代码 KeyError，见 README「多模型适配」）:
    PYTHONPATH=/opt/tf452:$ASCEND_PYTHONPATH python test_19b_chat.py <模型目录>
"""
import sys
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./Index-1.9B-Chat"

torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.float16,
    attn_implementation="eager").npu().eval()

prompts = ["你好，请用一句话介绍你自己。", "把这句话翻译成英文：今天天气真不错。"]
for p in prompts:
    enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("npu:0")
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(enc.input_ids, max_new_tokens=150, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    n_new, dt = out.shape[1] - enc.input_ids.shape[1], time.time() - t1
    print(f">> [{n_new} tok / {dt:.1f}s = {n_new/dt:.1f} tok/s] {p}")
    print("   ->", tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()[:200])
print("19B_TEST_OK")
