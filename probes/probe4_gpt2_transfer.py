import torch
import torch_npu
import torch.nn.functional as F
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

# 1. gelu tanh 近似（GPT2 用）
try:
    F.gelu(torch.randn(8,8,dtype=torch.float16,device='npu:0'), approximate='tanh').sum()
    print("[OK] gelu(tanh)")
except Exception as e: print("[FAIL] gelu(tanh):", str(e)[:80])

# 2. transfer_to_npu 官方CUDA兼容垫片
try:
    from torch_npu.contrib import transfer_to_npu
    print("[OK] transfer_to_npu 导入成功")
    x = torch.randn(4,4).cuda()  # 现在应映射到npu
    print("[OK] .cuda() -> ", x.device)
except Exception as e: print("[FAIL] transfer_to_npu:", str(e)[:100])

# 3. HF GPT2 骨干实测（tiny 随机权重，eager attention，fp16）
try:
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, vocab_size=50257)
    m = GPT2LMHeadModel(cfg).half().npu()
    m.config._attn_implementation = "eager"
    m.eval()
    ids = torch.randint(0, 50000, (1, 16), device='npu:0')
    with torch.no_grad():
        out = m(ids)
        gen = m.generate(ids, max_new_tokens=20, do_sample=True, top_k=50)
    print("[OK] GPT2 forward+generate on NPU, gen shape:", tuple(gen.shape))
except Exception as e: print("[FAIL] GPT2:", str(e)[:150])
