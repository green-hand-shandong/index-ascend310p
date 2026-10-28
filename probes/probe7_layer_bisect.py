import npu_compat  # noqa: F401
import sys
sys.path.insert(0, "/data2/gxw/indextts-310p/index-tts")
import torch
npu_compat.install_indextts_patches()
from indextts.s2mel.modules.bigvgan import bigvgan

CKPT = "/data2/gxw/indextts-310p/checkpoints/hf_cache/bigvgan"
torch.manual_seed(42)
mel = (torch.randn(1, 80, 144, dtype=torch.float32) * 2.7 - 6.0).clamp(-11.6, 0.26)

def run(device):
    m = bigvgan.BigVGAN.from_pretrained(CKPT, use_cuda_kernel=False)
    m.remove_weight_norm(); m.eval(); m = m.to(device)
    order, acts = [], {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o) and o.ndim >= 2:
                order.append(name); acts[name] = o.detach().float().cpu()
        return hook
    hs = [mod.register_forward_hook(mk(n)) for n, mod in m.named_modules()
          if isinstance(mod, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Upsample))]
    with torch.no_grad(): m(mel.clone().to(device))
    for h in hs: h.remove()
    del m
    return order, acts

order_c, a_cpu = run("cpu")
order_n, a_npu = run("npu:0")

# 按前向执行顺序找第一个 max|d|>0.05 的层
print("== 按执行顺序的前 12 个分歧层 ==")
shown = 0
for name in order_c:
    if name not in a_npu: continue
    a, b = a_cpu[name], a_npu[name]
    if a.shape != b.shape: continue
    d = (a-b).abs().max().item()
    if d > 0.05 and shown < 12:
        print(f"{name:42s} {str(tuple(a.shape)):22s} max|d|={d:.4f} cpu_std={a.std():.4f} npu_std={b.std():.4f}")
        shown += 1
