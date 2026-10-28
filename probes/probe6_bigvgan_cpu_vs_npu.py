"""BigVGAN CPU vs NPU 同输入对比：定位饱和误差来源。"""
import npu_compat  # noqa: F401
import sys

sys.path.insert(0, "/data2/gxw/indextts-310p/index-tts")
import torch
npu_compat.install_indextts_patches()

from indextts.s2mel.modules.bigvgan import bigvgan

CKPT = "/data2/gxw/indextts-310p/checkpoints/hf_cache/bigvgan"
# 固定输入：模拟真实 log-mel 分布
torch.manual_seed(42)
mel = (torch.randn(1, 80, 144, dtype=torch.float32) * 2.7 - 6.0).clamp(-11.6, 0.26)
mel_cpu = mel.clone()
mel_npu = mel.clone().to("npu:0")

# ---- CPU 前向 ----
bg_cpu = bigvgan.BigVGAN.from_pretrained(CKPT, use_cuda_kernel=False).cpu()
bg_cpu.remove_weight_norm()
bg_cpu.eval()
with torch.no_grad():
    out_cpu = bg_cpu(mel_cpu)
print(f"CPU  out: min={out_cpu.min():.4f} max={out_cpu.max():.4f} std={out_cpu.std():.4f} "
      f"rms={out_cpu.pow(2).mean().sqrt():.4f}")
del bg_cpu

# ---- NPU 前向 ----
bg_npu = bigvgan.BigVGAN.from_pretrained(CKPT, use_cuda_kernel=False).to("npu:0")
bg_npu.remove_weight_norm()
bg_npu.eval()
with torch.no_grad():
    out_npu = bg_npu(mel_npu).cpu()
print(f"NPU  out: min={out_npu.min():.4f} max={out_npu.max():.4f} std={out_npu.std():.4f} "
      f"rms={out_npu.pow(2).mean().sqrt():.4f}")
print(f"差异: max|d|={(out_cpu - out_npu).abs().max():.6f} "
      f"mean|d|={(out_cpu - out_npu).abs().mean():.6f}")
