"""微测试：conv1d fp32 在不同 shape 下 CPU vs NPU 精度（复现 resblocks.3.convs1.0 失效）。"""
import npu_compat  # noqa: F401
import torch
import torch.nn.functional as F

torch.manual_seed(0)

cases = [
    # (cin, length, kernel, dilation)  前两个是 OK 层的形状，第三个是爆发层
    (768, 576, 3, 1),
    (768, 576, 3, 5),
    (384, 2304, 3, 1),
    (384, 2304, 5, 1),
    (384, 2304, 5, 11),
    (512, 2304, 5, 1),
]
for cin, L, k, dil in cases:
    x = torch.randn(1, cin, L, dtype=torch.float32)
    w = torch.randn(cin, cin, k, dtype=torch.float32) * 0.02
    b = torch.randn(cin, dtype=torch.float32) * 0.02
    ref = F.conv1d(x, w, b, dilation=dil, padding=dil * (k - 1) // 2)
    out = F.conv1d(x.npu(), w.npu(), b.npu(), dilation=dil, padding=dil * (k - 1) // 2).cpu()
    d = (ref - out).abs().max().item()
    print(f"cin={cin:4d} L={L:5d} k={k} dil={dil:2d}: max|d|={d:.6f} "
          f"ref_std={ref.std():.4f} npu_std={out.std():.4f}")

# 复现爆点：真实 resblock.3 输入分布（snake 激活后的输出，std~3）
print()
x = torch.randn(1, 384, 2304, dtype=torch.float32) * 3
w = torch.randn(384, 384, 5, dtype=torch.float32) * 0.05
ref = F.conv1d(x, w, padding=2)
out = F.conv1d(x.npu(), w.npu(), padding=2).cpu()
print(f"真实分布复现: max|d|={(ref-out).abs().max().item():.4f} ref_std={ref.std():.4f} npu_std={out.std():.4f}")
