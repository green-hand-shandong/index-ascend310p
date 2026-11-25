"""npu_fft 正确性测试：所有结果对标 CPU 原生 torch 计算。

跑法（310P 容器内）:
    python tests/test_npu_fft.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

DEV = "npu:0" if torch.npu.is_available() else "cpu"
print(f"device = {DEV}")

import npu_fft


def ref_cpu(fn, x, *a, **k):
    """在 CPU 上用原生 torch 算参考值。"""
    return fn(x.cpu(), *a, **k)


def cmp(name, got, want, tol=2e-4, is_complex=True):
    # 310P 上复数张量直接 .cpu() 会触发 ERR01007（Complex dtype 传输不支持），
    # 一律先在 NPU 上 view_as_real 拆成实虚两个浮点张量再搬运
    if got.is_complex():
        got = torch.view_as_real(got.resolve_conj())
    if want.is_complex():
        want = torch.view_as_real(want.resolve_conj())
    got = got.cpu()
    want = want.cpu()
    d = (got - want).abs().max().item()
    scale = want.abs().max().item() + 1e-8
    rel = d / scale
    ok = rel < tol
    print(f"  [{'OK' if ok else 'FAIL'}] {name:<38} max|d|={d:.3e} rel={rel:.2e} (tol {tol})")
    return ok


all_ok = True

# ---------------- 1. fft / ifft (pow2) ----------------
print("== fft/ifft (radix-2, N=1024) ==")
x = torch.randn(2, 3, 1024)
x_npu = x.to(DEV)
want = ref_cpu(torch.fft.fft, x)
got = npu_fft.fft(x_npu)
all_ok &= cmp("fft pow2 (2,3,1024)", got, want)

want = ref_cpu(torch.fft.ifft, x)
got = npu_fft.ifft(x_npu)
all_ok &= cmp("ifft pow2", got, want)

# round-trip: ifft(fft(x)) == x
rt = npu_fft.ifft(npu_fft.fft(x_npu)).real
all_ok &= cmp("round-trip ifft(fft(x))", rt, x, tol=1e-4)

# ---------------- 2. fft (dft, 非2次幂) ----------------
print("== fft (DFT 矩阵, N=500/1000) ==")
x5 = torch.randn(4, 500)
got = npu_fft.fft(x5.to(DEV))
want = ref_cpu(torch.fft.fft, x5)
all_ok &= cmp("fft dft (4,500)", got, want, tol=1e-4)

x10 = torch.randn(1, 1000)
got = npu_fft.fft(x10.to(DEV))
want = ref_cpu(torch.fft.fft, x10)
all_ok &= cmp("fft dft (1,1000)", got, want, tol=1e-4)

# ---------------- 3. rfft / irfft ----------------
print("== rfft/irfft ==")
x = torch.randn(2, 16000)
got = npu_fft.rfft(x.to(DEV))
want = ref_cpu(torch.fft.rfft, x)
all_ok &= cmp("rfft (2,16000)", got, want)

spec = npu_fft.rfft(x.to(DEV))
got = npu_fft.irfft(spec, n=16000)
all_ok &= cmp("irfft round-trip", got, x, tol=1e-4)

# ---------------- 4. pow2 vs dft 一致性 ----------------
print("== pow2 与 dft 两种实现互检 ==")
x = torch.randn(2, 512)
a = npu_fft.fft(x.to(DEV), _force="pow2")
b = npu_fft.fft(x.to(DEV), _force="dft")
all_ok &= cmp("pow2 vs dft (N=512)", a, b, tol=1e-4)

# ---------------- 5. stft ----------------
print("== stft (mel 前端同款参数 n_fft=1024, hop=256, center+reflect) ==")
wav = torch.randn(1, 22050) * 0.1
kwargs = dict(n_fft=1024, hop_length=256, win_length=1024,
              center=True, pad_mode="reflect", normalized=False, onesided=True)
got = npu_fft.stft(wav.to(DEV), window=npu_fft.hann_window(1024, device=DEV), **kwargs)
want = ref_cpu(torch.stft, wav, window=torch.hann_window(1024), return_complex=True, **kwargs)
print(f"  shape got={tuple(got.shape)} want={tuple(want.shape)}")
all_ok &= cmp("stft (1,22050)", got, want, tol=5e-4)

# n_fft=512 (小帧走 dft 也测)
kwargs512 = dict(n_fft=512, hop_length=128, win_length=512,
                 center=True, pad_mode="reflect", normalized=False, onesided=True)
got = npu_fft.stft(wav.to(DEV), window=npu_fft.hann_window(512, device=DEV), **kwargs512)
want = ref_cpu(torch.stft, wav, window=torch.hann_window(512), return_complex=True, **kwargs512)
all_ok &= cmp("stft n_fft=512", got, want, tol=5e-4)

# ---------------- 6. istft ----------------
print("== istft (stft→istft round-trip) ==")
spec = torch.stft(wav, n_fft=1024, hop_length=256, window=torch.hann_window(1024),
                  center=True, pad_mode="reflect", return_complex=True)
# 310P: 复数张量 .to(npu) 传输会触发 ERR01007 → 实虚分离搬运后在 NPU 上重组
spec_pair = torch.view_as_real(spec).to(DEV)
spec_npu = torch.view_as_complex(spec_pair)
got = npu_fft.istft(spec_npu, n_fft=1024, hop_length=256,
                    window=npu_fft.hann_window(1024, device=DEV), center=True)
want = ref_cpu(torch.istft, spec, n_fft=1024, hop_length=256,
               window=torch.hann_window(1024), center=True)
print(f"  shape got={tuple(got.shape)} want={tuple(want.shape)}")
all_ok &= cmp("istft full-spectrum", got, want, tol=5e-4)

# round-trip: istft(stft(x)) == x（波形级恢复；谱级 round-trip 因 center+reflect
# 边界帧在 istft 中被裁掉，CPU 原生 torch 也同样不严格相等，不做谱对比）
wav_rt = npu_fft.istft(npu_fft.stft(wav.to(DEV), window=npu_fft.hann_window(1024, device=DEV), **kwargs),
                       n_fft=1024, hop_length=256,
                       window=npu_fft.hann_window(1024, device=DEV), center=True)
all_ok &= cmp("istft(stft(x)) 波形恢复", wav_rt, wav[..., : wav_rt.shape[-1]], tol=5e-4)

# ---------------- 7. 大批量/长序列（真实前端负载） ----------------
print("== 真实负载: 15s 参考音频 mel 前端参数 ==")
wav = torch.randn(2, 24000 * 15) * 0.1
kwargs_ref = dict(n_fft=1024, hop_length=256, win_length=1024,
                  center=True, pad_mode="reflect", normalized=False, onesided=True)
got = npu_fft.stft(wav.to(DEV), window=npu_fft.hann_window(1024, device=DEV), **kwargs_ref)
want = ref_cpu(torch.stft, wav, window=torch.hann_window(1024), return_complex=True, **kwargs_ref)
all_ok &= cmp("stft (2,360000) 15s×2", got, want, tol=5e-4)

# ---------------- 8. 性能对比 ----------------
print("== 性能: stft 15s×2 音频 (2,360000), n_fft=1024/hop=256 ==")
wav_npu = wav.to(DEV)
win_npu = npu_fft.hann_window(1024, device=DEV)
wav_cpu, win_cpu = wav, torch.hann_window(1024)
for name, fn in (("NPU 重组", lambda: npu_fft.stft(wav_npu, window=win_npu, **kwargs_ref)),
                 ("CPU 原生", lambda: torch.stft(wav_cpu, window=win_cpu, return_complex=True, **kwargs_ref))):
    for _ in range(2):
        fn()
    if DEV != "cpu":
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        r = fn()
    if DEV != "cpu":
        torch.npu.synchronize()
    print(f"  {name}: {(time.perf_counter()-t0)/5*1000:.1f} ms/次")

print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
