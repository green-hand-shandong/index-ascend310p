"""310P 算子支持度探针：覆盖 IndexTTS 全部推理路径算子。
原则：NPU 上直接执行，异常即 UNSUPPORTED；用 fallback 检测 API 区分原生 NPU 与 CPU 回退。
"""
import os, traceback
import torch
import torch_npu
import torch.nn.functional as F

print("soc:", torch.npu.get_soc_version(), "torch:", torch.__version__)
HAS_FB = True
try:
    import torch_npu
    print("torch_npu:", torch_npu.__version__)
    print("fallback api:", [a for a in dir(torch_npu) if "fallback" in a.lower()])
except Exception:
    pass

dev = "npu:0"
torch.npu.set_device(0)
D = torch.device(dev)

results = []

def probe(name, fn):
    try:
        fn()
        results.append((name, "OK", ""))
    except Exception as e:
        msg = str(e).replace("\n", " ")[:120]
        results.append((name, "FAIL", msg))

def t(*shape, dtype=torch.float16):
    return torch.randn(*shape, dtype=dtype, device=D)

# ---------- 注意力 ----------
def sdpa_causal():
    q, k, v = t(1, 8, 64, 64), t(1, 8, 128, 64), t(1, 8, 128, 64)
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
probe("sdpa(fp16,is_causal)", sdpa_causal)

def sdpa_mask():
    q, k, v = t(1, 8, 32, 64), t(1, 8, 32, 64), t(1, 8, 32, 64)
    mask = torch.ones(1, 8, 32, 32, dtype=torch.bool, device=D).tril()
    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
probe("sdpa(fp16,mask)", sdpa_mask)

def sdpa_fp32():
    q, k, v = t(1, 4, 16, 32, dtype=torch.float32), t(1, 4, 16, 32, dtype=torch.float32), t(1, 4, 16, 32, dtype=torch.float32)
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
probe("sdpa(fp32)", sdpa_fp32)

# ---------- NN 层 ----------
probe("layer_norm", lambda: F.layer_norm(t(2, 512), (512,)))
probe("gelu", lambda: F.gelu(t(2, 512)))
probe("silu", lambda: F.silu(t(2, 512)))
probe("softmax", lambda: F.softmax(t(2, 512), dim=-1))
probe("log_softmax", lambda: F.log_softmax(t(2, 512), dim=-1))
probe("cross_entropy", lambda: F.cross_entropy(t(4, 100), torch.randint(0, 100, (4,), device=D)))
probe("mse_loss", lambda: F.mse_loss(t(2, 8), t(2, 8)))
probe("smooth_l1_loss", lambda: F.smooth_l1_loss(t(2, 8), t(2, 8)))
probe("embedding", lambda: F.embedding(torch.randint(0, 100, (4,), device=D), t(100, 64)))
probe("linear", lambda: F.linear(t(2, 128), t(64, 128), t(64)))
def conv1d_():
    F.conv1d(t(1, 64, 100), t(128, 64, 5), padding=2)
probe("conv1d", conv1d_)
def convT1d_():
    F.conv_transpose1d(t(1, 128, 50), t(128, 64, 5), stride=4)
probe("conv_transpose1d", convT1d_)
def dropout_():
    F.dropout(t(2, 512), p=0.1, training=True)
probe("dropout", dropout_)
probe("one_hot", lambda: F.one_hot(torch.randint(0, 10, (4,), device=D), 10).float())
def interp_():
    F.interpolate(t(1, 64, 50), scale_factor=4, mode="linear")
probe("interpolate(linear)", interp_)
probe("pad(constant)", lambda: F.pad(t(1, 8, 10), (3, 3)))
probe("pad(reflect)", lambda: F.pad(t(1, 8, 10), (3, 3), mode="reflect"))
def fold_():
    F.fold(t(1, 12, 20), (4, 5), (3, 3))
probe("fold", fold_)
probe("unfold", lambda: t(1, 1, 6, 6).unfold(2, 3, 1).unfold(3, 3, 1).shape)

# ---------- 线代/归约 ----------
probe("matmul", lambda: (t(64, 64) @ t(64, 64)).sum())
probe("bmm", lambda: torch.bmm(t(4, 32, 32), t(4, 32, 32)).sum())
probe("einsum", lambda: torch.einsum("bhid,bhjd->bhij", t(1, 4, 8, 16), t(1, 4, 8, 16)).sum())
probe("norm", lambda: torch.norm(t(2, 64)))
probe("rsqrt", lambda: torch.rsqrt(t(2, 64).abs() + 0.1).sum())
probe("cumsum", lambda: torch.cumsum(t(2, 64), dim=-1).sum())
probe("logsumexp", lambda: torch.logsumexp(t(2, 64), dim=-1).sum())
probe("argmax", lambda: torch.argmax(t(2, 64), dim=-1))
probe("topk", lambda: torch.topk(t(2, 64), 5, dim=-1))
probe("sort", lambda: torch.sort(t(2, 64), dim=-1))
probe("multinomial", lambda: torch.multinomial(F.softmax(t(100, dtype=torch.float32), dim=-1), 5))
probe("bincount", lambda: torch.bincount(torch.randint(0, 10, (100,), device=D)))
probe("gather", lambda: torch.gather(t(2, 8), 1, torch.randint(0, 8, (2, 4), device=D)))
probe("index_select", lambda: torch.index_select(t(8, 8), 0, torch.tensor([1, 3], device=D)))
probe("scatter_add", lambda: torch.zeros(4, 8, dtype=torch.float16, device=D).scatter_add_(1, torch.randint(0, 8, (4, 4), device=D), t(4, 4)).sum())

# ---------- 逐元素 ----------
for op in ["tanh", "sigmoid", "pow", "exp", "log", "sin", "cos", "flip", "tril", "outer",
           "sign", "isnan", "atan2", "round", "abs"]:
    probe(op, eval(f"lambda: torch.{op}(*([t(2,8)]*2 if op in ('pow','atan2','outer') else [t(2,8)])).sum()"))
probe("where", lambda: torch.where(t(2, 8) > 0, t(2, 8), -t(2, 8)).sum())
probe("clamp", lambda: torch.clamp(t(2, 8), -0.5, 0.5).sum())
probe("hann_window", lambda: torch.hann_window(64, device=D))
probe("kaiser_window", lambda: torch.kaiser_window(64, device=D))
probe("linspace", lambda: torch.linspace(0, 1, 32, device=D))
probe("randn", lambda: torch.randn(4, 4, device=D).sum())
probe("randint", lambda: torch.randint(0, 10, (4,), device=D))
probe("randperm", lambda: torch.randperm(64, device=D))

# ---------- 复数/FFT（vocos、mel 相关）----------
def fft_():
    torch.fft.fft(torch.randn(256, dtype=torch.float32, device=D))
probe("fft.fft", fft_)
def irfft_():
    torch.fft.irfft(torch.view_as_real(torch.randn(129, dtype=torch.complex64, device=D)).to(torch.complex64) if False else torch.fft.rfft(torch.randn(256, dtype=torch.float32, device=D)))
probe("fft.irfft", irfft_)
def ifft_():
    torch.fft.ifft(torch.fft.fft(torch.randn(64, dtype=torch.float32, device=D)))
probe("fft.ifft", ifft_)
def stft_():
    torch.stft(torch.randn(16000, dtype=torch.float32, device=D), 1024, 256, window=torch.hann_window(1024, device=D), return_complex=True)
probe("stft", stft_)
def istft_():
    spec = torch.stft(torch.randn(8000, dtype=torch.float32, device=D), 512, 128, window=torch.hann_window(512, device=D), return_complex=True)
    torch.istft(spec, 512, 128, window=torch.hann_window(512, device=D))
probe("istft", istft_)
def polar_():
    torch.polar(torch.ones(8, dtype=torch.float32, device=D), torch.zeros(8, dtype=torch.float32, device=D))
probe("polar", polar_)
def complex_():
    x = torch.randn(8, 16, dtype=torch.float32, device=D)
    c = torch.view_as_complex(x.reshape(8, 16 // 2, 2).contiguous())
    torch.view_as_real(c).sum()
probe("view_as_complex/real", complex_)
probe("conj_abs", lambda: (torch.view_as_complex(torch.randn(4, 8, 2, device=D)).abs()).sum())

# ---------- bf16 对照 ----------
def bf16_():
    (torch.randn(32, 32, dtype=torch.bfloat16, device=D) @ torch.randn(32, 32, dtype=torch.bfloat16, device=D)).sum()
probe("matmul(bf16,预期失败)", bf16_)

# ---------- autocast ----------
def autocast_():
    with torch.autocast("npu", dtype=torch.float16):
        (t(32, 32) @ t(32, 32)).sum()
probe("autocast(npu,fp16)", autocast_)

# ---------- 汇总 ----------
ok = [r for r in results if r[1] == "OK"]
fail = [r for r in results if r[1] == "FAIL"]
print(f"\n===== TOTAL {len(results)} | OK {len(ok)} | FAIL {len(fail)} =====")
for n, s, m in fail:
    print(f"[FAIL] {n}: {m}")
print("===== OK list =====")
print(", ".join(n for n, _, _ in ok))
