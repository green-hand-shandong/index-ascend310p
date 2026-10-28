"""探针2：验证失败算子的已知解法是否在 310P 上生效。"""
import torch
import torch_npu
import torch.nn.functional as F

torch.npu.set_device(0)
D = torch.device("npu:0")
print("soc:", torch.npu.get_soc_version())

# 关键开关：禁用内部格式转换（ND -> FRACTAL_NZ），解决 sdpa 格式错误与复数算子
torch.npu.config.allow_internal_format = False

results = []
def probe(name, fn):
    try:
        fn()
        results.append((name, "OK", ""))
    except Exception as e:
        results.append((name, "FAIL", str(e).replace("\n", " ")[:150]))

def t(*shape, dtype=torch.float16):
    return torch.randn(*shape, dtype=dtype, device=D)

# --- SDPA 变体 ---
def sdpa_2d_causal():
    q, k, v = t(8, 64, 64), t(8, 128, 64), t(8, 128, 64)  # 3维 (B*H, L, D)
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
probe("sdpa(3D,is_causal,noIF)", sdpa_2d_causal)

def sdpa_4d_causal():
    q, k, v = t(1, 8, 64, 64), t(1, 8, 128, 64), t(1, 8, 128, 64)
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
probe("sdpa(4D,is_causal,noIF)", sdpa_4d_causal)

def sdpa_4d_mask():
    q, k, v = t(1, 8, 32, 64), t(1, 8, 32, 64), t(1, 8, 32, 64)
    mask = torch.ones(32, 32, dtype=torch.bool, device=D).tril()
    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
probe("sdpa(4D,2Dmask,noIF)", sdpa_4d_mask)

def sdpa_4d_mask4d():
    q, k, v = t(1, 8, 32, 64), t(1, 8, 32, 64), t(1, 8, 32, 64)
    mask = torch.ones(1, 8, 32, 32, dtype=torch.bool, device=D).tril()
    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
probe("sdpa(4D,4Dmask,noIF)", sdpa_4d_mask4d)

def sdpa_transposed():
    # HQ-Qwen式转置布局 (B,H,D,L)
    q, k, v = t(1, 8, 64, 64), t(1, 8, 64, 128), t(1, 8, 128, 64)
    F.scaled_dot_product_attention(q, k, v.transpose(-2, -1).transpose(-2, -1), is_causal=True) if False else None
    F.scaled_dot_product_attention(q.transpose(-2, -1), k, v, is_causal=True)
probe("sdpa(transposed-q,noIF)", sdpa_transposed)

# --- eager attention（GPT2 eager 路径）---
def eager_attn():
    q, k, v = t(1, 8, 64, 64), t(1, 8, 128, 64), t(1, 8, 128, 64)
    att = (q @ k.transpose(-2, -1)) * (1.0 / 64 ** 0.5)
    mask = torch.ones(64, 128, dtype=torch.bool, device=D).tril()
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    (att @ v).sum()
probe("eager_attention(fp16)", eager_attn)

# --- 复数/FFT ---
def fft_():
    torch.fft.fft(torch.randn(256, dtype=torch.float32, device=D))
probe("fft.fft(noIF)", fft_)
def rfft_():
    torch.fft.rfft(torch.randn(256, dtype=torch.float32, device=D))
probe("fft.rfft(noIF)", rfft_)
def stft_():
    torch.stft(torch.randn(16000, dtype=torch.float32, device=D), 1024, 256,
               window=torch.hann_window(1024, device=D), return_complex=True)
probe("stft(noIF)", stft_)
def istft_():
    spec = torch.stft(torch.randn(8000, dtype=torch.float32, device=D), 512, 128,
                      window=torch.hann_window(512, device=D), return_complex=True)
    torch.istft(spec, 512, 128, window=torch.hann_window(512, device=D))
probe("istft(noIF)", istft_)
def polar_():
    torch.polar(torch.ones(8, dtype=torch.float32, device=D), torch.zeros(8, dtype=torch.float32, device=D))
probe("polar(noIF)", polar_)
def complex_abs_():
    c = torch.view_as_complex(torch.randn(4, 8, 2, device=D))
    c.abs().sum()
probe("complex.abs(noIF)", complex_abs_)

# --- 修正后的 fold / flip / outer / bincount ---
def fold_():
    F.fold(t(1, 9, 8), (4, 5), (3, 3))  # N, C*k*k, L
probe("fold(修正)", fold_)
def flip_():
    torch.flip(t(1, 8, 16), dims=[-1]).sum()
probe("flip(修正)", flip_)
def outer_():
    torch.outer(t(8), t(8)).sum()
probe("outer(修正)", outer_)
def bincount_():
    torch.bincount(torch.randint(0, 10, (100,), device=D).long())
probe("bincount(int64)", bincount_)

# --- kaiser_window：CPU创建后搬运（模拟修补） ---
def kaiser_cpu_then_npu():
    w = torch.kaiser_window(64, dtype=torch.float32).to(D)
    (t(1, 8, 64, dtype=torch.float32) * w).sum()
probe("kaiser_window(CPU建+NPU用)", kaiser_cpu_then_npu)

# --- conv1d 在 noIF 模式下回归（确认开关没有破坏已通过项） ---
def conv1d_():
    F.conv1d(t(1, 64, 100), t(128, 64, 5), padding=2)
probe("conv1d(noIF回归)", conv1d_)
def matmul_():
    (t(64, 64) @ t(64, 64)).sum()
probe("matmul(noIF回归)", matmul_)

ok = [r for r in results if r[1] == "OK"]
fail = [r for r in results if r[1] == "FAIL"]
print(f"\n===== TOTAL {len(results)} | OK {len(ok)} | FAIL {len(fail)} =====")
for n, s, m in fail:
    print(f"[FAIL] {n}: {m}")
print("OK:", ", ".join(n for n, _, _ in ok))
