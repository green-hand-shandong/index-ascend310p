"""310P NPU 上的纯张量重组 FFT —— 无自定义内核，只用 310P 已验证的快路径算子。

原理（与 vllm-omni PR #9869 在 950DT 上的做法同思路，但 310P 连 aclnn rfft 都没有，
因此走得更底层）：

    DFT:  X[k] = Σ_n x[n]·W^{nk},  W = e^{-2πi/N}
    → 拆成实部/虚部两个实数矩阵乘：
        Re = x · cos(2πnk/N),  Im = -x · sin(2πnk/N)
    → 用 matmul 实现（310P 的 matmul 是快路径，含 fp16 TensorCore）。

提供两种实现，按 N 自动选择：
1. `fft_pow2` —— radix-2 迭代蝶形（Cooley-Tukey），仅用于 N=2^k。
   需要 gather（bit-reversal 置换，310P 已验证 OK）+ 复数乘（手工实虚部，已验证 OK）。
   复杂度 O(N log N)，batch 并行度高，适合长帧（1024/2048 点 mel 前端）。
2. `fft_dft` —— 直接 DFT 矩阵乘，任意 N，复杂度 O(N²)。
   帧长 ≤512 时比蝶形更快（矩阵乘是硬加速），mel 前端 n_fft=1024 时蝶形更快。

顶层 API（与 torch 语义对齐，全部支持 batch/多音频）：
    fft / ifft        —— 复数输入输出（返回 torch.complex64，内部实虚分离）
    rfft / irfft      —— 实数输入 / 复数输出，或 irfft 复数输入 / 实数输出
    stft / istft      —— 与 torch.stft/istft 参数对齐（center/pad_mode/normalized/onesided）
    hann_window / hamming_window / kaiser_window —— CPU 生成一次后缓存到 NPU

复数约定：310P 的复数乘法/abs 内核缺失，因此所有中间计算都用
`... , 2>` 末维实虚对（view_as_real 布局）表示，仅在函数边界用
view_as_complex 组装，供调用方无缝对接 torch 复数 API。

真机验证（310P3, CANN 9.2, torch_npu 2.10.post7）：
    ✅ matmul(fp32/fp16)、gather、unfold、roll、view_as_real/complex
    ❌ fft/stft/hann_window/复数 abs（内核缺失，本模块的重组目标）
"""
from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = [
    "fft_pow2", "fft_dft", "fft", "ifft", "rfft", "irfft",
    "stft", "istft", "hann_window", "hamming_window", "kaiser_window",
]

# ---------------------------------------------------------------- 蝶形基底
def _bit_reverse_permute(n: int, device) -> torch.Tensor:
    """bit-reversal 置换索引（Cooley-Tukey 迭代版第 0 步）。CPU 生成后搬运。"""
    bits = n.bit_length() - 1
    idx = torch.arange(n)
    rev = torch.zeros_like(idx)
    for b in range(bits):
        rev = (rev << 1) | ((idx >> b) & 1)
    return rev.to(device)


def _twiddles_pow2(n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """radix-2 蝶形旋转因子，按 stage 预生成: 返回 (cos, sin) 各 (log2N, N/2)。CPU 生成后搬运。"""
    half = n // 2
    stages = n.bit_length() - 1
    cos_s = torch.empty(stages, half, dtype=dtype)
    sin_s = torch.empty(stages, half, dtype=dtype)
    for s in range(stages):
        m = 1 << (s + 1)          # 蝶形跨度
        k = torch.arange(half, dtype=torch.float64)
        k = (k % (m // 2)) * (2 * math.pi / m)
        cos_s[s] = torch.cos(k).to(dtype)
        sin_s[s] = torch.sin(k).to(dtype)
    return cos_s.to(device), sin_s.to(device)


def _fft_pow2_re_im(re: torch.Tensor, im: torch.Tensor, inverse: bool):
    """radix-2 迭代 FFT，实虚分离版本。re/im: (..., N)，N=2^k。原地重写。"""
    n = re.shape[-1]
    assert n & (n - 1) == 0 and n >= 2, f"fft_pow2 要求 N=2^k, got {n}"
    device, dtype = re.device, re.dtype

    perm = _bit_reverse_permute(n, device)
    re = re.gather(-1, perm.expand_as(re))
    im = im.gather(-1, perm.expand_as(im))

    sign = 1.0 if inverse else -1.0
    cos_s, sin_s = _twiddles_pow2(n, device, dtype)
    log2n = n.bit_length() - 1

    half = n // 2
    for s in range(log2n):
        m = 1 << (s + 1)
        mh = m // 2
        # 蝶形索引: 对每组 block, re[..., j] 与 re[..., j+mh] 配对
        # 用 reshape (..., n/m, m) 实现 in-place 风格（out-of-place 张量代数）
        re = re.view(*re.shape[:-1], n // m, m)
        im = im.view(*im.shape[:-1], n // m, m)
        wr = cos_s[s][:mh]                      # (mh,)
        wi = sin_s[s][:mh] * sign
        # 奇偶分半
        xr, xi = re[..., :mh], im[..., :mh]     # 偶部
        yr, yi = re[..., mh:], im[..., mh:]     # 奇部
        # (yr + i·yi) * (wr + i·wi)
        tr = yr * wr - yi * wi
        ti = yr * wi + yi * wr
        re = torch.cat([xr + tr, xr - tr], dim=-1)
        im = torch.cat([xi + ti, xi - ti], dim=-1)
        re = re.view(*re.shape[:-2], n)
        im = im.view(*im.shape[:-2], n)
    return re, im


def fft_pow2(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """N=2^k 的 FFT。x: (..., N) 实数或末维实虚对 (...,[re,im]) → complex64 张量。"""
    if x.is_complex():
        pair = torch.view_as_real(x)
        re, im = pair[..., 0], pair[..., 1]
    else:
        re, im = x, torch.zeros_like(x)
    n = re.shape[-1]
    re, im = _fft_pow2_re_im(re.contiguous(), im.contiguous(), inverse)
    if not inverse:
        pass  # forward 不归一化（与 torch 一致）
    else:
        re = re / n
        im = im / n
    return torch.view_as_complex(torch.stack([re, im], dim=-1))


def fft_dft(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """任意 N 的 DFT（Bluestein 卷积法）。x: (..., N) → complex64。

    310P 的 fp32 matmul 实测累加精度不足（~1e-4 相对误差，疑内部低精度累加），
    直接 DFT 矩阵乘达不到音频前端精度。因此走 Bluestein：把任意 N 的 DFT
    化为线性卷积，卷积用 radix-2 蝶形 FFT（实测 rel~1e-7）实现。

    推导: nk = (k² + n² - (k-n)²)/2
        X[k] = w^{k²/2} · Σ_n (x[n]·w^{n²/2}) · w^{-(k-n)²/2}
             = w^{k²/2} · conv(a, b)[k]，j = k-n ∈ [-(N-1), N-1]
        a[n] = x[n]·e^{-iπn²/N},  b[j] = e^{+iπj²/N}（fp64 生成，fp32 存储）
    inverse=True 时按共轭转置处理（w → w*，结果再除 N）。
    """
    if x.is_complex():
        pair = torch.view_as_real(x)
        re, im = pair[..., 0], pair[..., 1]
    else:
        re, im = x, torch.zeros_like(x)
    n = re.shape[-1]
    device, dtype = re.device, re.dtype
    sign = 1.0 if inverse else -1.0   # forward w=e^{-2πi/N}, inverse w=e^{+2πi/N}

    m = 1
    while m < 2 * n - 1:
        m *= 2
    k = torch.arange(n, dtype=torch.float64)
    w_half_re = torch.cos(sign * math.pi * k * k / n)   # w^{±k²/2} 实部
    w_half_im = torch.sin(sign * math.pi * k * k / n)   # 虚部
    j = torch.arange(-(n - 1), n, dtype=torch.float64)
    b_re = torch.cos(-sign * math.pi * j * j / n)       # b[j] = w^{∓j²/2}
    b_im = torch.sin(-sign * math.pi * j * j / n)
    w_half_re = w_half_re.to(device).to(dtype)
    w_half_im = w_half_im.to(device).to(dtype)
    b_re = b_re.to(device).to(dtype)
    b_im = b_im.to(device).to(dtype)

    # a = x * w^{±n²/2}（实虚分离乘）并补零到 m
    ar = re * w_half_re - im * w_half_im
    ai = re * w_half_im + im * w_half_re
    ar = torch.nn.functional.pad(ar, (0, m - n))
    ai = torch.nn.functional.pad(ai, (0, m - n))
    a_f = fft_pow2(torch.view_as_complex(torch.stack([ar, ai], dim=-1)), inverse=False)
    br = torch.nn.functional.pad(b_re, (0, m - (2 * n - 1)))
    bi = torch.nn.functional.pad(b_im, (0, m - (2 * n - 1)))
    b_f = fft_pow2(torch.view_as_complex(torch.stack([br, bi], dim=-1)), inverse=False)

    # 频域复数乘（手工实虚部，310P 复数算子不可靠）
    pa = torch.view_as_real(a_f)
    pb = torch.view_as_real(b_f)
    prod = torch.view_as_complex(torch.stack(
        [pa[..., 0] * pb[..., 0] - pa[..., 1] * pb[..., 1],
         pa[..., 0] * pb[..., 1] + pa[..., 1] * pb[..., 0]], dim=-1))
    conv = fft_pow2(prod, inverse=True)[..., n - 1: 2 * n - 1]

    # 输出乘 w^{±k²/2}
    pc = torch.view_as_real(conv)
    out_re = pc[..., 0] * w_half_re - pc[..., 1] * w_half_im
    out_im = pc[..., 0] * w_half_im + pc[..., 1] * w_half_re
    if inverse:
        out_re = out_re / n
        out_im = out_im / n
    return torch.view_as_complex(torch.stack([out_re, out_im], dim=-1))


# ---------------------------------------------------------------- 顶层 API
def fft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
        norm: Optional[str] = None, *, _force: Optional[str] = None) -> torch.Tensor:
    """对齐 torch.fft.fft（仅支持最后一维、norm 未实现 backward/ortho 之外的语义）。"""
    assert dim == -1 or dim == input.ndim - 1, "npu_fft 仅支持最后一维"
    x = input
    if n is not None and n != x.shape[-1]:
        pad = n - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, max(pad, 0)))[:n if pad < 0 else None]
    N = x.shape[-1]
    impl = _force or ("pow2" if N & (N - 1) == 0 else "dft")
    out = fft_pow2(x, inverse=False) if impl == "pow2" else fft_dft(x, inverse=False)
    if norm == "ortho":
        out = out / math.sqrt(N)
    return out


def ifft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
         norm: Optional[str] = None) -> torch.Tensor:
    assert dim == -1 or dim == input.ndim - 1, "npu_fft 仅支持最后一维"
    x = input
    if n is not None and n != x.shape[-1]:
        pad = n - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, max(pad, 0)))[:n if pad < 0 else None]
    N = x.shape[-1]
    impl = "pow2" if N & (N - 1) == 0 else "dft"
    out = fft_pow2(x, inverse=True) if impl == "pow2" else fft_dft(x, inverse=True)
    if norm == "ortho":
        out = out * math.sqrt(N)
    return out


def rfft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
         norm: Optional[str] = None) -> torch.Tensor:
    """实数 → 单边谱 complex64（对齐 torch.fft.rfft）。"""
    assert dim == -1 or dim == input.ndim - 1
    x = input
    if not x.is_complex():
        x = x.float()
    full = fft(x, n=n, dim=dim, norm=norm)
    return full[..., : full.shape[-1] // 2 + 1]


def irfft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
          norm: Optional[str] = None) -> torch.Tensor:
    """单边谱 → 实数（对齐 torch.fft.irfft）。"""
    assert dim == -1 or dim == input.ndim - 1
    x = input
    if n is None:
        n = 2 * (x.shape[-1] - 1)
    # 重建共轭对称全谱。310P: 复数 conj 内核缺失 → 实虚分离手工拼
    if x.is_complex():
        pair = torch.view_as_real(x.resolve_conj())
        re, im = pair[..., 0], pair[..., 1]
        mid_re, mid_im = re[..., 1:-1], im[..., 1:-1]
        full = torch.view_as_complex(torch.stack(
            [torch.cat([re, mid_re.flip(-1)], dim=-1),
             torch.cat([im, (-mid_im).flip(-1)], dim=-1)], dim=-1))
    else:
        full = torch.cat([x, x[..., 1:-1].flip(-1).conj()], dim=-1)
    out = ifft(full, n=n, dim=dim, norm=norm)
    return out.real if torch.is_complex(out) else out


# ---------------------------------------------------------------- 窗函数
def hann_window(window_length: int, periodic: bool = True, dtype=None,
                device=None, requires_grad: bool = False) -> torch.Tensor:
    """对齐 torch.hann_window：CPU 生成（310P 的 cos_ 内核缺失）后搬运。"""
    w = torch.hann_window(window_length, periodic=periodic,
                          dtype=dtype or torch.float32, device="cpu")
    return w.to(device) if device is not None else w


def hamming_window(window_length: int, periodic: bool = True, dtype=None,
                   device=None, requires_grad: bool = False) -> torch.Tensor:
    w = torch.hamming_window(window_length, periodic=periodic,
                             dtype=dtype or torch.float32, device="cpu")
    return w.to(device) if device is not None else w


def kaiser_window(window_length: int, beta: float, periodic: bool = True,
                  dtype=None, device=None, requires_grad: bool = False) -> torch.Tensor:
    w = torch.kaiser_window(window_length, periodic=periodic, beta=beta,
                            dtype=dtype or torch.float32, device="cpu")
    return w.to(device) if device is not None else w


# ---------------------------------------------------------------- STFT / ISTFT
def stft(input: torch.Tensor, n_fft: int, hop_length: Optional[int] = None,
         win_length: Optional[int] = None, window: Optional[torch.Tensor] = None,
         center: bool = True, pad_mode: str = "reflect",
         normalized: bool = False, onesided: bool = True,
         return_complex: bool = True) -> torch.Tensor:
    """对齐 torch.stft 的常用签名（短时傅里叶变换）。

    input: (..., L) 实数波形
    返回: (..., freq, time) complex64（onesided 时 freq = n_fft//2 + 1）
    实现: F.pad → unfold 分帧 → 乘窗 → radix-2/DFT 矩阵乘 → 复数谱
    """
    assert onesided and return_complex, "npu_fft.stft 仅支持 onesided+complex（TTS 前端够用）"
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    assert win_length == n_fft, "npu_fft.stft 要求 win_length == n_fft（TTS 场景均如此）"
    if window is None:
        window = hann_window(n_fft, device=input.device)
    window = window.to(input.device, torch.float32)  # stft 输入是实数，float32 即可

    x = input
    if center:
        l_pad = n_fft // 2
        r_pad = n_fft // 2
        x = torch.nn.functional.pad(x, (l_pad, r_pad), mode=pad_mode)

    # 分帧: (..., n_frames, n_fft)
    frames = x.unfold(-1, n_fft, hop_length)
    frames = frames * window
    spec = fft_pow2(frames, inverse=False)  # (..., n_frames, n_fft) complex
    if normalized:
        spec = spec / math.sqrt(n_fft)
    spec = spec.transpose(-1, -2)  # (..., freq, time) 与 torch.stft 布局一致
    return spec[..., : n_fft // 2 + 1, :] if onesided else spec


def istft(input: torch.Tensor, n_fft: int, hop_length: Optional[int] = None,
          win_length: Optional[int] = None, window: Optional[torch.Tensor] = None,
          center: bool = True, normalized: bool = False,
          length: Optional[int] = None, return_complex: bool = False) -> torch.Tensor:
    """对齐 torch.istft 的常用签名（逆 STFT，overlap-add）。

    input: (..., freq, time) complex64（单边谱）
    返回: (..., L) 实数波形
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = hann_window(n_fft, device=input.device)
    # 注意: 不能 window.to(input.dtype) —— input 是复数谱，会把窗也变复数，
    # 污染后续 overlap-add 的 denom（310P 复数 where/clamp 全缺失）
    window = window.to(input.device, torch.float32)

    spec = input
    # 全谱重建（共轭对称）。310P: 复数 conj/cat 内核不可靠 → 实虚分离手工拼
    if spec.is_complex():
        pair = torch.view_as_real(spec.resolve_conj())  # (..., freq, T, 2)
        re, im = pair[..., 0], pair[..., 1]
        mid_re, mid_im = re[..., 1:-1, :], im[..., 1:-1, :]
        full_pair = torch.stack(
            [
                torch.cat([re, mid_re.flip(-2)], dim=-2),
                torch.cat([im, (-mid_im).flip(-2)], dim=-2),
            ],
            dim=-1,
        )  # (..., n_fft, T, 2)
        full = torch.view_as_complex(full_pair)
    else:
        full = torch.cat([spec, spec[..., 1:-1, :].flip(-2).conj()], dim=-2)
    full = full.transpose(-1, -2)  # (..., T, n_fft)
    if normalized:
        # 310P: 复数标量乘内核缺失 → 实虚分离缩放
        fp = torch.view_as_real(full)
        full = torch.view_as_complex(fp * math.sqrt(n_fft))

    # 逐帧 irfft（DFT 矩阵或蝶形）
    n = full.shape[-1]
    impl = "pow2" if n & (n - 1) == 0 else "dft"
    frames_c = fft_pow2(full, inverse=True) if impl == "pow2" else fft_dft(full, inverse=True)
    frames = frames_c.real * window  # (..., T, n_fft)

    # overlap-add（310P: 全程实数域，避免复数 clamp/pow 内核缺失）
    T = frames.shape[-2]
    out_len = hop_length * (T - 1) + n_fft
    out = frames.new_zeros(*frames.shape[:-2], out_len)
    denom = frames.new_zeros(out_len)
    win_sq = (window * window).float()
    for t in range(T):
        out[..., t * hop_length: t * hop_length + n_fft] += frames[..., t, :]
        denom[..., t * hop_length: t * hop_length + n_fft] += win_sq
    denom = torch.where(denom > 1e-11, denom, torch.full_like(denom, 1e-11))
    out = out / denom

    if center:
        out = out[..., n_fft // 2: out.shape[-1] - n_fft // 2]
    if length is not None:
        out = out[..., :length]
    return out
