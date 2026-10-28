"""昇腾 NPU（Atlas 300I Duo / Ascend 310P3）兼容层 —— IndexTTS 移植核心补丁。

必须在 import 任何 indextts 模块之前导入：`import npu_compat`

包含（全部经 310P3 真机探针验证，见 ascend-310p-research/probe_ops*.py）：
1. transfer_to_npu：官方垫片，.cuda()/device="cuda"/torch.cuda.* 全局映射到 npu
2. 两个关键开关：allow_internal_format=False（修复 SDPA ERR01007）、
   jit_compile=False（修复 polar/bincount/复数算子调度失败）
3. FFT 家族全局 CPU 化：310P 的 aclnn FFT 内核实测报 NPU function error，
   stft/istft/fft.* 在 CPU 计算后按需回传
4. IndexTTS 参考音频前处理函数的 CPU 包装（mel 谱提取等，每请求仅一次）
"""
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401  cuda -> npu 全局映射

NPU_DEV = int(os.environ.get("NPU_DEV", "0"))
torch.npu.set_device(NPU_DEV)

# 实测有效的关键开关
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

# ---------- FFT 家族全局 CPU 化 ----------
_orig_stft = torch.stft
_orig_istft = torch.istft


def _cpu_stft(x, *a, **k):
    if not torch.is_tensor(x) or x.device.type == "cpu":
        return _orig_stft(x, *a, **k)
    k2 = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in k.items()}
    # 复数谱留在 CPU：复数算子(abs/mul等)在 310P 上不可靠，由调用方在 CPU 上转实数
    return _orig_stft(x.cpu(), *a, **k2)


def _cpu_istft(x, *a, **k):
    if not torch.is_tensor(x) or x.device.type == "cpu":
        return _orig_istft(x, *a, **k)
    dev = x.device
    k2 = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in k.items()}
    return _orig_istft(x.cpu(), *a, **k2).to(dev)


torch.stft = _cpu_stft
torch.istft = _cpu_istft

import torch.fft as _tfft  # noqa: E402

for _name in ("fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fftn", "ifftn"):
    _fn = getattr(_tfft, _name, None)
    if _fn is None:
        continue

    def _mk_fft(fn):
        def wrapped(x, *a, **k):
            if not torch.is_tensor(x) or x.device.type == "cpu":
                return fn(x, *a, **k)
            k2 = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in k.items()}
            return fn(x.cpu(), *a, **k2)  # 复数输出留 CPU
        return wrapped

    setattr(_tfft, _name, _mk_fft(_fn))

# kaiser_window 在 BigVGAN filter.py 中本就在 CPU 上创建（无 device 参数），无需处理


# ---------- torchaudio.save 走 soundfile（torchaudio 2.10 需 torchcodec，镜像内无）----------
try:
    import torchaudio as _ta
    import soundfile as _sf

    def _sf_save(filepath, src, sample_rate, **k):  # noqa: E301
        w = src.detach().cpu().float()
        if w.ndim == 2:  # torchaudio 约定 (channels, time)，soundfile 要 (time, channels)
            w = w.t()
        _sf.write(str(filepath), w.numpy(), sample_rate)

    _ta.save = _sf_save
except Exception as e:  # pragma: no cover
    print(f"[npu_compat] torchaudio.save patch skip: {e}")


# ---------- 函数/方法级 CPU 包装：实数进实数出，中间复数留 CPU ----------
def cpu_wrap(fn):
    """把函数的张量入参搬到 CPU 计算，输出搬回首个非 CPU 入参的设备。"""
    def wrapped(*a, **k):
        tensors = [t for t in a if torch.is_tensor(t)] + \
                  [v for v in k.values() if torch.is_tensor(v)]
        dev = next((t.device for t in tensors if t.device.type != "cpu"), None)
        ca = tuple(t.cpu() if torch.is_tensor(t) else t for t in a)
        ck = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in k.items()}
        out = fn(*ca, **ck)
        if dev is not None and torch.is_tensor(out):
            out = out.to(dev)
        elif dev is not None and isinstance(out, (tuple, list)):
            out = type(out)(o.to(dev) if torch.is_tensor(o) else o for o in out)
        return out
    return wrapped


def install_indextts_patches():
    """在 import indextts 之后调用，安装各模块的 CPU 化补丁。"""
    import importlib

    # 1) s2mel mel 谱提取（参考音频 + 生成路径条件）
    audio = importlib.import_module("indextts.s2mel.modules.audio")
    audio.mel_spectrogram = cpu_wrap(audio.mel_spectrogram)
    # infer_v2_5 以模块级名字引用了该函数，两个命名空间都要打补丁
    inf = importlib.import_module("indextts.infer_v2_5")
    inf.mel_spectrogram = audio.mel_spectrogram

    # 2) openvoice 幅度谱提取
    try:
        mp = importlib.import_module("indextts.s2mel.modules.openvoice.mel_processing")
        for name in ("spectrogram", "mel_spectrogram", "spectrogram_torch"):
            if hasattr(mp, name):
                setattr(mp, name, cpu_wrap(getattr(mp, name)))
    except Exception as e:  # pragma: no cover
        print(f"[npu_compat] skip openvoice mel_processing: {e}")

    # 3) codec（EnhancedCodec）的量化/解码入口：权重已在 CPU（infer_v2_5 钉死），
    #    包装保证 NPU 张量进 CPU 算完回 NPU
    try:
        codec_models = importlib.import_module("indextts.codec.models")
        EnhancedCodec = codec_models.EnhancedCodec
        for mname in ("quantize", "decode", "encode", "forward"):
            if hasattr(EnhancedCodec, mname):
                setattr(EnhancedCodec, mname, cpu_wrap(getattr(EnhancedCodec, mname)))
    except Exception as e:  # pragma: no cover
        print(f"[npu_compat] skip EnhancedCodec: {e}")

    # 4) 参考音频前端链的入口方法（w2v-bert 在 CPU）：NPU 张量进 -> CPU 算 -> 回 NPU
    for mname in ("get_emb", "get_scode"):
        if hasattr(inf.IndexTTS2, mname):
            setattr(inf.IndexTTS2, mname, cpu_wrap(getattr(inf.IndexTTS2, mname)))

    # 5) torchaudio kaldi 特征（fbank 内部在 CPU 建窗/滤波器组，混设备会崩）
    try:
        import torchaudio.compliance.kaldi as _kaldi
        for name in ("fbank", "spectrogram", "mfcc", "log_fbank"):
            if hasattr(_kaldi, name):
                setattr(_kaldi, name, cpu_wrap(getattr(_kaldi, name)))
    except Exception as e:  # pragma: no cover
        print(f"[npu_compat] skip kaldi: {e}")

    # 6) 310P 内核缺陷修复：conv_transpose1d(groups=C>64) 输出错误（C=64 正常，
    #    384/768 全错，逐层二分定位，见 debug_convt 探针）。
    #    用 matmul 等效实现替换（与 CPU 原生输出完全一致，已验证 max|d|=0）。
    import importlib as _imp
    for _mod_name in ("indextts.s2mel.modules.bigvgan.alias_free_activation.torch.resample",
                      "indextts.s2mel.modules.alias_free_torch.resample",
                      "indextts.BigVGAN.alias_free_activation.torch.resample",
                      "indextts.BigVGAN.alias_free_torch.resample"):
        try:
            _rs = _imp.import_module(_mod_name)
        except Exception:
            continue
        if hasattr(_rs, "UpSample1d") and not getattr(_rs, "_npu_convt_fixed", False):
            def _fixed_forward(self, x):
                import torch.nn.functional as _F
                _, C, _ = x.shape
                x2 = _F.pad(x, (self.pad, self.pad), mode="replicate")
                f = self.filter  # (1,1,k)
                B, _, L = x2.shape
                k = f.shape[-1]
                # grouped convT 的等效实现（k 次小循环，无 L 次大循环）:
                # outer[c,t,j] = x[c,t]*f[c,j]，out[c, s*t+j] += outer[c,t,j]
                outer = x2.squeeze(0).unsqueeze(-1) * f[:, 0, :].unsqueeze(1)  # (C,L,k)
                out = x2.new_zeros(C, (L - 1) * self.stride + k)
                for j in range(k):
                    out[:, j:j + self.stride * L:self.stride][:, :L] += outer[..., j]
                out = self.ratio * out.unsqueeze(0)
                return out[..., self.pad_left:-self.pad_right]

            _rs.UpSample1d.forward = _fixed_forward
            _rs._npu_convt_fixed = True
            print(f"[npu_compat] UpSample1d grouped-convT bug 已修复: {_mod_name}")

    print("[npu_compat] IndexTTS NPU 补丁安装完成（310P3）")


def npu_sync():
    torch.npu.synchronize()


print(f"[npu_compat] 已初始化: NPU_DEV={NPU_DEV}, soc={torch.npu.get_soc_version()}, "
      f"torch={torch.__version__}, torch_npu={torch_npu.__version__}")
