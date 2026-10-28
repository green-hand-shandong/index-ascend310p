"""合成音频深度验证：削波统计 + ASR 回读（paraformer，需 funasr）。

用法: python validate_audio.py <合成音频.wav> [更多.wav ...]
"""
import sys

import numpy as np
import soundfile as sf

paths = sys.argv[1:] or ["out_npu.wav", "out_npu_2.wav"]
for p in paths:
    d, sr = sf.read(p)
    clip = np.mean(np.abs(d) > 0.999) * 100
    print(f"{p}: sr={sr} dur={len(d)/sr:.2f}s clip%={clip:.3f} "
          f"rms={np.sqrt((d**2).mean()):.3f} peak={np.abs(d).max():.4f} nan={np.isnan(d).any()}")

print("=== ASR 回读（funasr paraformer-zh） ===")
from funasr import AutoModel  # noqa: E402
import librosa  # noqa: E402

asr = AutoModel(model="paraformer-zh", hub="ms")
for p in paths:
    d, sr = sf.read(p)
    d16 = librosa.resample(d, orig_sr=sr, target_sr=16000)
    sf.write("/tmp/asr_in.wav", d16, 16000)
    print(p, "->", asr.generate(input="/tmp/asr_in.wav")[0]["text"])
