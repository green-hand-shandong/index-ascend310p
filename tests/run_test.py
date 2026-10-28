"""IndexTTS-2.5 昇腾 310P 端到端合成测试。

环境变量:
    INDEXTTS_DIR  打补丁后的 index-tts 代码目录 (默认 ../workspace/index-tts)
    CKPT_DIR      模型目录 (默认 ../workspace/checkpoints)
    REF_WAV       参考音频 (可选, 任意 3~15s 干净人声)
用法:
    python run_test.py [输出.wav] [要合成的文本]
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import npu_compat  # noqa: F401  必须最先导入（本仓库根目录）
import time

INDEXTTS_DIR = os.environ.get("INDEXTTS_DIR", os.path.join(os.path.dirname(__file__), "..", "workspace", "index-tts"))
CKPT = os.environ.get("CKPT_DIR", os.path.join(os.path.dirname(__file__), "..", "workspace", "checkpoints"))
REF = os.environ.get("REF_WAV", "")
TEXT = sys.argv[2] if len(sys.argv) > 2 else "欢迎来到昇腾三一零P语音合成测试，这段语音完全在国产算力上生成。"
OUT = sys.argv[1] if len(sys.argv) > 1 else "out_npu.wav"

if not REF:
    print("请通过 REF_WAV 环境变量指定 3~15 秒干净人声参考音频（用于音色克隆）")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(INDEXTTS_DIR))
import torch  # noqa: E402
npu_compat.install_indextts_patches()
from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

t0 = time.time()
tts = IndexTTS2(cfg_path=os.path.join(CKPT, "config.yaml"), model_dir=CKPT, use_qwen_emo=False)
print(f">> 模型加载耗时: {time.time()-t0:.1f}s, device={tts.device}")

for i in range(2):  # 第二次为稳态性能
    t1 = time.time()
    tts.infer(REF, TEXT, OUT if i == 0 else OUT.replace(".wav", "_2.wav"), lang="zh", verbose=(i == 0))
    torch.npu.synchronize()
    print(f">> 第{i+1}次合成耗时: {time.time()-t1:.2f}s")

# ---------- 音频有效性验证 ----------
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

for path in (OUT, OUT.replace(".wav", "_2.wav")):
    try:
        data, sr = sf.read(path)
    except Exception as e:
        print(f"[FAIL] {path}: {e}")
        continue
    dur, nan = len(data) / sr, bool(np.isnan(data).any())
    rms, peak = float(np.sqrt((data ** 2).mean())), float(np.abs(data).max())
    clip = float(np.mean(np.abs(data) > 0.999) * 100)
    zcr = float(np.mean(np.abs(np.diff(np.sign(data))) > 0))
    status = "OK" if (not nan and dur > 1.0 and clip < 1.0) else "SUSPECT"
    print(f"[{status}] {path}: sr={sr} dur={dur:.2f}s rms={rms:.4f} peak={peak:.3f} "
          f"clip%={clip:.3f} nan={nan} zcr={zcr:.3f}")
