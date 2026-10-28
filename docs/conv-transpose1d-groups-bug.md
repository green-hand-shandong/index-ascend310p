# 310P `conv_transpose1d(groups=C>64)` 内核缺陷：定位与修复

> 这是 IndexTTS-2.5 在昇腾 310P 上适配时发现的 **op-plugin 内核级数值缺陷**，
> 目前公开资料中未见记载。torch_npu 2.10.0.post7 / CANN 9.2 / 310P3 实测。

## 现象

BigVGAN 声码器输出波形大面积饱和（~87% 采样点钳在 ±1，std 从 CPU 参考的 0.24 膨胀到 0.72），
合成语音为爆音噪声。GPT/DiT（s2mel）各级数值均正常。

## 定位过程（逐层二分，复现脚本见 `probes/`）

1. **`probe6_bigvgan_cpu_vs_npu.py`**：同一输入跑 BigVGAN，CPU 与 NPU 输出 mean|d|=0.62 —— 问题在 BigVGAN 内部。
2. **`probe7_layer_bisect.py`**（全模块 hook 按前向顺序对比）：首个爆炸点在 `resblocks.3.convs1.0`，但单独测该卷积又完全正常 —— 问题在其输入。
3. 继续二分到 `Activation1d`（Snake 激活 + 抗混叠上下采样）内部：**`UpSample1d.forward` 中的 `F.conv_transpose1d(x, filter.expand(C,-1,-1), stride=2, groups=C)` 首步即爆（max|d|≈18）**。
4. 最小复现（`probes/probe5_conv_accuracy.py` + convT 专项）：

| groups | 结果 |
|---|---|
| C=8 / 64 | ✅ max\|d\| ≈ 3e-4（正常数值噪声） |
| C=384 / 768 | ❌ max\|d\| ≈ 2~19（输出完全错误） |

即：**分组转置卷积在组数 > 64（超过某个内核分块阈值）时输出错误**，无任何报错或告警。

## 修复：数学等效实现（无精度损失）

分组 convT 每组独立：`out[c, s·t+j] += x[c,t] · f[c,j]`。用 outer 积 + k 次步长切片散射替代
（k=12 次小循环，远小于原 L≈2304 次的朴素循环）：

```python
def grouped_convt_equivalent(x2, f, stride):  # x2:(1,C,L) f:(1,1,k)
    B, C, L = x2.shape
    k = f.shape[-1]
    outer = x2.squeeze(0).unsqueeze(-1) * f[:, 0, :].unsqueeze(1)   # (C, L, k)
    out = x2.new_zeros(C, (L - 1) * stride + k)
    for j in range(k):
        out[:, j:j + stride * L:stride][:, :L] += outer[..., j]
    return out.unsqueeze(0)
```

实测（本机 310P3）：
- 与 CPU 原生 `conv_transpose1d(groups=C)` 输出 **max|d| = 0**
- 性能 23.8 ms/次，**快于原生 NPU convT 的 30.2 ms**（768×576×12 场景）

本仓库 `npu_compat.py` 在 `install_indextts_patches()` 中自动 monkeypatch 了
IndexTTS 两处 `UpSample1d`（BigVGAN 与 s2mel 副本），对上游代码零侵入。

## 建议

- 若你在 310P 上跑其他含分组 convT 的模型（如其他 GAN 声码器、扩散上采样头）且出现"无报错但输出饱和"，优先怀疑此缺陷。
- 更干净的根治方案是给 op-plugin 的 convT 分组内核修 bug / 升级 CANN，欢迎在 issue 中同步新版本验证结果。
