# 310P 真机探针套件

在容器内逐个运行（挂载方式见 scripts/docker_exec.sh 注释）：

| 脚本 | 内容 |
|---|---|
| probe1_basic_ops.py | 70 算子基线扫描（IndexTTS 全推理路径算子清单） |
| probe2_sdpa_fft_workarounds.py | SDPA 3D/4D、FFT 绕行方案验证 |
| probe3_jit_compile.py | jit_compile=False 对 fft/polar/bincount/sdpa 的效果 |
| probe4_gpt2_transfer.py | transfer_to_npu 垫片 + HF GPT2 fp16 真模型 generate |
| probe5_conv_accuracy.py | conv1d/convT 各 shape 精度对照（发现 convT 缺陷） |
| probe6_bigvgan_cpu_vs_npu.py | BigVGAN 整体 CPU vs NPU 输出对照 |
| probe7_layer_bisect.py | 全模块 hook 按前向顺序二分定位首个分歧层 |

结论汇总见 ../docs/ascend310p-op-support.md 与 ../docs/conv-transpose1d-groups-bug.md。
