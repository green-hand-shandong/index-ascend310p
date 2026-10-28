#!/usr/bin/env bash
# 在带 NPU 直通的容器内执行命令（vLLM 系统栈，不带 IndexTTS 的隔离 transformers）
# 用法: bash docker_exec_vllm.sh <容器名> <工作目录> <命令...>
# 容器启动示例（vllm-ascend 310P 镜像，单芯片即可）:
#   docker run -d --name indextts-310p \
#     --device /dev/davinci0 --device /dev/davinci_manager \
#     --device /dev/devmm_svm --device /dev/hisi_hdc \
#     -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
#     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
#     -v $PWD:/workspace -w /workspace \
#     quay.io/ascend/vllm-ascend:v0.30.0rc2-310p sleep infinity
set -e
C="${1:?容器名}"; WD="${2:?工作目录}"; shift 2
docker exec -e NPU_DEV="${NPU_DEV:-0}" -w "$WD" "$C" bash -c "
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:\$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=\${ASCEND_RT_VISIBLE_DEVICES:-0}
$*
"
