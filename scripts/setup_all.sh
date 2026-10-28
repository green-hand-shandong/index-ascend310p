#!/usr/bin/env bash
# IndexTTS-2.5 昇腾 310P 一键部署脚本
# 用法: bash setup_all.sh [目标目录]   (默认 ./workspace)
set -e
WS="${1:-workspace}"
mkdir -p "$WS"
cd "$WS"

echo "== [1/3] 克隆上游 index-tts 并打补丁 =="
if [ ! -d index-tts/.git ]; then
    git clone --depth 1 https://github.com/index-tts/index-tts.git index-tts \
        || git clone --depth 1 https://gh-proxy.com/https://github.com/index-tts/index-tts.git index-tts \
        || git clone --depth 1 https://gitee.com/chongho/index-tts.git index-tts
fi
PATCH="$(cd "$(dirname "$0")/.." && pwd)/patches/0001-indextts-2.5-ascend310p-port.patch"
cd index-tts
if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH" && echo "补丁已应用"
else
    echo "补丁已存在或已应用，跳过"
fi
cd ..

echo "== [2/3] 下载模型权重 (~6GB) =="
mkdir -p checkpoints/hf_cache/{bigvgan,w2v-bert-2.0}
MS="https://www.modelscope.cn/api/v1/models/IndexTeam/IndexTTS-2.5/repo?Revision=master"
for f in gpt.pth codec.pth s2mel.pth config.yaml feat1.pt feat2.pt wav2vec2bert_stats.pt multilingual_zh_ja_yue_char_del.tiktoken; do
    [ -s "checkpoints/$f" ] || curl -sL --retry 3 -o "checkpoints/$f" "${MS}&FilePath=$f"
done
for f in .gitkeep Modelfile added_tokens.json chat_template.jinja config.json generation_config.json merges.txt model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
    [ -s "checkpoints/qwen0.6bemo4-merge/$f" ] || {
        mkdir -p checkpoints/qwen0.6bemo4-merge
        curl -sL --retry 3 -o "checkpoints/qwen0.6bemo4-merge/$f" "${MS}&FilePath=qwen0.6bemo4-merge/$f"; }
done
# 辅助模型: MaskGCT / CAMPPlus(ModelScope) + BigVGAN / w2v-bert(hf-mirror)
[ -s checkpoints/hf_cache/semantic_codec_model.safetensors ] || curl -sL --retry 3 -o checkpoints/hf_cache/semantic_codec_model.safetensors \
    "https://www.modelscope.cn/api/v1/models/amphion/MaskGCT/repo?Revision=master&FilePath=semantic_codec/model.safetensors"
[ -s checkpoints/hf_cache/campplus_cn_common.bin ] || curl -sL --retry 3 -o checkpoints/hf_cache/campplus_cn_common.bin \
    "https://www.modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=campplus_cn_common.bin"
for f in config.json bigvgan_generator.pt; do
    [ -s "checkpoints/hf_cache/bigvgan/$f" ] || curl -sL --retry 3 -o "checkpoints/hf_cache/bigvgan/$f" \
        "https://hf-mirror.com/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/$f"
done
W2V="https://www.modelscope.cn/api/v1/models/AI-ModelScope/w2v-bert-2.0/repo?Revision=master"
for f in config.json configuration.json preprocessor_config.json model.safetensors; do
    [ -s "checkpoints/hf_cache/w2v-bert-2.0/$f" ] || curl -sL --retry 3 -o "checkpoints/hf_cache/w2v-bert-2.0/$f" "${W2V}&FilePath=$f"
done

echo "== [3/3] 容器内依赖安装 =="
echo "请参考 README「快速开始」用 vllm-ascend 310P 镜像启动容器后执行:"
echo "  pip install transformers==4.52.1 tokenizers==0.21.0 accelerate==1.8.1 \\"
echo "      torchaudio==2.10.0 librosa soundfile scipy einops omegaconf munch \\"
echo "      jieba jieba_fast cn2an g2p-en json5 sentencepiece wetext whisper \\"
echo "      fugashi unidic-lite modelscope addict simplejson sortedcontainers \\"
echo "      'huggingface_hub==0.36.2' 'datasets==3.6.0' pyarrow pandas descript-audiotools"
echo "完成。工作目录: $WS"
