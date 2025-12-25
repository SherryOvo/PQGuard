#!/bin/bash
# 模型环境变量配置
# 设置 HuggingFace 模型缓存目录，指向已迁移的模型位置

export HF_HOME="/root/private_data/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# 验证路径是否存在
if [ ! -d "$HF_HOME" ]; then
    echo "警告: HuggingFace 缓存目录不存在: $HF_HOME" >&2
fi

