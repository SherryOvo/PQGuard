#!/usr/bin/env python3
"""
模型环境变量配置（Python 版本）
设置 HuggingFace 模型缓存目录，指向已迁移的模型位置
"""

import os
import sys

# 设置 HuggingFace 相关的环境变量
HF_HOME = "/root/private_data/.cache/huggingface"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")

# 验证路径是否存在
if not os.path.isdir(HF_HOME):
    print(f"警告: HuggingFace 缓存目录不存在: {HF_HOME}", file=sys.stderr)

