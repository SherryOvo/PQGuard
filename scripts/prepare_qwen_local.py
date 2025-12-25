#!/usr/bin/env python3
"""
将 Qwen1.5-7B-Chat 模型完整下载到当前项目目录下，方便后续离线使用 / 训练。

下载位置默认：models/Qwen1.5-7B-Chat

用法示例：
    cd /root/private_data/Yijunhao
    .venv/bin/python prepare_qwen_local.py

注意：
  - 首次运行需要从 HuggingFace 拉取约十几 GB 权重，请确保网络畅通、磁盘空间充足。
  - 如果你已经有本地缓存，脚本会复用缓存并保存到指定目录。
"""

import argparse
import sys
from pathlib import Path

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="下载 Qwen1.5-7B-Chat 到本地项目目录")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen1.5-7B-Chat",
        help="远程模型名称（HuggingFace Hub）",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="models/Qwen1.5-7B-Chat",
        help="本地保存目录（相对当前项目路径）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_path = Path(args.local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"将模型 {args.model_id} 下载/保存到本地目录: {local_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    print("开始保存 tokenizer 和 model 到本地目录...")
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    print(f"完成：模型已保存到 {local_path}")


if __name__ == "__main__":
    main()


