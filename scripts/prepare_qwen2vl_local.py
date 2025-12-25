#!/usr/bin/env python3
"""
将 Qwen2-VL-7B-Instruct 多模态模型下载到本地项目目录。

Qwen2-VL 支持：
- 图像理解（图像描述、图像问答）
- 文本对话
- 多模态推理

下载位置：models/Qwen2-VL-7B-Instruct

用法：
    cd /root/private_data/Yijunhao
    .venv/bin/python prepare_qwen2vl_local.py
"""

import sys
from pathlib import Path

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch


def main():
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    local_dir = Path("models/Qwen2-VL-7B-Instruct")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载多模态模型 {model_id} 到本地目录: {local_dir}")
    print("注意：模型大小约 15GB，下载可能需要较长时间...")
    
    # 加载 processor（包含 tokenizer 和 image processor）
    print("加载 processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # 保存到本地
    print("保存 processor 和 model 到本地目录...")
    processor.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    
    print(f"完成：多模态模型已保存到 {local_dir}")


if __name__ == "__main__":
    main()


