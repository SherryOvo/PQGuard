#!/usr/bin/env python3
"""
使用啤酒智慧精酿多模态数据对 Qwen2-VL-7B-Instruct 进行指令微调。

支持：
- 文本问答
- 图像+文本多模态查询
- 语音转文本+文本查询

示例运行：
    .venv/bin/python train_qwen2vl_beer.py \
        --model-name-or-path Qwen/Qwen2-VL-7B-Instruct \
        --train-file data/beer_smart_brew_multimodal_kg_train_30k.jsonl \
        --output-dir outputs/qwen2vl_beer_sft
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import io
import base64

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
)


@dataclass
class MultimodalSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None  # Qwen2-VL 需要的图像网格信息


class BeerMultimodalDataset(Dataset):
    """从 JSONL 中加载多模态训练样本（文本QA + 图像+文本 + 语音+文本）。"""

    def __init__(
        self,
        path: str,
        processor,
        max_length: int = 2048,
        system_prompt: str = "你是一名啤酒智慧精酿领域的智能助手，擅长工艺决策、设备管理和异常诊断。",
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.samples: List[MultimodalSample] = []

        self._load(path)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"找不到训练文件: {p}")

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rt = data.get("record_type")
                if rt not in {"assistant_qa", "assistant_multimodal_query"}:
                    continue

                user_query = data.get("user_query") or data.get("user_query_text")
                expected_answer = data.get("expected_answer")
                if not user_query or not expected_answer:
                    continue

                # 构造对话消息
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": expected_answer},
                ]

                # 处理图像
                # Qwen2-VL 需要所有样本都有图像输入，即使是纯文本样本也需要占位图像
                is_multimodal = (rt == "assistant_multimodal_query" and data.get("modality") == "image+text")
                
                # 为所有样本创建图像（多模态样本使用占位图像，文本样本也使用占位图像）
                # 实际应用中，多模态样本应加载真实图像：Image.open(data.get("image_ref"))
                placeholder_image = Image.new('RGB', (224, 224), color='white')
                pixel_values = placeholder_image

                # 使用 processor 处理文本和图像
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                
                # 所有样本都传入图像（文本样本使用占位图像）
                inputs = self.processor(
                    text=text,
                    images=[pixel_values],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                
                # 提取处理后的图像特征和 grid_thw
                processed_pixel_values = inputs.get("pixel_values")
                image_grid_thw = inputs.get("image_grid_thw")  # Qwen2-VL 需要的图像网格信息
                
                if processed_pixel_values is not None and len(processed_pixel_values.shape) > 1:
                    # 去掉批次维度 [1, C, H, W] -> [C, H, W]
                    processed_pixel_values = processed_pixel_values.squeeze(0)
                else:
                    # 如果 processor 没有返回 pixel_values，创建一个默认张量
                    processed_pixel_values = torch.zeros(3, 224, 224)
                
                # 处理 image_grid_thw
                # Qwen2-VL 的 image_grid_thw 格式是 [t, h, w]，表示时间、高度、宽度的网格数
                if image_grid_thw is not None:
                    # image_grid_thw 通常是 [1, 3] 或 [3] 形状
                    if len(image_grid_thw.shape) > 1:
                        image_grid_thw = image_grid_thw.squeeze(0)
                else:
                    # 如果 processor 没有返回，根据图像大小计算
                    # Qwen2-VL 的 patch size 通常是 14x14
                    # 对于 224x224 图像：grid_h = 224/14 = 16, grid_w = 224/14 = 16
                    if processed_pixel_values is not None and len(processed_pixel_values.shape) >= 2:
                        # 获取图像的高度和宽度
                        img_h, img_w = processed_pixel_values.shape[-2], processed_pixel_values.shape[-1]
                        patch_size = 14  # Qwen2-VL 的默认 patch size
                        grid_h = img_h // patch_size
                        grid_w = img_w // patch_size
                        image_grid_thw = torch.tensor([1, grid_h, grid_w], dtype=torch.long)
                    else:
                        # 默认值（对于 224x224 图像）
                        image_grid_thw = torch.tensor([1, 16, 16], dtype=torch.long)

                input_ids = inputs["input_ids"][0]
                attention_mask = inputs["attention_mask"][0] if "attention_mask" in inputs else torch.ones_like(input_ids)

                self.samples.append(
                    MultimodalSample(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=processed_pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                )

        if not self.samples:
            raise ValueError(f"在 {path} 中没有找到可用的训练样本。")

        print(f"共加载训练样本: {len(self.samples)}")
        print(f"  - 文本样本: {len([s for s in self.samples if s.pixel_values is None])}")
        print(f"  - 多模态样本: {len([s for s in self.samples if s.pixel_values is not None])}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        result = {
            "input_ids": s.input_ids,
            "attention_mask": s.attention_mask,
            "labels": s.input_ids.clone(),
        }
        # 所有样本都有 pixel_values 和 image_grid_thw（因为我们在数据加载时为所有样本创建了占位图像）
        if s.pixel_values is not None:
            result["pixel_values"] = s.pixel_values
        if s.image_grid_thw is not None:
            result["image_grid_thw"] = s.image_grid_thw
        return result


def collate_fn(batch: List[Dict[str, Any]], processor) -> Dict[str, Any]:
    """对批次进行padding。"""
    input_ids = [item["input_ids"].tolist() for item in batch]
    attention_masks = [item["attention_mask"].tolist() for item in batch]
    labels = [item["labels"].tolist() for item in batch]

    # 找到最大长度
    max_len = max(len(ids) for ids in input_ids)
    pad_id = processor.tokenizer.pad_token_id
    ignore_index = -100

    # Padding
    padded_input_ids = []
    padded_attention = []
    padded_labels = []

    for ids, attn, lbl in zip(input_ids, attention_masks, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_attention.append(attn + [0] * pad_len)
        padded_labels.append(lbl + [ignore_index] * pad_len)

    result = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }

    # 处理图像（所有样本都有 pixel_values 和 image_grid_thw）
    pixel_values_list = [item.get("pixel_values") for item in batch]
    image_grid_thw_list = [item.get("image_grid_thw") for item in batch]
    
    # 确保所有样本都有 pixel_values（不应该有 None，但为了安全起见检查一下）
    pixel_values_list = [pv if pv is not None else torch.zeros(3, 224, 224) for pv in pixel_values_list]
    result["pixel_values"] = torch.stack(pixel_values_list)
    
    # 处理 image_grid_thw（堆叠成批次）
    if all(igthw is not None for igthw in image_grid_thw_list):
        # 所有样本都有 image_grid_thw，直接堆叠
        result["image_grid_thw"] = torch.stack(image_grid_thw_list)
    else:
        # 如果有缺失，使用默认值
        default_grid_thw = torch.tensor([1, 14, 14], dtype=torch.long)
        image_grid_thw_list = [igthw if igthw is not None else default_grid_thw for igthw in image_grid_thw_list]
        result["image_grid_thw"] = torch.stack(image_grid_thw_list)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="使用啤酒智慧精酿多模态数据微调 Qwen2-VL-7B-Instruct")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="基础模型路径或 HuggingFace 模型 ID",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/beer_smart_brew_multimodal_kg_train_30k.jsonl",
        help="训练数据 JSONL 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/qwen2vl_beer_sft",
        help="保存微调后模型的目录",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,  # 多模态模型显存占用更大，batch size 设为 1
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,  # 多模态模型显存占用大，进一步减少序列长度
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="使用梯度检查点节省显存",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,  # 减少数据加载器工作进程，节省显存
    )
    parser.add_argument(
        "--use-8bit-optimizer",
        action="store_true",
        default=True,
        help="使用 8-bit 优化器节省显存",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="优化器类型",
    )
    return parser.parse_args()


def main():
    import os
    args = parse_args()
    
    # 设置显存优化环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # 启用梯度检查点以节省显存
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()

    train_dataset = BeerMultimodalDataset(
        path=args.train_file,
        processor=processor,
        max_length=args.max_length,
    )

    # 配置优化器（使用 8-bit 优化器节省显存）
    optim_args = {}
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            # 使用 8-bit AdamW 优化器
            optim_args["optim"] = "adamw_bnb_8bit"
            print("使用 8-bit 优化器以节省显存")
        except ImportError:
            print("警告：bitsandbytes 未安装，使用标准优化器")
            optim_args["optim"] = args.optim
    else:
        optim_args["optim"] = args.optim
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        save_only_model=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        evaluation_strategy="no",
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,  # 启用梯度检查点
        dataloader_num_workers=args.dataloader_num_workers,  # 减少数据加载器工作进程
        dataloader_pin_memory=False,  # 禁用 pin_memory 以节省显存
        remove_unused_columns=False,  # 保留所有列（包括 pixel_values 和 image_grid_thw）
        max_grad_norm=1.0,  # 梯度裁剪
        **optim_args,  # 添加优化器配置
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=lambda batch: collate_fn(batch, processor),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"训练完成，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

