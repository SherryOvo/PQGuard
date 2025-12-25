#!/usr/bin/env python3
"""
使用啤酒智慧精酿模拟数据对 Qwen1.5-7B-Chat 进行指令微调（单机版本）。

默认使用 data/beer_smart_brew_multimodal_kg_train_large.jsonl 中的 QA 样本（assistant_qa / assistant_multimodal_query），
通过 Qwen 的 chat_template 构造对话，进行监督微调。

示例运行（建议用你已有的虚拟环境解释器）：
    .venv/bin/python train_qwen_beer.py \
        --model-name-or-path Qwen/Qwen1.5-7B-Chat \
        --train-file data/beer_smart_brew_multimodal_kg_train_large.jsonl \
        --output-dir outputs/qwen_beer_sft

注意：完整微调 7B 模型需要 GPU 和较大显存，本脚本更多用于给出完整训练流程示例。
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainSample:
    # 这里存储的是纯 Python list，方便后续自定义 padding
    input_ids: List[int]
    attention_mask: List[int]


class BeerQADataset(Dataset):
    """从 JSONL 中抽取 assistant_qa / assistant_multimodal_query 样本。"""

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 2048,
        system_prompt: str = "你是一名啤酒智慧精酿领域的智能助手，擅长工艺决策、设备管理和异常诊断。",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.samples: List[TrainSample] = []

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

                context_ids: Optional[List[str]] = data.get("context_kg_ids")
                context_str = ""
                if context_ids:
                    context_str = "\n（内部关联知识节点ID，仅供模型检索使用）" + ", ".join(context_ids)

                # 构造对话轮次
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"{user_query}{context_str}",
                    },
                    {
                        "role": "assistant",
                        "content": expected_answer,
                    },
                ]

                # 使用 Qwen 官方 chat_template 构造训练文本
                enc = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                )
                ids_tensor = enc[0]
                input_ids = ids_tensor.tolist()
                attention_mask = [1] * len(input_ids)

                self.samples.append(
                    TrainSample(input_ids=input_ids, attention_mask=attention_mask)
                )

        if not self.samples:
            raise ValueError(f"在 {path} 中没有找到可用的 QA 样本（assistant_qa / assistant_multimodal_query）。")

        print(f"共加载训练样本: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "input_ids": s.input_ids,
            "attention_mask": s.attention_mask,
            # 简化起见，labels 与 input_ids 相同（即对整段文本做自回归学习）
            "labels": s.input_ids,
        }


def collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    """对变长序列进行 padding，并为 labels 使用 ignore_index。"""
    # 确保有 pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id
    ignore_index = -100

    batch_input_ids: List[List[int]] = [item["input_ids"] for item in batch]
    batch_attention: List[List[int]] = [item["attention_mask"] for item in batch]
    batch_labels: List[List[int]] = [item["labels"] for item in batch]

    max_len = max(len(ids) for ids in batch_input_ids)

    padded_input_ids = []
    padded_attention = []
    padded_labels = []

    for ids, attn, lbl in zip(batch_input_ids, batch_attention, batch_labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_attention.append(attn + [0] * pad_len)
        padded_labels.append(lbl + [ignore_index] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="使用啤酒智慧精酿 QA 数据微调 Qwen1.5-7B-Chat")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen1.5-7B-Chat",
        help="基础模型本地路径或 HuggingFace 模型 ID（默认从云端加载，节省本地磁盘空间）",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/beer_smart_brew_multimodal_kg_train_large.jsonl",
        help="训练数据 JSONL 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/qwen_beer_sft",
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
        default=1,
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
        default=2048,
        help="单条样本的最大 token 长度（超过会截断）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    train_dataset = BeerQADataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=1000,  # 增加保存间隔，减少磁盘写入
        save_total_limit=1,  # 只保留最新的1个checkpoint，节省空间
        save_only_model=True,  # 只保存模型权重，不保存optimizer/scheduler状态（节省约50%空间）
        bf16=torch.cuda.is_available(),
        fp16=False,
        evaluation_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=lambda batch: collate_fn(batch, tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"训练完成，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()


