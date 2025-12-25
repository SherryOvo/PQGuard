#!/usr/bin/env python3
"""
Qwen 1.5 7B Chat 交互式对话脚本（本地模型版）。

用法：
    cd /root/private_data/Yijunhao
    .venv/bin/python chat_qwen.py

特点：
    - 一次加载模型，循环多轮对话，不会每问一句就重新加载。
    - 默认使用项目内本地模型目录 models/Qwen1.5-7B-Chat（如需改为微调后的模型，可改成 outputs/qwen_beer_sft）。
    - 输入 exit / quit 退出。
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# 自动查找最新的 checkpoint，如果不存在或不完整则使用基础模型
# 注意：tokenizer 总是从基础模型加载（因为 checkpoint 里通常没有 tokenizer 文件）

# 优先使用本地模型，如果不存在则从云端加载
_LOCAL_MODEL_PATH = Path("models/Qwen1.5-7B-Chat")
if _LOCAL_MODEL_PATH.exists() and (_LOCAL_MODEL_PATH / "config.json").exists():
    BASE_MODEL_ID = "models/Qwen1.5-7B-Chat"  # 使用本地模型
else:
    BASE_MODEL_ID = "Qwen/Qwen1.5-7B-Chat"  # 从云端加载（会缓存到 ~/.cache/huggingface/）


def is_checkpoint_complete(checkpoint_path: Path) -> bool:
    """检查 checkpoint 是否完整（是否有所有模型文件）。"""
    if not (checkpoint_path / "config.json").exists():
        return False
    
    # 检查是否有 safetensors 文件
    safetensors_files = list(checkpoint_path.glob("model-*.safetensors"))
    if not safetensors_files:
        # 如果没有 safetensors，检查是否有 pytorch_model.bin
        if (checkpoint_path / "pytorch_model.bin").exists():
            return True
        return False
    
    # 如果有 safetensors，检查是否完整（通过尝试加载 config 来判断需要多少个分片）
    try:
        import json
        with open(checkpoint_path / "config.json", "r") as f:
            config = json.load(f)
        # 简单检查：如果文件数量太少（比如只有1-2个但模型很大），可能不完整
        # 这里我们尝试加载，如果失败就说明不完整
        return True  # 先返回 True，让加载时再判断
    except:
        return False


# 优先使用训练好的模型（30k数据集训练的版本）
_checkpoint_dir_30k = Path("outputs/qwen_beer_sft_30k")
_checkpoint_dir = Path("outputs/qwen_beer_sft")
MODEL_ID = BASE_MODEL_ID  # 默认使用基础模型

# 优先检查30k数据集训练的模型
if _checkpoint_dir_30k.exists():
    # 先检查根目录是否有完整模型
    if (_checkpoint_dir_30k / "config.json").exists() and is_checkpoint_complete(_checkpoint_dir_30k):
        MODEL_ID = "outputs/qwen_beer_sft_30k"
    else:
        # 查找最新的 checkpoint 目录
        checkpoints = sorted([d for d in _checkpoint_dir_30k.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            if is_checkpoint_complete(latest_checkpoint):
                MODEL_ID = str(latest_checkpoint)
elif _checkpoint_dir.exists():
    # 查找最新的 checkpoint 目录
    checkpoints = sorted([d for d in _checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        if is_checkpoint_complete(latest_checkpoint):
            MODEL_ID = str(latest_checkpoint)
        else:
            print(f"警告：checkpoint {latest_checkpoint} 可能不完整，将使用基础模型。")
    else:
        # 如果 outputs/qwen_beer_sft 根目录有模型文件，直接用
        if (_checkpoint_dir / "config.json").exists() and is_checkpoint_complete(_checkpoint_dir):
            MODEL_ID = "outputs/qwen_beer_sft"


def load_model():
    global MODEL_ID
    model_path = MODEL_ID
    
    # 优先从训练好的模型目录加载 tokenizer（如果存在），否则从基础模型加载
    tokenizer_path = model_path if (Path(model_path) / "tokenizer.json").exists() else BASE_MODEL_ID
    print(f"加载 tokenizer：{tokenizer_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    
    # 尝试加载模型，如果失败则回退到基础模型
    print(f"尝试加载模型：{model_path} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        )
        if model_path != BASE_MODEL_ID:
            print(f"✓ 成功加载微调后的模型：{model_path}")
    except Exception as e:
        print(f"✗ 加载 {model_path} 失败：{e}")
        if model_path != BASE_MODEL_ID:
            print(f"回退到基础模型：{BASE_MODEL_ID}")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            )
        else:
            raise
    
    model.eval()
    return tokenizer, model


def generate_reply(
    tokenizer,
    model,
    history: List[Dict[str, Any]],
    max_new_tokens: int = 512,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.8,
        )
    response_ids = outputs[0][input_ids.shape[-1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def main():
    tokenizer, model = load_model()

    system_prompt = "你是一名精通精酿啤酒工艺、设备管理和异常诊断的中文智能助手。"
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    print("进入 Qwen 交互模式，输入内容后回车即可对话。输入 `exit` 或 `quit` 退出。")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话。")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        reply = generate_reply(tokenizer, model, history)
        history.append({"role": "assistant", "content": reply})

        print(f"Qwen：{reply}")


if __name__ == "__main__":
    main()


