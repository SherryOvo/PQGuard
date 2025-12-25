#!/usr/bin/env python3
"""
Qwen2-VL 多模态交互式对话脚本（支持图像+文本+语音）。

功能：
- 文本对话
- 图像理解（上传图片进行问答）
- 语音输入（语音转文本后对话）
- 语音输出（文本转语音）

用法：
    cd /root/private_data/Yijunhao
    .venv/bin/python chat_qwen2vl_multimodal.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import io

# 语音相关（可选，如果安装了相关库）
try:
    import whisper
    import soundfile as sf
    import edge_tts
    import asyncio
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("警告：语音功能未安装，仅支持文本和图像输入。安装：pip install openai-whisper edge-tts soundfile")


# 自动查找训练好的模型
_MODEL_DIR = Path("outputs/qwen2vl_beer_sft")
_BASE_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

if _MODEL_DIR.exists() and (_MODEL_DIR / "config.json").exists():
    MODEL_ID = str(_MODEL_DIR)
else:
    MODEL_ID = _BASE_MODEL_ID


def load_model():
    print(f"加载多模态模型：{MODEL_ID} ...")
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
    )
    model.eval()
    
    return processor, model


def process_voice_input(audio_path: str) -> str:
    """使用 Whisper 将语音转换为文本。"""
    if not SPEECH_AVAILABLE:
        return ""
    
    try:
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(audio_path, language="zh")
        return result["text"]
    except Exception as e:
        print(f"语音识别失败: {e}")
        return ""


async def _text_to_speech_async(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    """使用 edge-tts 异步合成语音。"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def text_to_speech(text: str, output_path: str = "output_audio.wav"):
    """将文本转换为语音（使用 edge-tts）。"""
    if not SPEECH_AVAILABLE:
        print("语音合成功能未安装")
        return None
    
    try:
        # 使用中文语音（zh-CN-XiaoxiaoNeural 是中文女声）
        # 可选语音：zh-CN-XiaoxiaoNeural, zh-CN-YunxiNeural, zh-CN-YunyangNeural
        asyncio.run(_text_to_speech_async(text, output_path, voice="zh-CN-XiaoxiaoNeural"))
        print(f"语音已保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"语音合成失败: {e}")
        return None


def load_image(image_path: str) -> Optional[Image.Image]:
    """加载图像文件。"""
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None


def generate_reply(
    processor,
    model,
    history: List[Dict[str, Any]],
    image: Optional[Image.Image] = None,
    max_new_tokens: int = 512,
) -> str:
    """生成回复（支持图像输入）。"""
    # 准备输入
    if image is not None:
        # 多模态输入（图像+文本）
        messages_text = processor.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=messages_text,
            images=[image],
            return_tensors="pt",
        )
    else:
        # 纯文本输入
        messages_text = processor.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=messages_text,
            return_tensors="pt",
        )
    
    # 移动到模型设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.8,
        )
    
    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return response


def main():
    processor, model = load_model()
    
    system_prompt = "你是一名精通精酿啤酒工艺、设备管理和异常诊断的中文智能助手，支持图像识别和语音交互。"
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    
    print("=" * 60)
    print("Qwen2-VL 多模态交互模式")
    print("=" * 60)
    print("支持功能：")
    print("  1. 文本对话：直接输入问题")
    print("  2. 图像查询：输入 'image:图片路径' 进行图像问答")
    print("  3. 语音输入：输入 'voice:音频文件路径' 进行语音查询")
    print("  4. 语音输出：输入 'tts:文本内容' 将文本转为语音")
    print("  5. 退出：输入 'exit' 或 'quit'")
    print("=" * 60)
    
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
        
        # 处理图像输入
        image = None
        if user_input.startswith("image:"):
            image_path = user_input[6:].strip()
            image = load_image(image_path)
            if image is None:
                continue
            user_query = "请分析这张图片并回答相关问题。"
        # 处理语音输入
        elif user_input.startswith("voice:") and SPEECH_AVAILABLE:
            audio_path = user_input[6:].strip()
            user_query = process_voice_input(audio_path)
            if not user_query:
                print("语音识别失败，请重试。")
                continue
            print(f"识别到的文本: {user_query}")
        # 处理语音输出
        elif user_input.startswith("tts:"):
            text_to_speech(user_input[4:].strip())
            continue
        else:
            user_query = user_input
        
        # 添加到历史
        history.append({"role": "user", "content": user_query})
        
        # 生成回复
        reply = generate_reply(processor, model, history, image=image)
        history.append({"role": "assistant", "content": reply})
        
        print(f"Qwen：{reply}")


if __name__ == "__main__":
    main()

