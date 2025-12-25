#!/usr/bin/env python3
"""
Qwen2-VL 多模态模型 GUI 界面

功能：
- 文本对话
- 图像识别和问答
- 语音输入（麦克风录音）
- 语音输出（TTS）

使用 Gradio 构建 Web 界面
"""

import sys
import os

# 在导入 Gradio 之前设置环境变量，避免启动检查问题
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_IS_COLAB_HOST"] = "False"
os.environ["GRADIO_IS_SPACES"] = "False"

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import io

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import gradio as gr

# 语音相关
try:
    import whisper
    import soundfile as sf
    import edge_tts
    import asyncio
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("警告：语音识别功能未安装，某些功能可能不可用")

# 麦克风录音（可选，服务器环境可能不需要）
try:
    import pyaudio
    import wave
    MICROPHONE_AVAILABLE = True
except ImportError:
    MICROPHONE_AVAILABLE = False
    print("提示：pyaudio未安装，无法使用麦克风录音，但仍可通过上传音频文件使用语音功能")


# 自动查找训练好的模型
_MODEL_DIR = Path("outputs/qwen2vl_beer_sft")
_BASE_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

if _MODEL_DIR.exists() and (_MODEL_DIR / "config.json").exists():
    MODEL_ID = str(_MODEL_DIR)
    print(f"✓ 使用训练好的模型: {MODEL_ID}")
else:
    MODEL_ID = _BASE_MODEL_ID
    print(f"⚠ 使用基础模型: {MODEL_ID}")


# 全局变量存储模型和处理器
processor = None
model = None
chat_history: List[Dict[str, Any]] = []


def load_model():
    """加载模型和处理器。"""
    global processor, model
    
    if processor is not None and model is not None:
        return processor, model
    
    print(f"正在加载模型: {MODEL_ID}...")
    
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
    
    print("✓ 模型加载完成")
    return processor, model


def generate_reply(
    text: str,
    image: Optional[Image.Image] = None,
    max_new_tokens: int = 512,
) -> str:
    """生成回复（支持图像输入）。"""
    global processor, model, chat_history
    
    if processor is None or model is None:
        processor, model = load_model()
    
    # 准备消息
    if not chat_history or chat_history[0].get("role") != "system":
        system_prompt = "你是一名精通精酿啤酒工艺、设备管理和异常诊断的中文智能助手，支持图像识别和语音交互。"
        chat_history = [{"role": "system", "content": system_prompt}]
    
    # 添加用户消息
    user_message = {"role": "user", "content": text}
    chat_history.append(user_message)
    
    # 准备输入
    if image is not None:
        # 多模态输入（图像+文本）
        messages_text = processor.apply_chat_template(
            chat_history,
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
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=messages_text,
            return_tensors="pt",
        )
    
    # 移动到模型设备
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
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
    
    # 添加助手回复到历史
    chat_history.append({"role": "assistant", "content": response})
    
    return response


def text_chat(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """文本对话处理。"""
    if not message.strip():
        return "", history
    
    try:
        response = generate_reply(message)
        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"错误: {str(e)}"
        history.append((message, error_msg))
        return "", history


def image_chat(message: str, image: Image.Image, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """图像+文本对话处理。"""
    if image is None:
        return "请先上传图片", history
    
    if not message.strip():
        message = "请分析这张图片并回答相关问题。"
    
    try:
        response = generate_reply(message, image=image)
        history.append((f"[图像] {message}", response))
        return "", history
    except Exception as e:
        error_msg = f"错误: {str(e)}"
        history.append((f"[图像] {message}", error_msg))
        return "", history


# Gradio 的 Audio 组件自带录音功能，不需要单独的录音函数


def transcribe_audio(audio_path: str) -> str:
    """将音频转换为文本。"""
    if not SPEECH_AVAILABLE:
        return "语音识别功能未安装"
    
    if audio_path is None:
        return "未提供音频文件"
    
    try:
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(audio_path, language="zh")
        return result["text"]
    except Exception as e:
        return f"语音识别失败: {str(e)}"


def voice_chat(audio_input, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """语音对话处理。"""
    if audio_input is None:
        return "请先录制或上传音频", history
    
    # Gradio Audio 组件返回的可能是元组 (filepath, sample_rate) 或只是 filepath
    if isinstance(audio_input, tuple):
        audio_path = audio_input[0]
    else:
        audio_path = audio_input
    
    if not audio_path or not os.path.exists(audio_path):
        return "音频文件不存在", history
    
    try:
        # 语音转文本
        text = transcribe_audio(audio_path)
        if not text or text.startswith("语音识别失败") or text.startswith("未提供"):
            return text, history
        
        # 生成回复
        response = generate_reply(text)
        history.append((f"[语音] {text}", response))
        return "", history
    except Exception as e:
        error_msg = f"错误: {str(e)}"
        history.append(("语音输入", error_msg))
        return "", history


def clear_history():
    """清空对话历史。"""
    global chat_history
    chat_history = []
    return []


def text_to_speech(text: str) -> Optional[str]:
    """将文本转换为语音。"""
    if not SPEECH_AVAILABLE:
        return None
    
    try:
        import edge_tts
        import asyncio
        
        async def _tts_async():
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            await communicate.save(temp_file.name)
            return temp_file.name
        
        audio_path = asyncio.run(_tts_async())
        return audio_path
    except Exception as e:
        print(f"语音合成失败: {e}")
        return None


# 创建 Gradio 界面
def create_interface():
    """创建 Gradio 界面。"""
    
    # 预加载模型
    print("正在预加载模型...")
    load_model()
    
    # 创建 Gradio 界面（不使用 theme 参数以兼容旧版本）
    with gr.Blocks(title="精酿啤酒智能助手 - Qwen2-VL") as demo:
        gr.Markdown("""
        # 🍺 精酿啤酒智能助手
        
        基于 Qwen2-VL 多模态大模型的精酿啤酒知识问答系统
        
        **功能：**
        - 📝 文本对话：直接输入问题
        - 🖼️ 图像识别：上传图片进行问答
        - 🎤 语音输入：使用麦克风录音或上传音频文件
        - 🔊 语音输出：将回复转换为语音
        """)
        
        with gr.Tabs():
            # Tab 1: 文本对话
            with gr.Tab("📝 文本对话"):
                text_chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                )
                with gr.Row():
                    text_input = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：什么是浑浊IPA？如何判断发酵是否完成？",
                        scale=4,
                    )
                    text_submit = gr.Button("发送", variant="primary", scale=1)
                
                text_clear = gr.Button("清空历史", variant="secondary")
                
                text_submit.click(
                    text_chat,
                    inputs=[text_input, text_chatbot],
                    outputs=[text_input, text_chatbot],
                )
                text_input.submit(
                    text_chat,
                    inputs=[text_input, text_chatbot],
                    outputs=[text_input, text_chatbot],
                )
                text_clear.click(clear_history, outputs=[text_chatbot])
            
            # Tab 2: 图像识别
            with gr.Tab("🖼️ 图像识别"):
                image_chatbot = gr.Chatbot(
                    label="对话历史",
                    height=400,
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="上传图片",
                            type="pil",
                        )
                    with gr.Column(scale=2):
                        image_text_input = gr.Textbox(
                            label="问题（可选）",
                            placeholder="例如：请分析这张图片，判断酵母活性是否正常？",
                            lines=3,
                        )
                        image_submit = gr.Button("发送", variant="primary")
                
                image_clear = gr.Button("清空历史", variant="secondary")
                
                image_submit.click(
                    image_chat,
                    inputs=[image_text_input, image_input, image_chatbot],
                    outputs=[image_text_input, image_chatbot],
                )
                image_clear.click(clear_history, outputs=[image_chatbot])
            
            # Tab 3: 语音对话
            with gr.Tab("🎤 语音对话"):
                voice_chatbot = gr.Chatbot(
                    label="对话历史",
                    height=400,
                )
                with gr.Row():
                    with gr.Column():
                        if SPEECH_AVAILABLE:
                            gr.Markdown("**方式1：使用麦克风录音**")
                            voice_audio_input = gr.Audio(
                                label="录音或上传音频文件",
                                type="filepath",
                                sources=["microphone", "upload"],
                            )
                        else:
                            gr.Markdown("⚠️ 语音功能未安装，请安装：`pip install openai-whisper pyaudio soundfile`")
                            voice_audio_input = gr.Audio(
                                label="上传音频文件",
                                type="filepath",
                                sources=["upload"],
                            )
                        voice_submit = gr.Button("发送", variant="primary")
                
                voice_clear = gr.Button("清空历史", variant="secondary")
                
                voice_submit.click(
                    voice_chat,
                    inputs=[voice_audio_input, voice_chatbot],
                    outputs=[voice_audio_input, voice_chatbot],
                )
                voice_clear.click(clear_history, outputs=[voice_chatbot])
            
            # Tab 4: 语音输出
            with gr.Tab("🔊 语音输出"):
                tts_input = gr.Textbox(
                    label="输入文本",
                    placeholder="输入要转换为语音的文本",
                    lines=5,
                )
                tts_output = gr.Audio(label="生成的语音")
                tts_submit = gr.Button("生成语音", variant="primary")
                
                def generate_tts(text):
                    if not text.strip():
                        return None
                    audio_path = text_to_speech(text)
                    return audio_path if audio_path else None
                
                tts_submit.click(
                    generate_tts,
                    inputs=[tts_input],
                    outputs=[tts_output],
                )
        
        gr.Markdown("""
        ---
        **使用提示：**
        - 文本对话：直接输入问题即可
        - 图像识别：上传图片后可以输入问题，也可以直接发送让模型自动分析
        - 语音输入：点击录音按钮或上传音频文件，系统会自动识别并回答
        - 语音输出：输入文本后点击生成语音，可以听到回复
        """)
    
    return demo


def get_server_ip():
    """获取服务器IP地址。"""
    import socket
    try:
        # 连接到一个外部地址来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    """主函数。"""
    print("=" * 60)
    print("启动 Qwen2-VL 多模态模型 GUI 界面")
    print("=" * 60)
    
    demo = create_interface()
    
    # 获取服务器IP
    server_ip = get_server_ip()
    server_port = 7860
    
    print("\n" + "=" * 60)
    print("服务器配置信息：")
    print(f"  服务器IP: {server_ip}")
    print(f"  端口: {server_port}")
    print(f"  访问地址: http://{server_ip}:{server_port}")
    print("=" * 60)
    print("\n注意：")
    print("  1. 如果是远程服务器，请使用上述IP地址访问")
    print("  2. 确保防火墙已开放端口 7860")
    print("  3. 如果无法访问，请检查网络配置")
    print("\n正在启动界面...\n")
    
    # 设置环境变量以禁用启动检查
    import os
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = str(server_port)
    
    # 启动界面
    print("启动 Gradio 界面...\n")
    print("=" * 60)
    print("重要提示：")
    print("1. 如果启动失败，请使用 SSH 端口转发访问")
    print(f"   在本地运行: ssh -L 7860:localhost:7860 root@{server_ip}")
    print("   然后访问: http://localhost:7860")
    print("2. 或者使用公共链接（share=True）")
    print("=" * 60)
    print()
    
    # 尝试使用 share=True 启动（创建公共链接）
    try:
        demo.launch(
            share=True,  # 创建公共链接，绕过本地检查
            server_port=server_port,
            show_error=True,
            inbrowser=False,
        )
    except Exception as e:
        print(f"\n使用 share=True 启动失败: {e}")
        print("\n尝试使用本地服务器模式（需要 SSH 端口转发）...")
        print(f"SSH 命令: ssh -L {server_port}:localhost:{server_port} root@{server_ip}\n")
        # 回退到本地模式
        demo.launch(
            server_name="127.0.0.1",
            server_port=server_port,
            share=False,
            show_error=True,
            inbrowser=False,
        )


if __name__ == "__main__":
    main()

