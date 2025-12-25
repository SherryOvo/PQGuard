#!/usr/bin/env python3
"""
Qwen2-VL 多模态模型 GUI 界面（使用 Streamlit）

功能：
- 文本对话
- 图像识别和问答
- 语音输入（上传音频文件）
- 调用训练好的模型

使用 Streamlit 构建 Web 界面（更简单稳定）
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import warnings

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import streamlit as st

# 语音相关
try:
    import whisper
    import soundfile as sf
    import librosa
    import numpy as np
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    whisper = None
    sf = None
    librosa = None

# 自动查找训练好的模型
_MODEL_DIR = Path("outputs/qwen2vl_beer_sft")
_BASE_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

if _MODEL_DIR.exists() and (_MODEL_DIR / "config.json").exists():
    MODEL_ID = str(_MODEL_DIR)
else:
    MODEL_ID = _BASE_MODEL_ID


@st.cache_resource
def load_model():
    """加载模型和处理器（使用 Streamlit 缓存）。"""
    print(f"加载模型: {MODEL_ID}...")
    
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


@st.cache_resource
def load_whisper_model():
    """加载 Whisper 模型（使用 Streamlit 缓存）。
    使用 'small' 模型以获得更好的识别准确率（比 'base' 更准确，比 'medium' 更快）。
    """
    if not SPEECH_AVAILABLE:
        return None
    try:
        # 尝试加载 'small' 模型（准确率更高）
        # 如果显存不足，可以回退到 'base'
        try:
            return whisper.load_model("small")
        except Exception as e1:
            print(f"加载 Whisper small 模型失败: {e1}，尝试 base 模型...")
            return whisper.load_model("base")
    except Exception as e:
        print(f"加载 Whisper 模型失败: {e}")
        return None


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """调整图像大小以节省显存（多模态模型图像处理显存占用大）。"""
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    # 保持宽高比缩放（限制最大边长为512像素）
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _convert_chat_history(chat_history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """将 Streamlit 对话历史格式转换为模型需要的格式。
    
    Args:
        chat_history: List of (role, content) tuples, e.g., [("user", "问题"), ("assistant", "回答")]
    
    Returns:
        List of message dicts, e.g., [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]
    """
    messages = []
    for role, content in chat_history:
        messages.append({"role": role, "content": content})
    return messages


def generate_reply(text: str, image: Optional[Image.Image] = None, max_new_tokens: int = 128, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
    """生成回复（支持图像输入和对话历史）。
    
    Args:
        text: 用户输入的文本
        image: 可选的图像输入
        max_new_tokens: 最大生成token数
        chat_history: 对话历史，格式为 List[Tuple[role, content]]
    """
    processor, model = load_model()
    
    # 清空显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 准备消息（使用对话历史或创建新对话）
    system_prompt = "你是一名精通精酿啤酒工艺、设备管理和异常诊断的中文智能助手，支持图像识别和语音交互。"
    
    if chat_history is not None and len(chat_history) > 0:
        # 转换对话历史格式
        history_messages = _convert_chat_history(chat_history)
        # 检查历史中是否已经有系统提示
        has_system = any(msg.get("role") == "system" for msg in history_messages)
        # 使用提供的对话历史，添加新用户消息
        if has_system:
            messages = history_messages + [{"role": "user", "content": text}]
        else:
            messages = [{"role": "system", "content": system_prompt}] + history_messages + [{"role": "user", "content": text}]
    else:
        # 创建新对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
    
    # 准备输入
    if image is not None:
        # 调整图像大小以节省显存（限制最大尺寸为512像素）
        image = resize_image(image, max_size=512)
        
        # 多模态输入（图像+文本）
        messages_text = processor.apply_chat_template(
            messages,
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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=messages_text,
            return_tensors="pt",
        )
    
    # 移动到模型设备
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 生成（优化参数以节省显存）
    with torch.no_grad():
        # 使用 torch.cuda.amp.autocast 进一步优化显存
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # 限制最大生成长度
                temperature=0.7,
                top_p=0.8,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    
    # 清空显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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


def _whisper_transcribe(whisper_model, audio_input, language="zh"):
    """统一的 Whisper 转录函数，使用优化的参数提高准确率。"""
    return whisper_model.transcribe(
        audio_input,
        language=language,
        fp16=False,
        verbose=False,
        beam_size=5,  # 使用 beam search 提高准确率
        best_of=5,    # 生成多个候选结果，选择最好的
        temperature=0.0,  # 使用贪心解码（确定性更高）
        condition_on_previous_text=False,  # 不依赖之前的文本，避免错误累积
    )


def transcribe_audio(audio_path: str) -> str:
    """将音频转换为文本（支持多种音频格式，包括 .m4a）。"""
    if not SPEECH_AVAILABLE:
        return "语音识别功能未安装，请安装：pip install openai-whisper soundfile librosa"
    
    if not os.path.exists(audio_path):
        return f"音频文件不存在: {audio_path}"
    
    # 加载 Whisper 模型（使用缓存）
    whisper_model = load_whisper_model()
    if whisper_model is None:
        return "无法加载 Whisper 模型，请检查安装"
    
    file_ext = Path(audio_path).suffix.lower()
    temp_wav_path = None
    
    try:
        # 策略1：优先直接使用 Whisper 处理（Whisper 内置了 ffmpeg 支持，可以直接处理多种格式）
        # 这对于 .m4a、.mp3 等格式特别有效
        try:
            result = _whisper_transcribe(whisper_model, audio_path, language="zh")
            text = result["text"].strip()
            if text:
                return text
            # 如果没有识别到文本，继续尝试其他方法
            print("Whisper 直接处理未识别到文本，尝试其他方法...")
        except Exception as e:
            error_msg = str(e).lower()
            # 如果错误与 ffmpeg 相关，记录并继续尝试其他方法
            if "ffmpeg" in error_msg or "no such file" in error_msg:
                print(f"Whisper 直接处理失败（可能缺少 ffmpeg）: {e}，尝试转换...")
            else:
                print(f"Whisper 直接处理失败: {e}，尝试转换...")
        
        # 策略2：使用 librosa 加载音频，然后转换（适用于 librosa 可以处理的格式）
        audio_array = None
        sample_rate = 16000
        
        try:
            # 抑制警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                # 使用 librosa 加载，支持多种格式
                # 对于 .m4a 文件，如果系统有 ffmpeg，librosa 应该能够加载
                audio_array, sample_rate = librosa.load(
                    audio_path,
                    sr=16000,  # 直接重采样到 16kHz
                    mono=True,  # 转换为单声道
                    res_type='kaiser_best'  # 高质量重采样
                )
        except Exception as e:
            error_msg = str(e)
            # 如果 librosa 也失败，尝试使用 ffmpeg 命令行工具转换
            print(f"librosa 加载失败: {error_msg}，尝试使用 ffmpeg 转换...")
            
            # 策略3：使用 ffmpeg 命令行工具转换为 WAV（最可靠的方法）
            import subprocess
            
            # 首先检查 ffmpeg 是否可用且能正常运行
            ffmpeg_available = False
            try:
                # 检查 ffmpeg 是否在 PATH 中
                which_result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True, timeout=5)
                if which_result.returncode == 0:
                    # 尝试运行 ffmpeg -version 检查是否能正常工作
                    version_result = subprocess.run(
                        ['ffmpeg', '-version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if version_result.returncode == 0:
                        ffmpeg_available = True
                    else:
                        # ffmpeg 存在但无法运行，可能是库依赖问题
                        error_output = version_result.stderr if version_result.stderr else version_result.stdout
                        if "cannot open shared object file" in error_output or "shared libraries" in error_output:
                            print(f"警告：ffmpeg 存在但缺少系统库依赖: {error_output[:200]}")
                            print("建议：如果遇到库缺失问题，请运行以下命令修复：")
                            print("  sudo ln -s /usr/lib/x86_64-linux-gnu/blas/libblas.so.3 /usr/lib/x86_64-linux-gnu/libblas.so.3")
                            print("  sudo ln -s /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3 /usr/lib/x86_64-linux-gnu/liblapack.so.3")
                            print("  sudo ldconfig")
                        ffmpeg_available = False
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                print(f"ffmpeg 检查失败: {e}")
                ffmpeg_available = False
            
            # 如果 ffmpeg 可用，尝试使用它转换
            if ffmpeg_available:
                try:
                    # 创建临时 WAV 文件
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_wav_path = temp_wav.name
                    temp_wav.close()
                    
                    # 使用 ffmpeg 转换为 WAV
                    cmd = [
                        'ffmpeg',
                        '-i', audio_path,  # 输入文件
                        '-ar', '16000',    # 采样率 16kHz
                        '-ac', '1',        # 单声道
                        '-y',              # 覆盖输出文件
                        '-loglevel', 'error',  # 只显示错误
                        temp_wav_path      # 输出文件
                    ]
                    
                    # 运行转换命令
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5分钟超时
                    )
                    
                    if result.returncode == 0 and os.path.exists(temp_wav_path):
                        # 转换成功，使用转换后的 WAV 文件
                        audio_path = temp_wav_path
                        file_ext = '.wav'
                        
                        # 尝试使用 Whisper 直接处理转换后的文件（最快）
                        try:
                            result = _whisper_transcribe(whisper_model, audio_path, language="zh")
                            text = result["text"].strip()
                            if text:
                                return text
                        except Exception:
                            pass
                        
                        # 如果 Whisper 直接处理失败，尝试用 librosa 加载
                        try:
                            audio_array, sample_rate = librosa.load(
                                audio_path,
                                sr=16000,
                                mono=True,
                                res_type='kaiser_best'
                            )
                        except Exception as e2:
                            return f"转换后的音频无法加载: {str(e2)}"
                    else:
                        error_output = result.stderr if result.stderr else result.stdout
                        # 如果 ffmpeg 转换失败，继续尝试其他方法
                        print(f"ffmpeg 转换失败: {error_output[:200]}")
                
                except subprocess.TimeoutExpired:
                    print("ffmpeg 转换超时，尝试其他方法...")
                except Exception as e:
                    print(f"ffmpeg 转换过程中发生错误: {e}，尝试其他方法...")
            
            # 如果 ffmpeg 不可用或转换失败，尝试让 Whisper 直接处理（Whisper 有内置 ffmpeg 支持）
            if audio_array is None:
                print("尝试让 Whisper 直接处理音频文件（使用内置 ffmpeg 支持）...")
                try:
                    result = _whisper_transcribe(whisper_model, audio_path, language="zh")
                    text = result["text"].strip()
                    if text:
                        return text
                    return "语音识别成功，但未识别到文本内容。请检查音频是否包含语音。"
                except Exception as e_whisper:
                    # Whisper 直接处理也失败
                    if ffmpeg_available:
                        return f"音频处理失败。librosa 错误: {error_msg}。Whisper 直接处理也失败: {str(e_whisper)}"
                    else:
                        return f"音频处理失败。librosa 错误: {error_msg}。Whisper 直接处理也失败: {str(e_whisper)}。提示：系统 ffmpeg 不可用，请检查系统依赖或尝试使用 WAV 格式文件。"
        
        # 如果成功加载了音频数组，使用它进行转录
        if audio_array is not None and len(audio_array) > 0:
            # 标准化音频数据（确保在合理范围内）
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # 策略4：直接传递 numpy array 给 Whisper
            try:
                result = _whisper_transcribe(whisper_model, audio_array, language="zh")
                text = result["text"].strip()
                if text:
                    return text
                return "语音识别成功，但未识别到文本内容。请检查音频是否包含语音。"
            except Exception as e:
                # 如果直接传递数组失败，保存为临时 WAV 文件再处理
                try:
                    if temp_wav_path is None:
                        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        temp_wav_path = temp_wav.name
                        temp_wav.close()
                    
                    # 保存为 WAV 格式
                    try:
                        sf.write(temp_wav_path, audio_array, sample_rate)
                    except Exception:
                        # 如果 soundfile 失败，使用 wave 模块
                        import wave
                        audio_int16 = (audio_array * 32767).astype(np.int16)
                        with wave.open(temp_wav_path, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sample_rate)
                            wf.writeframes(audio_int16.tobytes())
                    
                    # 使用文件路径进行转录
                    result = _whisper_transcribe(whisper_model, temp_wav_path, language="zh")
                    text = result["text"].strip()
                    if text:
                        return text
                    return "语音识别成功，但未识别到文本内容。请检查音频是否包含语音。"
                except Exception as e2:
                    return f"处理音频数组失败: {str(e)}。保存为文件后处理也失败: {str(e2)}"
        
        return "无法加载音频文件。请检查文件格式是否正确，或尝试转换为 WAV 格式。"
        
    except Exception as e:
        return f"语音识别过程中发生错误: {str(e)}"
    
    finally:
        # 清理临时文件
        if temp_wav_path and os.path.exists(temp_wav_path) and temp_wav_path != audio_path:
            try:
                os.unlink(temp_wav_path)
            except:
                pass


def main():
    """主函数。"""
    st.set_page_config(
        page_title="精酿啤酒智能助手",
        page_icon="🍺",
        layout="wide",
    )
    
    st.title("🍺 精酿啤酒智能助手")
    st.markdown("基于 Qwen2-VL 多模态大模型的精酿啤酒知识问答系统")
    
    # 显存优化提示
    with st.expander("💡 使用提示", expanded=False):
        st.markdown("""
        - **图像识别**：系统会自动调整图像大小（最大512像素）以优化显存使用
        - **显存不足**：如遇到显存错误，请尝试上传更小的图片或清空对话历史
        - **响应时间**：图像分析可能需要一些时间，请耐心等待
        - **语音识别**：推荐使用 WAV 格式音频文件，其他格式（MP3、M4A）可能需要在系统安装 ffmpeg
        """)
    
    # 侧边栏
    with st.sidebar:
        st.header("功能选择")
        mode = st.radio(
            "选择模式",
            ["📝 文本对话", "🖼️ 图像识别", "🎤 语音输入"],
        )
    
    # 初始化对话历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # 文本对话模式
    if mode == "📝 文本对话":
        st.header("文本对话")
        
        # 显示对话历史
        for i, (role, content) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.write(f"**你：** {content}")
            else:
                st.write(f"**助手：** {content}")
            st.divider()
        
        # 输入框
        user_input = st.text_input("输入问题", placeholder="例如：什么是浑浊IPA？如何判断发酵是否完成？")
        
        col1, col2 = st.columns([1, 10])
        with col1:
            submit = st.button("发送", type="primary")
        with col2:
            clear = st.button("清空历史")
        
        if clear:
            st.session_state.chat_history = []
            st.rerun()
        
        if submit and user_input:
            with st.spinner("正在思考..."):
                try:
                    # 清空显存缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 传递对话历史给模型
                    response = generate_reply(
                        user_input,
                        max_new_tokens=256,
                        chat_history=st.session_state.chat_history
                    )
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", response))
                    
                    # 清空显存缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    st.rerun()
                except torch.cuda.OutOfMemoryError as e:
                    st.error(f"显存不足：{str(e)}")
                    st.info("提示：请清空对话历史或重启界面释放显存")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    st.error(f"处理失败：{str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # 图像识别模式
    elif mode == "🖼️ 图像识别":
        st.header("图像识别")
        
        uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
        question = st.text_input("问题（可选）", placeholder="例如：请分析这张图片，判断酵母活性是否正常？")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # 提前调整图像大小以节省显存（限制为512像素）
            original_size = image.size
            image = resize_image(image, max_size=512)
            resized_size = image.size
            
            if original_size != resized_size:
                st.info(f"图像已从 {original_size} 调整为 {resized_size} 以优化显存使用")
            
            st.image(image, caption="上传的图片（已优化）", use_container_width=True)
            
            if st.button("分析图片", type="primary"):
                if not question:
                    question = "请分析这张图片并回答相关问题。"
                
                with st.spinner("正在分析图片（可能需要一些时间）..."):
                    try:
                        # 清空显存缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        response = generate_reply(question, image=image, max_new_tokens=128)
                        st.write(f"**问题：** {question}")
                        st.write(f"**回答：** {response}")
                        
                        # 再次清空显存缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError as e:
                        st.error(f"显存不足：{str(e)}")
                        st.info("提示：请尝试上传更小的图片，或重启界面释放显存")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        st.error(f"处理失败：{str(e)}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    
    # 语音输入模式
    elif mode == "🎤 语音输入":
        st.header("语音输入")
        
        # 初始化语音对话历史
        if "voice_chat_history" not in st.session_state:
            st.session_state.voice_chat_history = []
        
        # 显示对话历史
        if st.session_state.voice_chat_history:
            st.subheader("对话历史")
            for i, (role, content) in enumerate(st.session_state.voice_chat_history):
                if role == "user":
                    st.write(f"**你（语音）：** {content}")
                else:
                    st.write(f"**助手：** {content}")
                st.divider()
        
        if not SPEECH_AVAILABLE:
            st.warning("⚠️ 语音识别功能未安装，请安装：`pip install openai-whisper soundfile`")
        else:
            st.info("💡 **提示**：推荐使用 WAV 格式音频文件以获得最佳兼容性。支持格式：WAV（推荐）、MP3、M4A")
            uploaded_audio = st.file_uploader("上传音频文件", type=["wav", "mp3", "m4a"])
            
            # 清空历史按钮
            if st.button("清空对话历史", key="voice_clear"):
                st.session_state.voice_chat_history = []
                st.rerun()
            
            if uploaded_audio is not None:
                # 保存临时文件（保留原始扩展名）
                file_ext = Path(uploaded_audio.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_audio.read())
                    tmp_path = tmp_file.name
                
                st.audio(uploaded_audio)
                
                # 显示文件信息
                file_size = os.path.getsize(tmp_path) / (1024 * 1024)  # MB
                st.caption(f"文件大小: {file_size:.2f} MB | 格式: {file_ext}")
                
                if st.button("识别并回答", type="primary"):
                    with st.spinner("正在识别语音（可能需要一些时间）..."):
                        try:
                            text = transcribe_audio(tmp_path)
                            if text and not text.startswith("语音识别失败") and not text.startswith("语音识别功能未安装"):
                                st.success("✅ 识别成功！")
                                st.write(f"**识别到的文本：** {text}")
                                with st.spinner("正在生成回答..."):
                                    try:
                                        # 传递对话历史给模型，增加 token 限制以获得更完整的回答
                                        response = generate_reply(
                                            text,
                                            max_new_tokens=256,  # 增加到 256，与文本对话一致
                                            chat_history=st.session_state.voice_chat_history
                                        )
                                        
                                        # 保存到对话历史
                                        st.session_state.voice_chat_history.append(("user", text))
                                        st.session_state.voice_chat_history.append(("assistant", response))
                                        
                                        st.success("✅ 回答已生成！")
                                        st.rerun()  # 刷新页面以显示更新的对话历史
                                    except Exception as e:
                                        st.error(f"生成回答失败: {str(e)}")
                            else:
                                st.error("❌ " + text)
                                if "ffmpeg" in text.lower():
                                    st.info("💡 **解决方案：**\n"
                                            "1. 安装 ffmpeg：`sudo apt-get install ffmpeg`\n"
                                            "2. 或者上传 WAV 格式的音频文件（不需要 ffmpeg）")
                        except Exception as e:
                            st.error(f"处理失败: {str(e)}")
                        finally:
                            # 清理临时文件
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
    
    # 底部信息
    st.divider()
    st.markdown("**提示：** 模型已加载，可以直接使用各项功能。")


if __name__ == "__main__":
    main()

