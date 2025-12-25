#!/usr/bin/env python3
"""
测试训练好的 Qwen2-VL 多模态模型。

快速测试脚本，用于验证模型是否正常工作。
"""

import sys
from pathlib import Path
from PIL import Image

# 设置模型环境变量（在导入 transformers 之前）
sys.path.insert(0, str(Path(__file__).parent))
import env_model

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


def test_model():
    """测试训练好的模型。"""
    model_dir = Path("outputs/qwen2vl_beer_sft")
    base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    # 确定使用的模型路径
    if model_dir.exists() and (model_dir / "config.json").exists():
        model_id = str(model_dir)
        print(f"✓ 找到训练好的模型: {model_id}")
    else:
        model_id = base_model_id
        print(f"⚠ 未找到训练好的模型，使用基础模型: {model_id}")
    
    print(f"\n正在加载模型...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        )
        model.eval()
        print(f"✓ 模型加载成功")
        print(f"  设备: {next(model.parameters()).device}")
        print(f"  数据类型: {next(model.parameters()).dtype}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 测试1：文本问答
    print("\n" + "=" * 60)
    print("测试 1: 文本问答")
    print("=" * 60)
    
    test_questions = [
        "什么是浑浊IPA？",
        "如何判断发酵是否完成？",
        "糖化温度应该设定多少度？",
    ]
    
    system_prompt = "你是一名精通精酿啤酒工艺、设备管理和异常诊断的中文智能助手。"
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = processor(text=text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.8,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            print(f"回答: {response[:200]}..." if len(response) > 200 else f"回答: {response}")
            print("✓ 测试通过")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试2：多模态（图像+文本）- 如果有图像的话
    print("\n" + "=" * 60)
    print("测试 2: 多模态（图像+文本）")
    print("=" * 60)
    
    # 创建一个测试图像
    test_image = Image.new('RGB', (224, 224), color='white')
    
    try:
        print("\n问题: 请描述这张图像。")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请描述这张图像。"},
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = processor(
            text=text,
            images=[test_image],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.8,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        print(f"回答: {response[:200]}..." if len(response) > 200 else f"回答: {response}")
        print("✓ 多模态测试通过")
        
    except Exception as e:
        print(f"❌ 多模态测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n要使用交互式对话，请运行：")
    print("  .venv/bin/python scripts/chat_qwen2vl_multimodal.py")
    print("\n或者在 Python 中使用：")
    print("  import sys")
    print("  sys.path.insert(0, 'scripts')")
    print("  from chat_qwen2vl_multimodal import load_model, generate_reply")
    print("  processor, model = load_model()")
    print("  # 然后使用 generate_reply 函数")
    
    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)

