# 模型路径迁移说明

## 概述

Qwen 模型已从 `/root/.cache/huggingface/hub/` 迁移到 `/root/private_data/.cache/huggingface/hub/`，以符合镜像大小限制（单层镜像数据量不超过 10 GiB）。

## 环境变量配置

已创建环境变量配置文件，自动设置 HuggingFace 模型缓存目录：

- **Shell 脚本配置**: `scripts/env_model.sh`
- **Python 脚本配置**: `scripts/env_model.py`

这些配置文件设置了以下环境变量：
- `HF_HOME=/root/private_data/.cache/huggingface`
- `TRANSFORMERS_CACHE=/root/private_data/.cache/huggingface`
- `HF_DATASETS_CACHE=/root/private_data/.cache/huggingface/datasets`

## 已更新的脚本

### Python 脚本

所有 Python 脚本已在导入 `transformers` 之前加载环境变量配置：

1. `scripts/chat_qwen.py` - Qwen 1.5 对话脚本
2. `scripts/chat_qwen2vl_multimodal.py` - Qwen2-VL 多模态对话脚本
3. `scripts/train_qwen2vl_beer.py` - Qwen2-VL 训练脚本
4. `scripts/train_qwen_beer.py` - Qwen 1.5 训练脚本
5. `scripts/gui_qwen2vl_beer.py` - Qwen2-VL Gradio GUI
6. `scripts/gui_qwen2vl_streamlit.py` - Qwen2-VL Streamlit GUI
7. `scripts/prepare_qwen_local.py` - 下载 Qwen 1.5 模型
8. `scripts/prepare_qwen2vl_local.py` - 下载 Qwen2-VL 模型
9. `scripts/test_qwen2vl_beer.py` - 测试 Qwen2-VL 模型

### Shell 脚本

所有 Shell 脚本已在启动时加载环境变量配置：

1. `scripts/start_gui.sh` - 启动 Gradio GUI
2. `scripts/start_gui_streamlit.sh` - 启动 Streamlit GUI
3. `scripts/train_qwen_with_email.sh` - 训练脚本（带邮件通知）

## 使用方法

### 方式 1: 直接运行脚本（推荐）

所有脚本已自动配置，直接运行即可：

```bash
cd /root/private_data/Yijunhao
.venv/bin/python scripts/chat_qwen.py
```

或

```bash
bash scripts/start_gui.sh
```

### 方式 2: 手动设置环境变量

如果需要手动设置环境变量：

**Shell 环境：**
```bash
source scripts/env_model.sh
```

**Python 环境：**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").absolute()))
import env_model
```

## 验证

验证模型路径是否正确配置：

```bash
# 检查环境变量
source scripts/env_model.sh
echo $HF_HOME

# 验证模型文件是否存在
ls -lh /root/private_data/.cache/huggingface/hub/models--Qwen--*
```

## 迁移的模型

以下模型已迁移到新位置：

- `models--Qwen--Qwen1.5-7B-Chat` (15G)
- `models--Qwen--Qwen2-VL-7B-Instruct` (16G)

总计约 31G 的模型文件已从系统缓存目录移动到持久化存储。

## 注意事项

1. 所有脚本现在会自动使用新的模型路径
2. 如果 HuggingFace 库需要下载新的模型，它们会缓存到新的位置
3. 原有的模型缓存路径 (`~/.cache/huggingface`) 已不再使用
4. 如果需要迁移其他模型，可以使用相同的环境变量配置

## 故障排除

如果遇到模型加载问题，请检查：

1. 环境变量是否正确设置
2. 模型文件是否存在于新路径
3. 文件权限是否正确
4. 磁盘空间是否充足

```bash
# 检查模型文件
du -sh /root/private_data/.cache/huggingface/hub/models--Qwen--*

# 检查环境变量（在脚本中）
python -c "import os; print(os.environ.get('HF_HOME'))"
```

