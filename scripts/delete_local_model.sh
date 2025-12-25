#!/bin/bash
# 删除本地 Qwen 模型以节省磁盘空间
# 使用前请确认：训练脚本已改为从云端加载模型

echo "准备删除本地模型目录：models/Qwen1.5-7B-Chat"
echo "删除后，训练和推理将自动从 HuggingFace 云端加载模型。"
echo ""
read -p "确认删除？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消删除。"
    exit 0
fi

if [ -d "models/Qwen1.5-7B-Chat" ]; then
    echo "正在删除 models/Qwen1.5-7B-Chat ..."
    rm -rf models/Qwen1.5-7B-Chat
    echo "✓ 本地模型已删除"
    
    # 显示释放的磁盘空间
    echo ""
    echo "提示："
    echo "  - 训练时模型会从 HuggingFace 自动下载到 ~/.cache/huggingface/"
    echo "  - 训练完成后，只保存 checkpoint 到 outputs/qwen_beer_sft/"
    echo "  - 这样可以节省约 14GB 的本地磁盘空间"
else
    echo "模型目录不存在，无需删除。"
fi





