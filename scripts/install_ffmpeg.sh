#!/bin/bash
# 安装 FFmpeg 脚本

echo "=========================================="
echo "安装 FFmpeg 以支持语音识别功能"
echo "=========================================="
echo ""

# 检查是否已安装
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg 已安装"
    ffmpeg -version | head -n 1
    exit 0
fi

# 尝试使用 apt-get 安装
if command -v apt-get &> /dev/null; then
    echo "正在使用 apt-get 安装 FFmpeg..."
    apt-get update
    apt-get install -y ffmpeg
    
    # 验证安装
    if command -v ffmpeg &> /dev/null; then
        echo ""
        echo "✓ FFmpeg 安装成功！"
        ffmpeg -version | head -n 1
        exit 0
    else
        echo "✗ 安装失败"
        exit 1
    fi
else
    echo "错误：未找到 apt-get，请手动安装 FFmpeg"
    echo "参考：INSTALL_FFMPEG.md"
    exit 1
fi

