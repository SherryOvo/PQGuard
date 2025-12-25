#!/bin/bash
#
# 修复 ffmpeg 系统库依赖问题
# 解决 libblas.so.3 和 liblapack.so.3 等库无法找到的问题
#

echo "正在修复 ffmpeg 系统库依赖问题..."

# 创建必要的符号链接
if [ ! -f /usr/lib/x86_64-linux-gnu/libblas.so.3 ]; then
    if [ -f /usr/lib/x86_64-linux-gnu/blas/libblas.so.3 ]; then
        ln -s /usr/lib/x86_64-linux-gnu/blas/libblas.so.3 /usr/lib/x86_64-linux-gnu/libblas.so.3
        echo "✓ 已创建 libblas.so.3 符号链接"
    else
        echo "✗ 警告：找不到 libblas.so.3 文件"
    fi
else
    echo "✓ libblas.so.3 符号链接已存在"
fi

if [ ! -f /usr/lib/x86_64-linux-gnu/liblapack.so.3 ]; then
    if [ -f /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3 ]; then
        ln -s /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3 /usr/lib/x86_64-linux-gnu/liblapack.so.3
        echo "✓ 已创建 liblapack.so.3 符号链接"
    else
        echo "✗ 警告：找不到 liblapack.so.3 文件"
    fi
else
    echo "✓ liblapack.so.3 符号链接已存在"
fi

# 更新动态链接器缓存
echo "正在更新动态链接器缓存..."
ldconfig

# 验证 ffmpeg 是否正常工作
echo ""
echo "验证 ffmpeg 是否正常工作..."
if ffmpeg -version > /dev/null 2>&1; then
    echo "✓ ffmpeg 工作正常"
    ffmpeg -version | head -1
else
    echo "✗ ffmpeg 仍然无法正常工作"
    echo "请检查错误信息并手动修复"
fi

echo ""
echo "修复完成！"

