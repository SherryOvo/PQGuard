#!/bin/bash
# 多模态训练进程监控脚本

LOG_FILE="logs/train_qwen2vl_beer.out"
OUTPUT_DIR="outputs/qwen2vl_beer_sft"

echo "=== 多模态训练进程监控 ==="
echo ""

# 1. 检查训练进程是否在运行
echo "1. 检查训练进程状态："
if ps aux | grep -v grep | grep -q "scripts/train_qwen2vl_beer.py\|train_qwen2vl_beer.py"; then
    echo "   ✓ 训练进程正在运行"
    ps aux | grep -v grep | grep "scripts/train_qwen2vl_beer.py\|train_qwen2vl_beer.py" | awk '{print "   PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%"}'
else
    echo "   ✗ 训练进程未运行"
fi
echo ""

# 2. 查看训练日志最后几行
echo "2. 训练日志最新输出（最后20行）："
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "   日志文件不存在: $LOG_FILE"
fi
echo ""

# 3. 查看训练进度（loss、epoch等）
echo "3. 训练进度摘要："
if [ -f "$LOG_FILE" ]; then
    echo "   最新 loss 值："
    grep -E "'loss':|loss:" "$LOG_FILE" | tail -3 | sed 's/^/   /'
    echo ""
    echo "   当前 epoch："
    grep -E "'epoch':|epoch:" "$LOG_FILE" | tail -1 | sed 's/^/   /'
    echo ""
    echo "   训练步数："
    grep -E "step|Step" "$LOG_FILE" | tail -1 | sed 's/^/   /'
else
    echo "   无法读取日志文件"
fi
echo ""

# 4. 检查已保存的 checkpoint
echo "4. 已保存的 checkpoint："
if [ -d "$OUTPUT_DIR" ]; then
    CHECKPOINTS=$(find "$OUTPUT_DIR" -type d -name "checkpoint-*" 2>/dev/null | sort)
    if [ -n "$CHECKPOINTS" ]; then
        echo "$CHECKPOINTS" | while read -r ckpt; do
            CKPT_NAME=$(basename "$ckpt")
            CKPT_SIZE=$(du -sh "$ckpt" 2>/dev/null | awk '{print $1}')
            echo "   - $CKPT_NAME ($CKPT_SIZE)"
        done
    else
        echo "   暂无 checkpoint（训练可能刚开始）"
    fi
else
    echo "   输出目录不存在: $OUTPUT_DIR"
fi
echo ""

# 5. GPU 使用情况（如果有GPU）
echo "5. GPU 使用情况："
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{printf "   GPU %s (%s): %s%% 使用率, %s/%s 显存\n", $1, $2, $3, $4, $5}'
else
    echo "   未检测到 GPU 或 nvidia-smi 不可用"
fi
echo ""

# 6. 磁盘空间
echo "6. 磁盘空间使用："
df -h . | tail -1 | awk '{printf "   使用率: %s | 可用: %s | 总计: %s\n", $5, $4, $2}'
echo ""

echo "=== 监控完成 ==="
echo ""
echo "提示："
echo "  - 实时查看日志: tail -f $LOG_FILE"
echo "  - 只看 loss: tail -f $LOG_FILE | grep loss"
echo "  - 查看完整日志: cat $LOG_FILE"


