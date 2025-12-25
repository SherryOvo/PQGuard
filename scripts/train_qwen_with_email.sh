#!/usr/bin/env bash
# 使用 Qwen + 啤酒智慧精酿数据进行训练，并在结束后发送邮件通知。
# 适合配合 nohup 使用，从而在关闭 SSH 后训练仍然继续。
#
# 用法示例（在 /root/private_data/Yijunhao 下）：
#   nohup bash train_qwen_with_email.sh > logs/train_wrapper.out 2>&1 &
#
# 训练详细日志会写入 logs/qwen_train_*.log，成功或失败都会调用 email_notify.py 发送结果到你的邮箱。

set -euo pipefail

# 获取项目根目录（脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# 加载模型环境变量配置
source "${SCRIPT_DIR}/env_model.sh"

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/qwen_train_${TIMESTAMP}.log"

echo "开始训练，日志输出到: ${LOG_FILE}"

# 如果还未生成大规模数据，可以先取消注释下面一行：
# .venv/bin/python scripts/generate_beer_dataset.py --output data/beer_smart_brew_multimodal_kg_train_large.jsonl --num-recipes 200 --qas-per-recipe 10

# ==== 在这里定义你的训练命令 ====
# 注意：下面只是一个示例，请根据你实际想用的训练脚本/参数调整。
TRAIN_CMD=(
  .venv/bin/python
  -m transformers.examples.pytorch.language_modeling.run_clm
  --model_name_or_path Qwen/Qwen1.5-7B-Chat
  --train_file data/beer_smart_brew_multimodal_kg_train_large.jsonl
  --validation_file data/beer_smart_brew_multimodal_kg_train.jsonl
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 8
  --per_device_eval_batch_size 1
  --learning_rate 2e-5
  --num_train_epochs 3
  --block_size 2048
  --do_train
  --do_eval
  --save_total_limit 2
  --save_steps 1000
  --evaluation_strategy steps
  --eval_steps 1000
  --logging_steps 50
  --report_to none
  --output_dir outputs/qwen_beer_sft
)

TRAIN_STATUS="success"

{
  echo "[INFO] $(date '+%F %T') 训练命令: ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"
} >> "${LOG_FILE}" 2>&1 || TRAIN_STATUS="failed"

SUBJECT="Qwen 啤酒智慧精酿训练完成 (${TRAIN_STATUS})"
BODY=$(
  cat <<EOF
训练状态: ${TRAIN_STATUS}
项目目录: ${PROJECT_DIR}
日志文件: ${LOG_FILE}

如需查看详细训练过程，请登录服务器后查看上述日志文件。
EOF
)

echo "[INFO] $(date '+%F %T') 训练结束，状态=${TRAIN_STATUS}，尝试发送邮件通知..."

.venv/bin/python scripts/email_notify.py \
  --subject "${SUBJECT}" \
  --body "${BODY}" || {
  echo "[WARN] 邮件发送失败，请检查 SMTP 配置环境变量。"
}

echo "[INFO] $(date '+%F %T') 脚本执行完毕。"






