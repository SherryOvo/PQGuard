#!/bin/bash
# 启动 Qwen2-VL 多模态模型 GUI 界面

cd /root/private_data/Yijunhao

# 加载模型环境变量配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_model.sh"

echo "=========================================="
echo "启动 Qwen2-VL 多模态模型 GUI 界面"
echo "=========================================="
echo ""

# 获取服务器IP地址
SERVER_IP=$(hostname -I | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="localhost"
fi

echo "服务器信息："
echo "  服务器IP: $SERVER_IP"
echo "  端口: 7860"
echo "  访问地址: http://$SERVER_IP:7860"
echo ""
echo "注意："
echo "  - 如果是远程服务器，请使用上述IP地址在浏览器中访问"
echo "  - 确保防火墙已开放端口 7860"
echo "  - 按 Ctrl+C 可以停止服务"
echo ""
echo "=========================================="
echo ""

.venv/bin/python scripts/gui_qwen2vl_beer.py

