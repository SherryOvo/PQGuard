#!/bin/bash
# 启动 Qwen2-VL 多模态模型 GUI 界面（使用 Streamlit）

cd /root/private_data/Yijunhao

# 加载模型环境变量配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_model.sh"

echo "=========================================="
echo "启动 Qwen2-VL 多模态模型 GUI 界面 (Streamlit)"
echo "=========================================="
echo ""

# 获取服务器IP地址
SERVER_IP=$(hostname -I | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="localhost"
fi

PORT=8501

echo "服务器信息："
echo "  服务器IP: $SERVER_IP"
echo "  端口: $PORT"
echo "  访问地址: http://$SERVER_IP:$PORT"
echo ""
echo "注意："
echo "  - 如果是远程服务器，请使用上述IP地址在浏览器中访问"
echo "  - 或者使用 SSH 端口转发: ssh -L 8501:localhost:8501 root@服务器IP"
echo "  - 按 Ctrl+C 可以停止服务"
echo ""
echo "=========================================="
echo ""

.venv/bin/streamlit run scripts/gui_qwen2vl_streamlit.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true

