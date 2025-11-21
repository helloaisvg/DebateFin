#!/bin/bash
# DebateFin 启动脚本（支持虚拟环境）

echo "🚀 启动 DebateFin..."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 未找到虚拟环境，正在创建..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "🔌 激活虚拟环境..."
source venv/bin/activate

# 检查Python版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version (虚拟环境)"

# 检查依赖
if [ ! -f "requirements.txt" ]; then
    echo "❌ 未找到 requirements.txt"
    exit 1
fi

# 检查是否需要安装依赖
if [ "$1" == "--install" ] || [ ! -f "venv/.installed" ]; then
    echo "📦 安装/更新依赖..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.installed
    echo "✅ 依赖安装完成"
fi

# 检查API密钥
if [ -z "$DEEPSEEK_API_KEY" ] && [ -z "$OPENAI_API_KEY" ] && [ ! -f ".streamlit/secrets.toml" ]; then
    echo "⚠️  警告: 未设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
    echo "请设置环境变量或创建 .streamlit/secrets.toml"
    echo ""
    echo "方法1: 环境变量（推荐DeepSeek）"
    echo "  export DEEPSEEK_API_KEY='sk-your-deepseek-key'"
    echo "  export DEEPSEEK_API_BASE='https://api.deepseek.com'"
    echo "  export DEEPSEEK_MODEL='deepseek-chat'"
    echo ""
    echo "方法2: Streamlit secrets"
    echo "  复制 .streamlit/secrets.toml.example 为 .streamlit/secrets.toml"
    echo "  并填入 DEEPSEEK_API_KEY"
    echo ""
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 启动Streamlit
echo "🌐 启动Streamlit应用..."
echo "📍 访问地址: http://localhost:8501"
echo "💡 按 Ctrl+C 停止应用"
echo ""
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

