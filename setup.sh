# DebateFin 环境设置脚本

set -e  # 遇到错误立即退出

echo "🔧 DebateFin 环境设置"
echo "===================="

# 检查Python版本
echo "📋 检查Python版本..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请先安装Python 3.9+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python版本: $python_version"

# 创建虚拟环境
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  删除旧虚拟环境..."
        rm -rf venv
    else
        echo "📦 使用现有虚拟环境"
    fi
fi

if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "🔌 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "⬆️  升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 安装依赖（这可能需要几分钟）..."
pip install -r requirements.txt

# 标记已安装
touch venv/.installed

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "下一步："
echo "1. 配置API密钥:"
echo "   cp .streamlit/secrets.toml.example .streamlit/secrets.toml"
echo "   # 编辑 .streamlit/secrets.toml，填入 DEEPSEEK_API_KEY"
echo "   # 获取密钥: https://platform.deepseek.com/"
echo ""
echo "2. 运行应用:"
echo "   source venv/bin/activate"
echo "   streamlit run app.py"
echo ""
echo "   或使用启动脚本:"
echo "   ./run.sh"
echo ""

