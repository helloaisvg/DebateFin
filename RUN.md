# DebateFin 部署运行指南

## 🚀 快速开始（3步）

### 1. 创建虚拟环境并安装依赖

**推荐使用虚拟环境**（避免依赖冲突）：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

**不使用虚拟环境**（不推荐）：
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

**方法1: Streamlit Secrets（推荐）**

```bash
# 复制示例文件
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 编辑并填入你的API密钥
nano .streamlit/secrets.toml
```

在 `secrets.toml` 中填入：
```toml
# DeepSeek API（推荐）
DEEPSEEK_API_KEY = "sk-your-deepseek-key-here"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 可选配置
REDIS_URL = "redis://localhost:6379"  # Redis缓存（可选）
# 注意: 数据源使用yfinance + 雪球公开数据，无需额外API密钥
```

**方法2: 环境变量**

```bash
export DEEPSEEK_API_KEY="sk-your-deepseek-key-here"
export DEEPSEEK_API_BASE="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
export REDIS_URL="redis://localhost:6379"  # 可选
```

### 3. 运行应用

**使用启动脚本（推荐）**
```bash
chmod +x run.sh
./run.sh
```

**直接运行**
```bash
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动

---

## 🐳 Docker部署

### 构建镜像

```bash
docker build -t debatefin .
```

### 运行容器

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your-key" \
  -e REDIS_URL="redis://host.docker.internal:6379" \
  debatefin
```

### 使用docker-compose（推荐）

创建 `docker-compose.yml`:
```yaml
version: '3.8'
services:
  debatefin:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

运行：
```bash
docker-compose up -d
```

---

## ☁️ Streamlit Cloud部署（一键部署）

### 步骤

1. **推送代码到GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **访问Streamlit Cloud**
   - 打开 https://streamlit.io/cloud
   - 点击 "New app"
   - 选择你的GitHub仓库

3. **配置应用**
   - Main file path: `app.py`
   - Branch: `main`

4. **设置Secrets**
   在Streamlit Cloud Dashboard的"Secrets"中添加：
   ```toml
   OPENAI_API_KEY = "your-openai-key"
   REDIS_URL = "your-redis-url"  # 可选
   ```

5. **点击Deploy**

应用将在几分钟内自动部署！

---

## 🔧 本地开发环境

### Python版本要求

- Python >= 3.9 (推荐 3.11)

### 完整安装步骤（使用虚拟环境）

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd DebateFin

# 2. 创建虚拟环境（必需）
python -m venv venv

# 3. 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. 升级pip（推荐）
pip install --upgrade pip

# 5. 安装依赖
pip install -r requirements.txt

# 6. 配置密钥
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 编辑 .streamlit/secrets.toml，填入 OPENAI_API_KEY

# 7. 运行应用
streamlit run app.py
```

### 虚拟环境管理

**激活虚拟环境**：
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**退出虚拟环境**：
```bash
deactivate
```

**删除虚拟环境**（重新创建时）：
```bash
# Linux/Mac
rm -rf venv

# Windows
rmdir /s venv
```

**验证虚拟环境**：
```bash
# 检查Python路径（应该指向venv）
which python  # Linux/Mac
where python  # Windows

# 检查已安装的包
pip list
```

---

## 🧪 测试运行

### 基本测试

1. 打开应用：`http://localhost:8501`
2. 输入股票代码：`600519` (茅台) 或 `AAPL` (苹果)
3. 选择辩论模式：`完整分层辩论`
4. 点击"开始分析"
5. 查看结果

### 消融研究测试

1. 选择"运行消融对比实验"
2. 系统会同时运行有/无辩论版本
3. 查看对比结果和指标差异

---

## ⚙️ 配置选项

### 辩论模式

- **完整分层辩论 (debate)**: L1→L2→L3完整流程
- **无辩论直接综合 (no_debate)**: 跳过辩论，直接综合
- **单智能体 (single_agent)**: 仅Analyst，用于消融研究

### 环境变量

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI API密钥 |
| `REDIS_URL` | ❌ | Redis连接URL（缓存） |
| 数据源 | ✅ | yfinance + 雪球公开数据（无需API密钥） |

---

## 🐛 故障排除

### 常见问题

**1. API密钥错误**
```
⚠️ 请设置OPENAI_API_KEY在Streamlit secrets或环境变量中
```
解决：检查 `.streamlit/secrets.toml` 或环境变量

**2. 导入错误**
```
ModuleNotFoundError: No module named 'langchain'
```
解决：`pip install -r requirements.txt`

**3. 端口被占用**
```
Port 8501 is already in use
```
解决：`streamlit run app.py --server.port 8502`

**4. Redis连接失败**
```
Redis connection failed, using memory cache
```
解决：这是正常的，系统会自动回退到内存缓存

**5. 数据获取失败**
```
无法获取股票数据
```
解决：检查网络连接和股票代码格式

---

## 📊 性能优化

### 启用Redis缓存

1. 安装Redis：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   ```

2. 启动Redis：
   ```bash
   redis-server
   ```

3. 配置连接：
   ```toml
   # .streamlit/secrets.toml
   REDIS_URL = "redis://localhost:6379"
   ```

### 减少API调用

- 使用Redis缓存（减少重复API调用）
- 调整 `cache_utils.py` 中的TTL
- 使用消融研究模式减少智能体数量

---

## 🔒 安全注意事项

1. **永远不要提交API密钥**
   - `.streamlit/secrets.toml` 已在 `.gitignore` 中
   - 使用环境变量或Streamlit Cloud Secrets

2. **生产环境部署**
   - 使用HTTPS
   - 设置访问限制
   - 监控API使用量

---

## 📚 更多资源

- [README.md](README.md) - 完整项目文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [DEPLOYMENT.md](DEPLOYMENT.md) - 详细部署指南
- [FEATURES.md](FEATURES.md) - 功能说明
- [ICLR_FEATURES.md](ICLR_FEATURES.md) - ICLR创新点

---

## ✅ 验证部署

运行后应该看到：

1. ✅ Streamlit界面正常加载
2. ✅ 侧边栏显示配置选项
3. ✅ 可以输入股票代码
4. ✅ 点击"开始分析"后正常运行
5. ✅ 显示分析结果和图表

如果所有步骤都正常，部署成功！🎉

