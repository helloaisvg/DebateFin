# DebateFin: 可信赖的多智能体金融分析系统


DebateFin是一个基于LLM的多智能体系统，用于企业基本面分析，通过结构化辩论（Analyst、Risk、Trader智能体）来减少金融推理中的幻觉。

##  核心架构

### 技术栈
- **LLM**: DeepSeek Chat API (兼容 OpenAI 格式)
- **框架**: LangGraph 用于状态化多智能体图，带辩论循环（最多3轮：证据-反驳-综合）
- **智能体**: 
  - **Supervisor**: 路由和协调
  - **Analyst**: 提取财务指标和证据
  - **Risk**: 情感分析和风险标志
  - **Trader**: 预测和投资建议
  - **Judge**: 回测引导评分
- **工具**: 
  - **yfinance**: 纯 yfinance 数据获取（三保险方案：本地缓存 + Streamlit缓存 + 实时获取）
  - **PyTorch LSTM**: 增长预测
  - **pandas**: 财务指标计算（ROE、Sharpe等）
  - **HuggingFace**: 新闻情感分析
  - **VectorBT**: 回测（可选）
- **前端**: Streamlit 交互式界面
- **PDF生成**: WeasyPrint（完美支持中文）
- **安全**: API密钥通过Streamlit secrets管理；工具接地避免幻觉

##  主要功能

1. **多智能体辩论系统**: 结构化辩论流程，最多3轮（证据-反驳-综合）
2. **财务指标分析**: 自动计算ROE、Sharpe比率、波动率等关键指标
3. **增长预测**: 使用LSTM模型预测未来增长趋势
4. **风险评估**: 市场情感分析和风险标志识别
5. **投资建议**: 基于多智能体分析的买入/持有/卖出建议
6. **回测功能**: 5年历史数据实时回测
7. **消融研究**: 可切换有/无辩论模式，对比效果
8. **PDF报告**: 自动生成可下载的分析报告

##  快速开始

>  **详细部署指南**: 查看 [RUN.md](RUN.md) 获取完整的部署和运行说明

### 方法1: 使用设置脚本（推荐）

```bash
# 一键设置虚拟环境和依赖
./setup.sh

# 配置API密钥
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 编辑 .streamlit/secrets.toml，填入 DEEPSEEK_API_KEY
# 获取密钥: https://platform.deepseek.com/

# 运行应用
./run.sh
```

### 方法2: 手动设置

#### 1. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. 配置API密钥

#### 方法1: Streamlit Secrets (推荐用于Streamlit Cloud)

创建 `.streamlit/secrets.toml` 文件：

```toml
# DeepSeek API（推荐）
DEEPSEEK_API_KEY = "sk-your-deepseek-api-key-here"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
```

#### 方法2: 环境变量

```bash
export DEEPSEEK_API_KEY="sk-your-deepseek-api-key-here"
export DEEPSEEK_API_BASE="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
```

#### 3. 运行应用

**使用启动脚本（推荐）**：
```bash
./run.sh
```

**直接运行**：
```bash
# 确保虚拟环境已激活
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。

#### Docker部署

```bash
# 构建镜像
docker build -t debatefin .

# 运行容器
docker run -p 8501:8501 \
  -e DEEPSEEK_API_KEY=your-deepseek-key \
  -e DEEPSEEK_API_BASE=https://api.deepseek.com \
  -e DEEPSEEK_MODEL=deepseek-chat \
  debatefin
```


## 使用示例

### 基本使用

1. **输入股票代码**: 在侧边栏输入股票代码（如 `600519` 茅台，`AAPL` 苹果）
2. **输入查询**: 描述你的分析需求（如"预测2026年ROE和投资建议"）
3. **配置选项**:
   - 启用/禁用多智能体辩论（消融研究）
   - 设置最大辩论轮数（1-3轮）
4. **点击"开始分析"**: 系统将运行多智能体分析流程
5. **查看结果**:
   - **综合分析**: 最终投资建议和综合报告
   - **辩论日志**: 查看各智能体的辩论过程
   - **图表分析**: 财务指标图表和回测结果
   - **导出报告**: 下载PDF格式的分析报告

### 示例查询

```
股票代码: 600519
查询: 预测2026年ROE和投资建议
```

系统将：
1. Analyst智能体提取财务指标（ROE、Sharpe等）
2. Risk智能体评估风险因素和市场情绪
3. Trader智能体生成增长预测和投资建议
4. Supervisor智能体综合所有观点，给出最终建议
5. 显示交互式图表和回测结果

##  输出示例

### 投资建议
-  **买入**: Sharpe比率 > 1.5，ROE > 15%，低风险
-  **持有**: 中等指标，需要观察
-  **卖出**: 高风险信号，财务指标恶化

### 财务指标
- **ROE**: 净资产收益率
- **Sharpe比率**: 风险调整后收益（>1.0为良好）
- **波动率**: 年化波动率
- **PE/PB**: 估值指标

### 回测结果
- 5年历史数据回测
- 累计收益率曲线
- 移动平均策略表现
- 风险指标对比

##  消融研究功能

系统支持对比有/无辩论机制的效果：

1. **启用辩论**: 多智能体进行3轮结构化辩论
2. **禁用辩论**: 直接综合各智能体观点，无辩论过程

通过切换此选项，可以研究辩论机制对减少幻觉和提升分析质量的影响。

##  项目结构

```
DebateFin/
├── app.py                 # Streamlit主应用 + LangGraph后端
├── tools.py               # 工具函数（数据获取、指标计算、预测、回测）
├── models.py              # PyTorch LSTM模型
├── report_generator.py    # PDF报告生成器（WeasyPrint）
├── cache_utils.py         # 缓存工具
├── guardrail_validator.py # 工具强制校验器
├── hallucination_checker.py # 幻觉检查器
├── ppo_router.py          # PPO路由器（回测引导）
├── requirements.txt       # Python依赖
├── setup.sh              # 一键环境设置脚本
├── run.sh                # 启动脚本
├── Dockerfile            # Docker容器配置
├── data_cache/           # 本地数据缓存目录（自动创建）
└── README.md             # 本文档
```


##  依赖说明

主要依赖包：
- `langchain`, `langgraph`: 多智能体框架
- `langchain-openai`: DeepSeek API 支持（兼容 OpenAI 格式）
- `streamlit`: Web界面
- `yfinance`: 股票数据获取（纯 yfinance，三保险缓存方案）
- `torch`: LSTM模型
- `transformers`: 情感分析
- `plotly`: 交互式图表
- `vectorbt`: 回测（可选）
- `weasyprint`: PDF生成（完美支持中文）

完整列表见 `requirements.txt`。


##  技术细节

### LangGraph工作流

```
开始 → Analyst → Risk → Trader → Supervisor
                              ↓
                        继续辩论? → 是 → Analyst (下一轮)
                              ↓
                            否 → 结束
```

### 辩论流程

1. **第1轮**: Analyst提取证据 → Risk评估风险 → Trader给出初步预测
2. **第2轮**: Supervisor综合观点，各智能体可以反驳或补充
3. **第3轮**: 最终综合，生成最终建议

### 工具接地

所有财务数据通过 **纯 yfinance** 获取，避免LLM幻觉：
- 实时股价和历史数据（10年数据，自动缓存）
- 财务指标（ROE、ROA等）
- 估值指标（PE、PB等）

**三保险数据获取方案**：
1. 本地文件缓存（永久保存，首次获取后不再联网）
2. Streamlit 缓存（1小时内不重复请求）
3. yfinance 实时获取（最后兜底）

### PDF 报告生成

使用 **WeasyPrint** 专业 PDF 库，完美支持中文：
- 基于 HTML/CSS 渲染
- 自动使用系统中文字体
- 支持复杂的表格和样式

##  贡献

欢迎提交Issue和Pull Request！


---

**DebateFin** - 让AI金融分析更可信赖 
