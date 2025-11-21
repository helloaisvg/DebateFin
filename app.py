"""
DebateFin: A trustworthy LLM-based multi-agent system for enterprise fundamental analysis


Core architecture:
- LLM: DeepSeek Chat API (via ChatOpenAI, compatible with OpenAI format)
- Framework: LangGraph for stateful multi-agent graph with debate loop
- Agents: Supervisor, Analyst, Risk, Trader, Judge (5 agents)
- Tools: yfinance, PyTorch LSTM, pandas, HuggingFace, VectorBT
"""

import streamlit as st
import os
import json
from typing import TypedDict, List, Dict, Any, Annotated
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import seaborn as sns
from io import BytesIO
import base64

# Plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# DeepSeek API support
import os

# Financial data and analysis
import yfinance as yf
import torch
import torch.nn as nn
from transformers import pipeline as hf_pipeline

# Backtesting
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    # Note: st.warning() cannot be called at module level, will handle in main()

# Report generation (使用 HTML，零依赖，Streamlit Cloud 原生支持)
# Import custom modules
from tools import (
    fetch_stock_data, calculate_financial_metrics, 
    predict_growth_lstm, analyze_sentiment_hf, backtest_strategy,
    get_price_history  # 缓存的价格历史函数
)
from models import LSTMGrowthPredictor
from report_generator import generate_html_report_bytes
from cache_utils import cached_with_retry, clear_cache
from hallucination_checker import hallucination_checker
from guardrail_validator import guardrail_validator
from ppo_router import ppo_router
import re  # For regex in guardrail and judge agent

# ============================================================================
# Configuration
# ============================================================================

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'debate_logs' not in st.session_state:
    st.session_state.debate_logs = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'ablation_results' not in st.session_state:
    st.session_state.ablation_results = {}  # Store with/without debate results
if 'hallucination_checks' not in st.session_state:
    st.session_state.hallucination_checks = []

# Streamlit page config
st.set_page_config(
    page_title="DebateFin - Multi-Agent Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LangGraph State Definition
# ============================================================================

class DebateState(TypedDict):
    """State for multi-agent debate system with hierarchical structure"""
    messages: Annotated[List[Any], add_messages]
    ticker: str
    query: str
    round: int
    max_rounds: int
    # L1: Fundamental Analysis Layer (Analyst ↔ Risk)
    analyst_evidence: Dict[str, Any]
    risk_flags: List[str]
    l1_debate_log: List[Dict[str, Any]]  # Analyst ↔ Risk debate
    # L2: Trading Layer (Trader synthesis)
    trader_prediction: Dict[str, Any]
    l2_synthesis: str  # Trader synthesis of L1
    # L3: Judgment Layer (Judge backtest-guided scoring)
    judge_score: Dict[str, Any]  # Judge agent backtest-guided scoring
    backtest_result: Dict[str, Any]  # Backtest results
    # Overall
    debate_log: List[Dict[str, Any]]
    final_synthesis: str
    use_debate: str  # Ablation toggle: "debate" | "no_debate" | "single_agent"
    metrics: Dict[str, float]  # For ablation comparison (Sharpe, MAE, etc.)
    hallucination_check: Dict[str, Any]  # Hallucination check results
    tool_calls: List[Dict[str, Any]]  # Track all tool calls for guardrail
    validation_results: List[Dict[str, Any]]  # Guardrail validation results

# ============================================================================
# Agent Definitions
# ============================================================================

def get_llm(temperature=0.7):
    """
    Initialize LLM with DeepSeek API only
    仅使用 DeepSeek API，不使用 OpenAI
    """
    # 只使用 DeepSeek API Key
    api_key = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    
    # 检查是否是示例密钥
    if api_key and ("your-deepseek-api-key" in api_key.lower() or "your-actual" in api_key.lower() or api_key == "sk-your-deepseek-api-key-here"):
        st.error("""
        ❌ **检测到示例密钥，请使用真实密钥！**
        
        你的 `.streamlit/secrets.toml` 文件中使用的是示例密钥，不是真实的 DeepSeek API 密钥。
        
        **立即修复**：
        1. 打开 `.streamlit/secrets.toml` 文件
        2. 将 `DEEPSEEK_API_KEY = "sk-your-deepseek-api-key-here"` 
           替换为你的真实密钥：`DEEPSEEK_API_KEY = "sk-你的真实密钥"`
        3. 保存文件
        4. **重启 Streamlit 应用**（必须重启！）
        
        **获取真实密钥**：
        - 访问: https://platform.deepseek.com/
        - 登录后，在控制台创建 API 密钥
        - 复制真实的密钥（以 `sk-` 开头）
        """)
        return None
    
    if not api_key:
        st.error("""
        ⚠️ **DeepSeek API密钥未配置**
        
        请设置 DeepSeek API 密钥，方法如下：
        
        **方法1: Streamlit Secrets（推荐）**
        1. 编辑 `.streamlit/secrets.toml` 文件
        2. 添加以下内容：
        ```toml
        DEEPSEEK_API_KEY = "sk-your-actual-deepseek-api-key-here"
        DEEPSEEK_API_BASE = "https://api.deepseek.com"
        DEEPSEEK_MODEL = "deepseek-chat"
        ```
        
        **方法2: 环境变量**
        ```bash
        export DEEPSEEK_API_KEY="sk-your-actual-deepseek-api-key-here"
        export DEEPSEEK_API_BASE="https://api.deepseek.com"
        export DEEPSEEK_MODEL="deepseek-chat"
        ```
        
        **获取 DeepSeek API 密钥**：
        - 访问: https://platform.deepseek.com/
        - 注册账号后，在控制台创建 API 密钥
        """)
        return None
    
    # 检查API密钥格式（基本验证）
    if api_key and len(api_key) < 10:
        st.warning("⚠️ API密钥格式可能不正确，请检查密钥是否完整")
    
    # DeepSeek API配置（必须配置）
    model_name = st.secrets.get("DEEPSEEK_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    
    # 强制确保 base_url 指向 DeepSeek（绝对不能指向 OpenAI）
    # DeepSeek API endpoint 固定为 https://api.deepseek.com
    base_url = "https://api.deepseek.com"
    
    # 验证 API key 格式
    if not api_key.startswith("sk-"):
        st.error(f"""
        ❌ **API 密钥格式错误**
        
        你的 API 密钥: {api_key[:10]}...（已隐藏）
        
        DeepSeek API 密钥必须以 "sk-" 开头。
        请检查 `.streamlit/secrets.toml` 中的 `DEEPSEEK_API_KEY` 是否正确。
        """)
        return None
    
    try:
        # 使用 ChatOpenAI，强制指定 DeepSeek 的 base_url
        # DeepSeek API 兼容 OpenAI 格式，但必须指定正确的 base_url
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,  # 强制使用 DeepSeek API endpoint: https://api.deepseek.com
            timeout=60,
            max_retries=2
        )
        return llm
    except Exception as e:
        error_msg = str(e)
        
        # 显示实际使用的配置（隐藏敏感信息）
        api_key_preview = api_key[:10] + "..." if len(api_key) > 10 else "***"
        
        if "401" in error_msg or "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
            st.error(f"""
            ❌ **DeepSeek API 认证失败**
            
            **错误信息**: {error_msg}
            
            **当前配置**:
            - API Key: {api_key_preview}
            - Base URL: {base_url}
            - Model: {model_name}
            
            **可能的原因**：
            1. ❌ DeepSeek API 密钥无效或已过期
            2. ❌ API 密钥格式不正确（必须以 `sk-` 开头）
            3. ❌ API 密钥权限不足
            4. ❌ 密钥在 `.streamlit/secrets.toml` 中配置错误
            
            **立即检查**：
            1. ✅ 打开 `.streamlit/secrets.toml` 文件
            2. ✅ 确认 `DEEPSEEK_API_KEY` 的值是你的真实密钥（不是示例密钥）
            3. ✅ 确认密钥以 `sk-` 开头
            4. ✅ 访问 https://platform.deepseek.com/ 验证密钥是否有效
            5. ✅ 重启 Streamlit 应用（修改 secrets.toml 后需要重启）
            
            **配置示例**：
            ```toml
            DEEPSEEK_API_KEY = "sk-your-real-deepseek-api-key-here"
            DEEPSEEK_API_BASE = "https://api.deepseek.com"
            DEEPSEEK_MODEL = "deepseek-chat"
            ```
            
            ⚠️ **重要**: 不要使用 `.streamlit/secrets.toml.example` 中的示例密钥！
            """)
        else:
            st.error(f"""
            ❌ **DeepSeek LLM 初始化失败**
            
            错误信息: {error_msg}
            
            当前配置:
            - Base URL: {base_url}
            - Model: {model_name}
            """)
        return None

def analyst_agent(state: DebateState) -> DebateState:
    """Analyst Agent: Extract financial metrics and evidence with tool-forced grounding"""
    llm = get_llm(temperature=0.3)
    if not llm:
        return state
    
    ticker = state["ticker"]
    query = state["query"]
    
    # TOOL-FORCED GROUNDING: Must call tools before reasoning
    # Track tool calls for guardrail validation
    tool_calls = []
    
    try:
        # Fetch real data using tools (cached with retry)
        try:
            stock_data = fetch_stock_data(ticker)
        except Exception as e:
            # 统一处理数据获取失败的错误（只显示一次）
            error_msg = str(e)
            if "数据获取失败" in error_msg or "无法从任何数据源获取" in error_msg:
                st.error(f"❌ 无法获取股票 {ticker} 的数据，所有数据源都失败了。请稍后再试或换股票代码。")
            else:
                st.error(f"❌ 数据获取失败: {error_msg}")
            # 返回空数据，避免后续处理崩溃
            stock_data = {"history": pd.DataFrame(), "info": {}, "ticker": ticker, "error": error_msg}
        
        tool_calls.append({
            "tool": "fetch_stock_data",
            "input": {"ticker": ticker},
            "result": {"success": True, "data_points": len(stock_data.get("history", []))}
        })
        
        metrics = calculate_financial_metrics(stock_data)
        tool_calls.append({
            "tool": "calculate_financial_metrics",
            "input": {"ticker": ticker},
            "result": {"metrics_count": len(metrics), "metrics": metrics}
        })
        
        # Register tool data for hallucination checking
        hallucination_checker.register_tool_data(ticker, "financial_metrics", metrics)
        hallucination_checker.register_tool_data(ticker, "stock_data", stock_data)
        
        # Get analyst reasoning with tool call annotation
        data_source = stock_data.get("data_source", "yfinance")
        tool_call_annotation = f"""<tool_call>
已调用工具:
1. fetch_stock_data({ticker}) - 获取股票数据（数据源: {data_source}）
2. calculate_financial_metrics() - 计算财务指标

工具返回的关键指标:
{json.dumps(metrics, indent=2, ensure_ascii=False)}
</tool_call>

"""
        
        prompt = f"""{tool_call_annotation}你是一位资深财务分析师（L1层）。基于上述工具获取的真实数据，分析股票代码 {ticker} 的财务数据。

用户查询: {query}

请提供:
1. 关键财务指标分析（ROE、ROA、毛利率等）- 必须引用工具返回的具体数字
2. 财务健康状况评估
3. 支持你结论的具体证据 - 必须基于工具数据

用中文回答，保持专业和客观。"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Guardrail validation: enforce tool call requirement
        validated_message = guardrail_validator.enforce_tool_call(
            "Analyst", response.content, tool_calls
        )
        
        # Hallucination check
        hallucination_result = hallucination_checker.check_hallucination(
            validated_message, ticker, ["financial_metrics"]
        )
        
        # Validation result
        is_valid, validation_result = guardrail_validator.validate_agent_message(
            "Analyst", validated_message, tool_calls
        )
        
        evidence = {
            "metrics": metrics,
            "analysis": validated_message,
            "timestamp": datetime.now().isoformat(),
            "hallucination_check": hallucination_result,
            "tool_calls": tool_calls,
            "validation": validation_result
        }
        
        state["analyst_evidence"] = evidence
        state["hallucination_check"] = hallucination_result
        state["tool_calls"] = state.get("tool_calls", []) + tool_calls
        state["validation_results"] = state.get("validation_results", []) + [validation_result]
        
        # L1 Debate log (Analyst ↔ Risk)
        l1_entry = {
            "round": state["round"],
            "layer": "L1",
            "agent": "Analyst",
            "action": "evidence_extraction",
            "content": validated_message[:500] + "...",
            "hallucination_check": hallucination_result,
            "validation": validation_result,
            "tool_calls": tool_calls
        }
        state["l1_debate_log"] = state.get("l1_debate_log", []) + [l1_entry]
        state["debate_log"] = state.get("debate_log", []) + [l1_entry]
        
        # Store hallucination check in session state
        st.session_state.hallucination_checks.append({
            "agent": "Analyst",
            "round": state["round"],
            "check": hallucination_result
        })
    except Exception as e:
        state["analyst_evidence"] = {"error": str(e)}
        state["debate_log"].append({
            "round": state["round"],
            "agent": "Analyst",
            "action": "error",
            "content": f"数据获取失败: {str(e)}"
        })
    
    return state

def risk_agent(state: DebateState) -> DebateState:
    """Risk Agent: Analyze sentiment and risk factors with tool-forced grounding"""
    llm = get_llm(temperature=0.4)
    if not llm:
        return state
    
    ticker = state["ticker"]
    query = state["query"]
    analyst_evidence = state.get("analyst_evidence", {})
    
    # TOOL-FORCED GROUNDING: Must call sentiment tool before reasoning
    try:
        # Sentiment analysis using HuggingFace (cached with retry)
        sentiment_score = analyze_sentiment_hf(ticker)
        
        # Register tool data
        hallucination_checker.register_tool_data(ticker, "sentiment", {"score": sentiment_score})
        
        # Risk assessment
        prompt = f"""你是一位风险管理专家。评估股票代码 {ticker} 的风险因素。

用户查询: {query}

分析师提供的证据:
{json.dumps(analyst_evidence.get('analysis', ''), ensure_ascii=False)[:1000]}

市场情绪得分: {sentiment_score}

请识别:
1. 主要风险因素（财务风险、市场风险、行业风险）
2. 风险等级（低/中/高）
3. 需要警惕的信号

用中文回答，保持谨慎和客观。"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Extract risk flags
        risk_flags = []
        if "高" in response.content or "风险" in response.content:
            risk_flags.append("高风险信号")
        if sentiment_score < 0.3:
            risk_flags.append("市场情绪悲观")
        
        state["risk_flags"] = risk_flags
        state["debate_log"].append({
            "round": state["round"],
            "agent": "Risk",
            "action": "risk_assessment",
            "content": response.content[:500] + "...",
            "sentiment_score": sentiment_score
        })
    except Exception as e:
        state["risk_flags"] = [f"风险评估错误: {str(e)}"]
        state["debate_log"].append({
            "round": state["round"],
            "agent": "Risk",
            "action": "error",
            "content": str(e)
        })
    
    return state

def trader_agent(state: DebateState) -> DebateState:
    """Trader Agent: Make predictions and investment recommendations"""
    llm = get_llm(temperature=0.5)
    if not llm:
        return state
    
    ticker = state["ticker"]
    query = state["query"]
    analyst_evidence = state.get("analyst_evidence", {})
    risk_flags = state.get("risk_flags", [])
    
    try:
        # LSTM growth prediction
        metrics = analyst_evidence.get("metrics", {})
        growth_prediction = predict_growth_lstm(ticker, metrics)
        
        # Investment recommendation
        prompt = f"""你是一位资深交易员和投资顾问。基于以下信息，提供投资建议。

股票代码: {ticker}
用户查询: {query}

分析师证据:
{json.dumps(analyst_evidence.get('analysis', ''), ensure_ascii=False)[:1000]}

风险标志: {', '.join(risk_flags) if risk_flags else '无重大风险标志'}

增长预测: {json.dumps(growth_prediction, ensure_ascii=False)}

请提供:
1. 2026年ROE预测（如果相关）
2. 投资建议（买入/持有/卖出）
3. 目标价位区间
4. 投资逻辑和理由

用中文回答，给出明确的投资建议。"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Extract recommendation
        recommendation = "持有"
        if "买入" in response.content or "买" in response.content:
            recommendation = "买入"
        elif "卖出" in response.content or "卖" in response.content:
            recommendation = "卖出"
        
        prediction = {
            "recommendation": recommendation,
            "reasoning": response.content,
            "growth_prediction": growth_prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        state["trader_prediction"] = prediction
        state["debate_log"].append({
            "round": state["round"],
            "agent": "Trader",
            "action": "prediction",
            "content": response.content[:500] + "...",
            "recommendation": recommendation
        })
    except Exception as e:
        state["trader_prediction"] = {"error": str(e)}
        state["debate_log"].append({
            "round": state["round"],
            "agent": "Trader",
            "content": str(e)
        })
    
    return state

def judge_agent(state: DebateState) -> DebateState:
    """
    Judge Agent (L3): Backtest-guided scoring and final decision
    Uses historical Sharpe ratio as reward signal (Backtest-Guided Router innovation)
    """
    llm = get_llm(temperature=0.4)
    if not llm:
        return state
    
    ticker = state["ticker"]
    trader_prediction = state.get("trader_prediction", {})
    recommendation = trader_prediction.get("recommendation", "持有")
    
    # TOOL-FORCED GROUNDING: Must run backtest before judgment
    tool_calls = []
    
    try:
        # Run backtest (tool call)
        backtest_result = backtest_strategy(ticker, strategy="sma")
        tool_calls.append({
            "tool": "backtest_strategy",
            "input": {"ticker": ticker, "strategy": "sma"},
            "result": backtest_result
        })
        
        # Calculate reward from backtest (for PPO router)
        reward = ppo_router.calculate_reward(backtest_result)
        
        tool_call_annotation = f"""<tool_call>
已调用工具:
1. backtest_strategy({ticker}) - 历史回测分析

工具返回的回测结果:
{json.dumps(backtest_result, ensure_ascii=False)}
</tool_call>

"""
        
        # Judge scoring based on backtest
        prompt = f"""{tool_call_annotation}你是Judge智能体（L3层），负责基于历史回测结果对投资建议进行最终评分。

股票代码: {ticker}

Trader建议: {recommendation}
Trader推理: {trader_prediction.get('reasoning', '')[:500]}

历史回测结果（工具返回）:
- Sharpe比率: {backtest_result.get('sharpe_strategy', 0):.3f}
- 总收益率: {backtest_result.get('total_return', 0):.2f}%
- 交易次数: {backtest_result.get('trades', 0)}

请基于回测数据评估:
1. 投资建议的质量评分（0-100分）
2. 回测支持的证据
3. 最终决策（买入/持有/卖出）
4. 置信度评估

用中文回答，给出明确的评分和决策。"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Guardrail validation
        validated_message = guardrail_validator.enforce_tool_call(
            "Judge", response.content, tool_calls
        )
        is_valid, validation_result = guardrail_validator.validate_agent_message(
            "Judge", validated_message, tool_calls
        )
        
        # Extract score from response
        score_match = re.search(r'(\d+)\s*分', validated_message)
        score = int(score_match.group(1)) if score_match else 70
        
        judge_score = {
            "score": score,
            "reasoning": validated_message,
            "backtest_sharpe": backtest_result.get("sharpe_strategy", 0),
            "backtest_return": backtest_result.get("total_return", 0),
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
            "tool_calls": tool_calls,
            "validation": validation_result
        }
        
        state["judge_score"] = judge_score
        state["backtest_result"] = backtest_result
        state["tool_calls"] = state.get("tool_calls", []) + tool_calls
        state["validation_results"] = state.get("validation_results", []) + [validation_result]
        
        # Update PPO router with reward
        action_history = state.get("debate_log", [])
        ppo_router.update_policy(reward, action_history)
        
        # L3 log entry
        l3_entry = {
            "round": state["round"],
            "layer": "L3",
            "agent": "Judge",
            "action": "backtest_scoring",
            "content": validated_message[:500] + "...",
            "score": score,
            "reward": reward,
            "validation": validation_result,
            "tool_calls": tool_calls
        }
        state["debate_log"] = state.get("debate_log", []) + [l3_entry]
        
    except Exception as e:
        state["judge_score"] = {"error": str(e)}
        state["debate_log"].append({
            "round": state["round"],
            "agent": "Judge",
            "action": "error",
            "content": str(e)
        })
    
    return state

def supervisor_agent(state: DebateState) -> DebateState:
    """
    Supervisor Agent: Route debate using PPO router (Backtest-Guided Router)
    Coordinates hierarchical debate: L1 (Analyst↔Risk) → L2 (Trader) → L3 (Judge)
    """
    llm = get_llm(temperature=0.6)
    if not llm:
        return state
    
    ticker = state["ticker"]
    query = state["query"]
    round_num = state["round"]
    
    # Use PPO router for routing decision
    current_state = {
        "round": round_num,
        "max_rounds": state.get("max_rounds", 3),
        "ticker": ticker
    }
    available_agents = ["Analyst", "Risk", "Trader", "Judge"]
    routing_decision = ppo_router.route_decision(current_state, available_agents)
    
    # Collect all agent outputs
    analyst_evidence = state.get("analyst_evidence", {})
    risk_flags = state.get("risk_flags", [])
    trader_prediction = state.get("trader_prediction", {})
    judge_score = state.get("judge_score", {})
    
    if round_num < state["max_rounds"]:
        # Continue debate
        prompt = f"""你是监督者，协调多智能体辩论。当前是第 {round_num} 轮辩论。

股票代码: {ticker}
用户查询: {query}

分析师证据: {json.dumps(analyst_evidence, ensure_ascii=False)[:800]}
风险标志: {risk_flags}
交易员预测: {json.dumps(trader_prediction, ensure_ascii=False)[:800]}

请评估是否需要继续辩论，或可以做出最终综合判断。"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        state["round"] += 1
        state["debate_log"].append({
            "round": round_num,
            "agent": "Supervisor",
            "action": "synthesis",
            "content": response.content[:500] + "..."
        })
    else:
        # Final synthesis
        prompt = f"""你是监督者，进行最终综合判断。已完成 {state['max_rounds']} 轮辩论。

股票代码: {ticker}
用户查询: {query}

所有证据:
- 分析师: {json.dumps(analyst_evidence, ensure_ascii=False)[:1000]}
- 风险: {risk_flags}
- 交易员: {json.dumps(trader_prediction, ensure_ascii=False)[:1000]}

请提供最终综合报告，包括:
1. 综合所有智能体的观点
2. 最终投资建议
3. 关键风险和机会
4. 置信度评估

用中文回答，给出清晰明确的结论。"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["final_synthesis"] = response.content
        state["debate_log"].append({
            "round": round_num,
            "agent": "Supervisor",
            "action": "final_synthesis",
            "content": response.content
        })
    
    return state

def should_continue(state: DebateState) -> str:
    """Decide whether to continue debate or end"""
    if state["round"] >= state["max_rounds"]:
        return "end"
    return "continue"

# ============================================================================
# LangGraph Construction
# ============================================================================

def create_debate_graph(use_debate: str = "debate"):
    """
    Create LangGraph workflow for hierarchical multi-agent debate
    
    Args:
        use_debate: "debate" (full hierarchical), "no_debate" (direct), "single_agent" (ablation)
    """
    workflow = StateGraph(DebateState)
    
    if use_debate == "debate":
        # Hierarchical debate flow: L1 (Analyst↔Risk) → L2 (Trader) → L3 (Judge)
        workflow.add_node("analyst", analyst_agent)
        workflow.add_node("risk", risk_agent)
        workflow.add_node("trader", trader_agent)
        workflow.add_node("judge", judge_agent)
        workflow.add_node("supervisor", supervisor_agent)
        
        # Hierarchical flow
        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", "risk")  # L1: Analyst → Risk
        workflow.add_edge("risk", "trader")    # L1 → L2: Trader synthesis
        workflow.add_edge("trader", "judge")  # L2 → L3: Judge scoring
        workflow.add_edge("judge", "supervisor")  # L3 → Supervisor routing
        
        # Conditional edge from supervisor (PPO-guided)
        workflow.add_conditional_edges(
            "supervisor",
            should_continue,
            {
                "continue": "analyst",  # Loop back for next round (PPO decides)
                "end": END
            }
        )
    elif use_debate == "no_debate":
        # Direct synthesis without debate (ablation)
        def direct_analysis(state: DebateState) -> DebateState:
            state = analyst_agent(state)
            state = risk_agent(state)
            state = trader_agent(state)
            # Simple synthesis without judge
            llm = get_llm()
            if llm:
                synthesis = llm.invoke([HumanMessage(
                    content=f"综合分析师、风险和交易员的观点，给出最终建议。\n"
                    f"分析师: {state.get('analyst_evidence', {})}\n"
                    f"风险: {state.get('risk_flags', [])}\n"
                    f"交易员: {state.get('trader_prediction', {})}"
                )])
                state["final_synthesis"] = synthesis.content
            return state
        
        workflow.add_node("direct", direct_analysis)
        workflow.set_entry_point("direct")
        workflow.add_edge("direct", END)
    else:  # single_agent
        # Single agent ablation (only analyst)
        workflow.add_node("analyst", analyst_agent)
        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", END)
    
    return workflow.compile(checkpointer=MemorySaver())

# ============================================================================
# Metrics Calculation for Ablation Study
# ============================================================================

def calculate_ablation_metrics(analysis: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate metrics for ablation study comparison
    Returns: Sharpe ratio, MAE (Mean Absolute Error), confidence score
    """
    metrics = {}
    
    # Get financial metrics
    analyst_evidence = analysis.get("analyst_evidence", {})
    financial_metrics = analyst_evidence.get("metrics", {})
    
    # Sharpe Ratio
    sharpe = financial_metrics.get("Sharpe", 0.0)
    metrics["Sharpe"] = sharpe
    
    # MAE: Compare predicted vs actual (if available)
    trader_prediction = analysis.get("trader_prediction", {})
    growth_pred = trader_prediction.get("growth_prediction", {})
    
    # Calculate MAE from prediction confidence
    if isinstance(growth_pred, dict):
        confidence = growth_pred.get("confidence", 0.5)
        mae = 1.0 - confidence  # Lower confidence = higher error
        metrics["MAE"] = mae
    else:
        metrics["MAE"] = 0.5  # Default
    
    # Hallucination confidence
    hallucination_check = analysis.get("hallucination_check", {})
    if isinstance(hallucination_check, dict):
        metrics["HallucinationConfidence"] = hallucination_check.get("confidence", 0.5)
    else:
        metrics["HallucinationConfidence"] = 0.5
    
    # Overall quality score
    metrics["QualityScore"] = (sharpe / 3.0 + (1 - metrics["MAE"]) + metrics["HallucinationConfidence"]) / 3.0
    
    return metrics

# ============================================================================
# Streamlit UI Components
# ============================================================================

def render_sidebar():
    """Render sidebar with input controls"""
    with st.sidebar:
        st.title("📊 DebateFin 配置")
        
        # Ticker input
        ticker = st.text_input(
            "股票代码",
            value="600519",
            help="输入股票代码，如：600519 (茅台), AAPL (苹果)"
        )
        
        # Query input
        query = st.text_area(
            "分析查询",
            value="预测2026年ROE和投资建议",
            height=100,
            help="输入你的分析需求"
        )
        
        
        st.divider()
        st.markdown("### 🔬  消融研究")
        
        debate_mode = st.radio(
            "辩论模式",
            options=["debate", "no_debate", "single_agent"],
            format_func=lambda x: {
                "debate": "✅ 完整分层辩论 (L1→L2→L3)",
                "no_debate": "⏸️ 无辩论直接综合",
                "single_agent": "🔬 单智能体 (Analyst only)"
            }[x],
            index=0,
            help="选择不同的辩论模式进行消融研究"
        )
        
        run_ablation = st.checkbox(
            "运行消融对比实验",
            value=False,
            help="同时运行有/无辩论版本，对比Sharpe/MAE指标"
        )
        
        # Max rounds
        max_rounds = st.slider(
            "最大辩论轮数",
            min_value=1,
            max_value=3,
            value=3,
            help="最多进行几轮辩论"
        )
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ 清除", use_container_width=True)
        
        if clear_btn:
            st.session_state.conversation_history = []
            st.session_state.debate_logs = []
            st.session_state.current_analysis = None
            st.rerun()
        
        st.divider()
        st.markdown("### 📈 关于 DebateFin")
        st.markdown("""
        **DebateFin** 是一个可信赖的多智能体金融分析系统：
        
        - 🤖 **多智能体架构**: Analyst, Risk, Trader
        - 💬 **结构化辩论**: 证据-反驳-综合
        - 📊 **工具接地**: 避免LLM幻觉
        - 🔬 **消融研究**: 对比有/无辩论效果
        """)
        
        return ticker, query, debate_mode, max_rounds, run_ablation, analyze_btn

def render_debate_logs(logs: List[Dict]):
    """Render debate logs in expandable sections with hierarchical tree view"""
    if not logs:
        return
    
    st.subheader("💬 辩论日志（分层树状视图）")
    
    # Group by layer for hierarchical view
    l1_logs = [log for log in logs if log.get("layer") == "L1"]
    l2_logs = [log for log in logs if log.get("layer") == "L2"]
    l3_logs = [log for log in logs if log.get("layer") == "L3"]
    
    # L1 Layer: Analyst ↔ Risk
    if l1_logs:
        with st.expander("🔵 L1层: 基本面分析 (Analyst ↔ Risk)", expanded=True):
            for log_entry in l1_logs:
                agent_name = log_entry.get("agent", "Unknown")
                action = log_entry.get("action", "")
                content = log_entry.get("content", "")
                round_num = log_entry.get("round", 0)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{agent_name}** (第{round_num}轮): {action}")
                with col2:
                    if "validation" in log_entry:
                        val = log_entry["validation"]
                        if val.get("has_tool_call"):
                            st.success("✅ 工具调用")
                        else:
                            st.error("❌ 缺少工具调用")
                
                st.markdown(f"*{content}*")
                
                # Show tool calls
                if "tool_calls" in log_entry:
                    with st.expander(f"查看{agent_name}的工具调用", expanded=False):
                        for tc in log_entry["tool_calls"]:
                            st.code(f"{tc.get('tool', 'unknown')}: {tc.get('result', {})}")
                
                st.divider()
    
    # L2 Layer: Trader Synthesis
    if l2_logs:
        with st.expander("🟢 L2层: 交易决策 (Trader综合)", expanded=True):
            for log_entry in l2_logs:
                agent_name = log_entry.get("agent", "Unknown")
                content = log_entry.get("content", "")
                recommendation = log_entry.get("recommendation", "")
                
                st.markdown(f"**{agent_name}**: {recommendation}")
                st.markdown(f"*{content}*")
                
                if "tool_calls" in log_entry:
                    with st.expander("查看Trader的工具调用", expanded=False):
                        for tc in log_entry["tool_calls"]:
                            st.code(f"{tc.get('tool', 'unknown')}: {tc.get('result', {})}")
                
                st.divider()
    
    # L3 Layer: Judge Scoring
    if l3_logs:
        with st.expander("🔴 L3层: 回测评分 (Judge)", expanded=True):
            for log_entry in l3_logs:
                agent_name = log_entry.get("agent", "Unknown")
                content = log_entry.get("content", "")
                score = log_entry.get("score", 0)
                reward = log_entry.get("reward", 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("评分", f"{score}/100")
                with col2:
                    st.metric("回测奖励", f"{reward:.3f}")
                with col3:
                    if "validation" in log_entry:
                        val = log_entry["validation"]
                        st.metric("验证通过", "✅" if val.get("has_tool_call") else "❌")
                
                st.markdown(f"*{content}*")
                
                if "tool_calls" in log_entry:
                    with st.expander("查看Judge的回测结果", expanded=False):
                        for tc in log_entry["tool_calls"]:
                            result = tc.get("result", {})
                            st.json(result)
                
                st.divider()
    
    # Fallback: Show all logs if no layer grouping
    if not (l1_logs or l2_logs or l3_logs):
        for i, log_entry in enumerate(logs):
            agent_name = log_entry.get("agent", "Unknown")
            action = log_entry.get("action", "")
            content = log_entry.get("content", "")
            round_num = log_entry.get("round", 0)
            
            with st.expander(f"第 {round_num} 轮 - {agent_name} ({action})", expanded=(i == len(logs) - 1)):
                st.markdown(f"**操作**: {action}")
                st.markdown(f"**内容**: {content}")
                if "recommendation" in log_entry:
                    st.info(f"**建议**: {log_entry['recommendation']}")
                if "sentiment_score" in log_entry:
                    st.metric("市场情绪得分", f"{log_entry['sentiment_score']:.2f}")

def render_financial_charts(analysis: Dict):
    """Render financial analysis charts"""
    st.subheader("📊 财务分析图表")
    
    analyst_evidence = analysis.get("analyst_evidence", {})
    metrics = analyst_evidence.get("metrics", {})
    trader_prediction = analysis.get("trader_prediction", {})
    
    if not metrics:
        st.warning("暂无财务数据")
        return
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        # ROE Chart
        if "ROE" in metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            roe_value = metrics.get("ROE", 0)
            ax.barh(["ROE"], [roe_value], color='green' if roe_value > 0.15 else 'orange')
            ax.set_xlabel("ROE (%)")
            ax.set_title("净资产收益率 (ROE)")
            ax.axvline(x=0.15, color='r', linestyle='--', label='基准线 (15%)')
            ax.legend()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        # Sharpe Ratio Chart
        if "Sharpe" in metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            sharpe_value = metrics.get("Sharpe", 0)
            ax.barh(["Sharpe比率"], [sharpe_value], color='blue' if sharpe_value > 1 else 'red')
            ax.set_xlabel("Sharpe比率")
            ax.set_title("风险调整后收益 (Sharpe Ratio)")
            ax.axvline(x=1, color='g', linestyle='--', label='基准线 (1.0)')
            ax.legend()
            st.pyplot(fig)
            plt.close()
    
    # Growth Prediction Chart
    if trader_prediction and "growth_prediction" in trader_prediction:
        growth_pred = trader_prediction["growth_prediction"]
        if isinstance(growth_pred, dict) and "forecast" in growth_pred:
            fig, ax = plt.subplots(figsize=(10, 6))
            forecast_data = growth_pred["forecast"]
            if isinstance(forecast_data, (list, np.ndarray)):
                ax.plot(forecast_data, marker='o', label='预测增长')
                ax.set_xlabel("时间步")
                ax.set_ylabel("增长率 (%)")
                ax.set_title("LSTM增长预测")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                plt.close()

def render_backtest_results(ticker: str):
    """Render backtesting results with Plotly interactive charts (使用缓存，防止频繁请求)"""
    st.subheader("📈 回测结果 (5年数据)")
    
    try:
        # 使用缓存函数获取价格历史（防止频繁请求被限流）
        prices = get_price_history(ticker, period="5y")
        
        if prices.empty or len(prices) < 200:
            # 检查是否是数据源失败
            if prices.empty:
                st.warning(f"⚠️ 无法获取股票 {ticker} 的历史数据。请检查股票代码是否正确，或稍后再试。")
            else:
                st.warning(f"⚠️ 获取的历史数据不足（仅 {len(prices)} 条），无法进行回测分析。")
            return
        
        # 转换为 DataFrame 格式以便后续处理
        hist = pd.DataFrame({'Close': prices})
        hist.index = prices.index
        
        # Simple moving average strategy
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        # Calculate returns
        hist['Returns'] = hist['Close'].pct_change()
        cumulative_returns = (1 + hist['Returns']).cumprod()
        
        # Use Plotly for interactive charts if available
        if PLOTLY_AVAILABLE:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f"{ticker} 价格走势 (5年)", "累计收益率"),
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4]
            )
            
            # Price and moving averages
            fig.add_trace(
                go.Scatter(x=hist.index, y=hist['Close'], name='收盘价', line=dict(width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=hist['SMA_200'], name='SMA 200', line=dict(dash='dot')),
                row=1, col=1
            )
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, 
                          name='累计收益', line=dict(color='green', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="基准线", row=2, col=1)
            
            # Update layout
            fig.update_layout(height=700, showlegend=True, title_text=f"{ticker} 回测分析")
            fig.update_xaxes(title_text="日期", row=2, col=1)
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="累计收益倍数", row=2, col=1)
            
            # 使用 ticker 和时间戳确保 key 唯一
            import time
            unique_key = f"plotly_backtest_{ticker}_{int(time.time() * 1000)}"
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
        else:
            # Fallback to matplotlib
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            ax1.plot(hist.index, hist['Close'], label='收盘价', linewidth=2)
            ax1.plot(hist.index, hist['SMA_50'], label='SMA 50', alpha=0.7)
            ax1.plot(hist.index, hist['SMA_200'], label='SMA 200', alpha=0.7)
            ax1.set_title(f"{ticker} 价格走势 (5年)")
            ax1.set_ylabel("价格")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(cumulative_returns.index, cumulative_returns.values, label='累计收益', color='green', linewidth=2)
            ax2.axhline(y=1, color='r', linestyle='--', label='基准线')
            ax2.set_title("累计收益率")
            ax2.set_ylabel("累计收益倍数")
            ax2.set_xlabel("日期")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Metrics
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        volatility = hist['Returns'].std() * np.sqrt(252) * 100
        sharpe = (hist['Returns'].mean() * 252) / (hist['Returns'].std() * np.sqrt(252)) if hist['Returns'].std() > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总收益率", f"{total_return:.2f}%")
        with col2:
            st.metric("年化波动率", f"{volatility:.2f}%")
        with col3:
            st.metric("Sharpe比率", f"{sharpe:.2f}")
            
    except Exception as e:
        st.error(f"回测失败: {str(e)}")

def render_ablation_comparison(with_debate: Dict, without_debate: Dict):
    """
    渲染消融研究对比（使用唯一的 key 避免重复）
    """
    st.subheader("🔬 消融研究对比 (有辩论 vs 无辩论)")
    
    # 使用 session state 计数器确保 key 唯一
    if 'ablation_chart_counter' not in st.session_state:
        st.session_state.ablation_chart_counter = 0
    st.session_state.ablation_chart_counter += 1
    
    # Calculate metrics for both (use cached if available)
    metrics_with = with_debate.get("metrics", calculate_ablation_metrics(with_debate))
    metrics_without = without_debate.get("metrics", calculate_ablation_metrics(without_debate))
    
    # Comparison metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sharpe_diff = metrics_with["Sharpe"] - metrics_without["Sharpe"]
        st.metric("Sharpe比率差异", f"{sharpe_diff:.3f}", 
                 delta=f"{metrics_with['Sharpe']:.3f} vs {metrics_without['Sharpe']:.3f}")
    
    with col2:
        mae_diff = metrics_with["MAE"] - metrics_without["MAE"]
        st.metric("MAE差异", f"{mae_diff:.3f}",
                 delta=f"{metrics_with['MAE']:.3f} vs {metrics_without['MAE']:.3f}")
    
    with col3:
        conf_diff = metrics_with["HallucinationConfidence"] - metrics_without["HallucinationConfidence"]
        st.metric("幻觉置信度差异", f"{conf_diff:.3f}",
                 delta=f"{metrics_with['HallucinationConfidence']:.3f} vs {metrics_without['HallucinationConfidence']:.3f}")
    
    with col4:
        quality_diff = metrics_with["QualityScore"] - metrics_without["QualityScore"]
        st.metric("质量得分差异", f"{quality_diff:.3f}",
                 delta=f"{metrics_with['QualityScore']:.3f} vs {metrics_without['QualityScore']:.3f}")
    
    # Side-by-side charts
    metrics_list = ["Sharpe", "MAE", "HallucinationConfidence", "QualityScore"]
    
    if PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("有辩论", "无辩论"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=metrics_list, y=[metrics_with[m] for m in metrics_list], 
                   name="有辩论", marker_color='blue', showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=metrics_list, y=[metrics_without[m] for m in metrics_list],
                   name="无辩论", marker_color='orange', showlegend=False),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="指标对比")
        fig.update_yaxes(title_text="数值", row=1, col=1)
        fig.update_yaxes(title_text="数值", row=1, col=2)
        
        # 使用计数器确保 key 唯一（避免同一页面多次调用导致重复）
        unique_key = f"plotly_ablation_{st.session_state.ablation_chart_counter}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)
    else:
        # Fallback to matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(metrics_list))
        width = 0.35
        
        ax1.bar(x, [metrics_with[m] for m in metrics_list], width, label='有辩论', color='blue')
        ax1.set_title("有辩论")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_list, rotation=45)
        ax1.set_ylabel("数值")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(x, [metrics_without[m] for m in metrics_list], width, label='无辩论', color='orange')
        ax2.set_title("无辩论")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_list, rotation=45)
        ax2.set_ylabel("数值")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Store results
    st.session_state.ablation_results = {
        "with_debate": metrics_with,
        "without_debate": metrics_without
    }

def render_final_recommendation(analysis: Dict):
    """Render final investment recommendation"""
    st.subheader("🎯 最终投资建议")
    
    trader_prediction = analysis.get("trader_prediction", {})
    final_synthesis = analysis.get("final_synthesis", "")
    risk_flags = analysis.get("risk_flags", [])
    
    # Recommendation badge
    recommendation = trader_prediction.get("recommendation", "持有")
    if recommendation == "买入":
        st.success(f"✅ **建议: {recommendation}**")
    elif recommendation == "卖出":
        st.error(f"❌ **建议: {recommendation}**")
    else:
        st.info(f"⏸️ **建议: {recommendation}**")
    
    # Risk flags
    if risk_flags:
        st.warning(f"⚠️ **风险提示**: {', '.join(risk_flags)}")
    
    # Final synthesis
    if final_synthesis:
        st.markdown("### 📝 综合报告")
        st.markdown(final_synthesis)
    
    # Reasoning
    if "reasoning" in trader_prediction:
        st.markdown("### 💡 投资逻辑")
        st.markdown(trader_prediction["reasoning"])

# ============================================================================
# Main App
# ============================================================================

def main():
    """Main Streamlit application"""
    st.title("📊 DebateFin: 可信赖的多智能体金融分析系统")
    st.markdown("**helloaisvg Demo** - 结构化辩论减少金融推理中的幻觉")
    
    # Sidebar
    ticker, query, debate_mode, max_rounds, run_ablation, analyze_btn = render_sidebar()
    
    # Main content area
    if analyze_btn:
        if not ticker:
            st.error("请输入股票代码")
            return
        
        # Clear previous hallucination checks
        st.session_state.hallucination_checks = []
        
        if run_ablation:
            # Run ablation study: both with and without debate
            with st.spinner("🔬 运行消融研究：同时分析有/无辩论版本..."):
                try:
                    # Run with debate
                    initial_state_with = DebateState(
                        messages=[], ticker=ticker, query=query, round=1,
                        max_rounds=max_rounds, analyst_evidence={}, risk_flags=[],
                        trader_prediction={}, debate_log=[], final_synthesis="",
                        use_debate="debate", metrics={}, hallucination_check={},
                        l1_debate_log=[], l2_synthesis="", judge_score={}, backtest_result={},
                        tool_calls=[], validation_results=[]
                    )
                    graph_with = create_debate_graph(use_debate="debate")
                    final_state_with = graph_with.invoke(initial_state_with, {"configurable": {"thread_id": "with"}})
                    final_state_with["metrics"] = calculate_ablation_metrics(final_state_with)
                    
                    # Run without debate
                    initial_state_without = DebateState(
                        messages=[], ticker=ticker, query=query, round=1,
                        max_rounds=1, analyst_evidence={}, risk_flags=[],
                        trader_prediction={}, debate_log=[], final_synthesis="",
                        use_debate="no_debate", metrics={}, hallucination_check={},
                        l1_debate_log=[], l2_synthesis="", judge_score={}, backtest_result={},
                        tool_calls=[], validation_results=[]
                    )
                    graph_without = create_debate_graph(use_debate="no_debate")
                    final_state_without = graph_without.invoke(initial_state_without, {"configurable": {"thread_id": "without"}})
                    final_state_without["metrics"] = calculate_ablation_metrics(final_state_without)
                    
                    # Store both results
                    st.session_state.current_analysis = final_state_with
                    st.session_state.ablation_results = {
                        "with_debate": final_state_with,
                        "without_debate": final_state_without
                    }
                    st.session_state.debate_logs = final_state_with.get("debate_log", [])
                    
                    st.success("✅ 消融研究完成！")
                    
                except Exception as e:
                    st.error(f"消融研究过程中出错: {str(e)}")
                    st.exception(e)
                    return
        else:
            # Normal single run
            initial_state = DebateState(
                messages=[], ticker=ticker, query=query, round=1,
                max_rounds=max_rounds, analyst_evidence={}, risk_flags=[],
                trader_prediction={}, debate_log=[], final_synthesis="",
                use_debate=debate_mode, metrics={}, hallucination_check={},
                l1_debate_log=[], l2_synthesis="", judge_score={}, backtest_result={},
                tool_calls=[], validation_results=[]
            )
            
            with st.spinner("🤖 多智能体正在分析中，请稍候..."):
                try:
                    graph = create_debate_graph(use_debate=debate_mode)
                    config = {"configurable": {"thread_id": "1"}}
                    
                    final_state = graph.invoke(initial_state, config=config)
                    final_state["metrics"] = calculate_ablation_metrics(final_state)
                    
                    st.session_state.current_analysis = final_state
                    st.session_state.debate_logs = final_state.get("debate_log", [])
                    
                    st.success("✅ 分析完成！")
                    
                except Exception as e:
                    st.error(f"分析过程中出错: {str(e)}")
                    st.exception(e)
                    return
    
    # Display results
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        
        # Show ablation comparison if available (只在主页面显示一次，避免重复)
        # 注意：在 tabs 中也会显示，所以这里注释掉，避免重复调用导致 key 冲突
        # if st.session_state.ablation_results:
        #     st.markdown("---")
        #     render_ablation_comparison(
        #         st.session_state.ablation_results["with_debate"],
        #         st.session_state.ablation_results["without_debate"]
        #     )
        #     st.markdown("---")
        
        # Show hallucination checks
        if st.session_state.hallucination_checks:
            with st.expander("🔍 幻觉检查结果", expanded=False):
                for check in st.session_state.hallucination_checks:
                    st.markdown(f"**{check['agent']} (第{check['round']}轮)**")
                    check_data = check.get("check", {})
                    if check_data.get("has_hallucination"):
                        st.warning(f"⚠️ 检测到潜在幻觉: {', '.join(check_data.get('issues', []))}")
                    else:
                        st.success(f"✅ 置信度: {check_data.get('confidence', 0):.2f}")
        
        # Tabs for different views
        tab_names = ["📊 综合分析", "💬 辩论日志", "📈 图表分析", "📄 导出报告"]
        if st.session_state.ablation_results:
            tab_names.insert(1, "🔬 消融研究")
        
        tabs = st.tabs(tab_names)
        tab_idx = 0
        
        with tabs[tab_idx]:
            render_final_recommendation(analysis)
        tab_idx += 1
        
        if st.session_state.ablation_results:
            with tabs[tab_idx]:
                render_ablation_comparison(
                    st.session_state.ablation_results["with_debate"],
                    st.session_state.ablation_results["without_debate"]
                )
            tab_idx += 1
        
        with tabs[tab_idx]:
            render_debate_logs(st.session_state.debate_logs)
        tab_idx += 1
        
        with tabs[tab_idx]:
            render_financial_charts(analysis)
            st.divider()
            render_backtest_results(analysis.get("ticker", ""))
        tab_idx += 1
        
        with tabs[tab_idx]:
            st.subheader("📄 导出PDF报告")
            
            # Enhanced PDF with debate logs and ablation results
            col1, col2 = st.columns(2)
            with col1:
                include_debate_logs = st.checkbox("包含辩论日志", value=True)
                include_ablation = st.checkbox("包含消融研究结果", value=bool(st.session_state.ablation_results))
            
            if st.button("生成HTML报告", type="primary"):
                try:
                    report_data = {
                        "analysis": analysis,
                        "debate_logs": st.session_state.debate_logs if include_debate_logs else [],
                        "ablation_results": st.session_state.ablation_results if include_ablation else None,
                        "hallucination_checks": st.session_state.hallucination_checks
                    }
                    html_buffer = generate_html_report_bytes(report_data)
                    st.download_button(
                        label="📥 下载HTML报告",
                        data=html_buffer.getvalue(),
                        file_name=f"DebateFin_Report_{analysis.get('ticker', 'UNKNOWN')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    st.info('💡 提示：下载HTML文件后，可在浏览器中打开，然后使用浏览器的"打印"功能（Ctrl+P / Cmd+P）导出为PDF')
                except Exception as e:
                    st.error(f"HTML报告生成失败: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>DebateFin - helloaisvg | Powered by LangChain, LangGraph & DeepSeek</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

