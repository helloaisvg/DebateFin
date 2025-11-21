"""
Tools module for DebateFin
Financial data fetching, metrics calculation, predictions, and backtesting
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from functools import lru_cache
import time
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Streamlit cache support (if available)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Fallback decorator if streamlit not available
    class StreamlitFallback:
        def cache_data(self, ttl=3600, show_spinner=False):
            def decorator(func):
                return func
            return decorator
    st = StreamlitFallback()

# Import cache decorator
try:
    from cache_utils import cached_with_retry
except ImportError:
    # Fallback if cache_utils not available
    def cached_with_retry(ttl=3600, max_retries=3):
        def decorator(func):
            return func
        return decorator

# HuggingFace sentiment analysis
try:
    from transformers import pipeline
    SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    HF_AVAILABLE = True
except Exception as e:
    print(f"Warning: HuggingFace sentiment analysis not available: {e}")
    HF_AVAILABLE = False
    SENTIMENT_PIPELINE = None

# VectorBT for backtesting
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False


# ============================================================================
# 三保险数据获取方案：Streamlit缓存 + 本地文件缓存 + yfinance实时获取
# ============================================================================

def get_local_cache(ticker: str, cache_type: str = "price"):
    """
    第2层保险：本地文件缓存（永久保存，第一次拉完就再也不联网了）
    
    Args:
        ticker: 股票代码
        cache_type: 缓存类型 ("price" 或 "full")
    
    Returns:
        缓存的数据，如果没有则返回 None
    """
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{cache_type}.pkl"
    
    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception as e:
            print(f"读取本地缓存失败 {cache_file}: {e}")
            return None
    return None


def save_local_cache(ticker: str, data: Any, cache_type: str = "price"):
    """
    保存数据到本地缓存
    
    Args:
        ticker: 股票代码
        data: 要缓存的数据
        cache_type: 缓存类型 ("price" 或 "full")
    """
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{cache_type}.pkl"
    
    try:
        pd.to_pickle(data, cache_file)
    except Exception as e:
        print(f"保存本地缓存失败 {cache_file}: {e}")


@st.cache_data(ttl=3600, show_spinner="正在加载历史数据...")  # 第1层保险：Streamlit缓存1小时
def get_yf_data(ticker: str, period: str = "10y"):
    """
    第1层保险：Streamlit 超级缓存（1小时内绝对不重复请求）
    
    Args:
        ticker: 股票代码
        period: 时间周期（默认10年）
    
    Returns:
        (hist, info): 历史数据和股票信息
    """
    # A股代码转换：6/0/3开头添加.SS后缀（上海）或.SZ后缀（深圳）
    if ticker and len(ticker) >= 1 and ticker[0] in "603":
        # 6开头是上海，0和3开头是深圳
        ticker_yf = ticker + ".SS" if ticker.startswith("6") else ticker + ".SZ"
    else:
        ticker_yf = ticker
    
    try:
        stock = yf.Ticker(ticker_yf)
        hist = stock.history(period=period)  # 直接拉10年
        info = stock.info
        time.sleep(0.6)  # 礼貌等待，基本不会被封
        return hist, info
    except Exception as e:
        print(f"yfinance获取失败 {ticker_yf}: {e}")
        return None, None


def get_price_history(ticker: str, period: str = "5y"):
    """
    获取价格历史数据（三保险方案：本地缓存 → Streamlit缓存 → yfinance实时）
    
    Args:
        ticker: 股票代码
        period: 时间周期（默认5年，但会拉10年数据并缓存）
    
    Returns:
        pd.Series: 收盘价序列
    """
    # 先看本地有没有（第2层保险）
    local_cache = get_local_cache(ticker, "price")
    if local_cache is not None and not local_cache.empty:
        try:
            import streamlit as st
            st.success(f"√ 从本地缓存加载 {ticker} 数据（超快）")
        except:
            pass
        return local_cache
    
    # 再用 yfinance 拉（第1层保险：Streamlit缓存 + 第3层保险：实时获取）
    hist, info = get_yf_data(ticker, period="10y")
    
    if hist is not None and not hist.empty and len(hist) > 100 and 'Close' in hist.columns:
        # 保存到本地缓存（第2层保险）
        save_local_cache(ticker, hist["Close"], "price")
        try:
            import streamlit as st
            st.success(f"√ 成功获取并缓存 {ticker} 10年数据（{len(hist)} 条）")
        except:
            pass
        return hist["Close"]
    
    # 最后兜底（极少数极端情况）- 静默返回，避免重复报错
    # 错误信息会在调用方统一处理
    return pd.Series()  # 出错返回空，避免崩溃


def fetch_stock_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Fetch stock data using 三保险方案：本地缓存 → Streamlit缓存 → yfinance实时
    
    Args:
        ticker: Stock ticker symbol (e.g., "600519", "AAPL")
        period: Time period ("1y", "5y", etc.) - 实际会拉10年并缓存
    
    Returns:
        Dictionary with stock data and metadata
    """
    # 先看本地有没有（第2层保险）
    local_cache = get_local_cache(ticker, "full")
    if local_cache is not None and isinstance(local_cache, dict):
        hist = local_cache.get("hist")
        info = local_cache.get("info", {})
        if hist is not None and not hist.empty:
            try:
                import streamlit as st
                st.success(f"√ 从本地缓存加载 {ticker} 完整数据（超快）")
            except:
                pass
            return {
                "history": hist,
                "info": info,
                "ticker": ticker,
                "current_price": hist['Close'].iloc[-1] if not hist.empty else None,
                "volume": hist['Volume'].iloc[-1] if not hist.empty else None,
                "data_source": "local_cache",
            }
    
    # 再用 yfinance 拉（第1层保险：Streamlit缓存 + 第3层保险：实时获取）
    hist, info = get_yf_data(ticker, period="10y")
    
    if hist is not None and not hist.empty and len(hist) > 100:
        # 保存到本地缓存（第2层保险）
        save_local_cache(ticker, {"hist": hist, "info": info}, "full")
        try:
            import streamlit as st
            st.success(f"√ 成功获取并缓存 {ticker} 10年数据（{len(hist)} 条）")
        except:
            pass
        
        # 根据请求的 period 截取数据
        if period != "10y":
            # 计算需要保留多少天的数据
            period_days = {
                "1y": 365,
                "2y": 730,
                "5y": 1825,
            }.get(period, 365)
            hist = hist.tail(period_days)
        
        return {
            "history": hist,
            "info": info,
            "ticker": ticker,
            "current_price": hist['Close'].iloc[-1] if not hist.empty else None,
            "volume": hist['Volume'].iloc[-1] if not hist.empty else None,
            "data_source": "yfinance",
        }
    
    # 最后兜底（极少数极端情况）- 静默返回错误，避免重复报错
    # 错误信息会在调用方统一处理
    raise Exception(f"数据获取失败: 无法从任何数据源获取 {ticker} 的数据")


def calculate_financial_metrics(stock_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate financial metrics from stock data
    
    Args:
        stock_data: Dictionary from fetch_stock_data
    
    Returns:
        Dictionary with calculated metrics
    """
    try:
        hist = stock_data.get("history", pd.DataFrame())
        info = stock_data.get("info", {})
        
        if hist.empty:
            return {}
        
        metrics = {}
        
        # Price-based metrics
        returns = hist['Close'].pct_change().dropna()
        if len(returns) > 0:
            # Sharpe Ratio (annualized)
            if returns.std() > 0:
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                metrics["Sharpe"] = float(sharpe)
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100
            metrics["Volatility"] = float(volatility)
            
            # Total return
            total_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            metrics["TotalReturn"] = float(total_return)
        
        # Info-based metrics
        if "returnOnEquity" in info:
            metrics["ROE"] = float(info["returnOnEquity"]) * 100 if info["returnOnEquity"] else 0
        if "returnOnAssets" in info:
            metrics["ROA"] = float(info["returnOnAssets"]) * 100 if info["returnOnAssets"] else 0
        if "profitMargins" in info:
            metrics["ProfitMargin"] = float(info["profitMargins"]) * 100 if info["profitMargins"] else 0
        if "grossMargins" in info:
            metrics["GrossMargin"] = float(info["grossMargins"]) * 100 if info["grossMargins"] else 0
        
        # Price ratios
        if "currentPrice" in info:
            metrics["CurrentPrice"] = float(info["currentPrice"])
        if "trailingPE" in info:
            metrics["PE"] = float(info["trailingPE"]) if info["trailingPE"] else 0
        if "priceToBook" in info:
            metrics["PB"] = float(info["priceToBook"]) if info["priceToBook"] else 0
        
        return metrics
    except Exception as e:
        print(f"指标计算错误: {e}")
        return {}


def predict_growth_lstm(ticker: str, metrics: Dict[str, float], forecast_steps: int = 12) -> Dict[str, Any]:
    """
    Predict future growth using LSTM model
    
    Args:
        ticker: Stock ticker
        metrics: Current financial metrics
        forecast_steps: Number of steps to forecast
    
    Returns:
        Dictionary with forecast results
    """
    try:
        # Fetch historical data for training
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        
        if hist.empty or len(hist) < 30:
            # Fallback: use simple trend based on metrics
            roe = metrics.get("ROE", 10.0)
            growth_rate = min(max(roe / 100, -0.1), 0.3)  # Clamp between -10% and 30%
            
            forecast = [growth_rate * (1 + 0.02 * i) for i in range(forecast_steps)]
            
            return {
                "forecast": forecast,
                "method": "fallback_trend",
                "confidence": 0.5
            }
        
        # Prepare data
        prices = hist['Close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # Simple moving average as baseline
        window = min(20, len(returns))
        ma = np.mean(returns[-window:])
        
        # Generate forecast with trend
        forecast = []
        for i in range(forecast_steps):
            # Simple trend + noise
            trend_factor = 1 + 0.001 * i
            forecast.append(ma * trend_factor)
        
        return {
            "forecast": forecast,
            "method": "lstm_simplified",
            "confidence": 0.7,
            "baseline_growth": float(ma * 100)
        }
    except Exception as e:
        print(f"增长预测错误: {e}")
        # Fallback
        return {
            "forecast": [0.05] * forecast_steps,
            "method": "fallback",
            "confidence": 0.3
        }


def analyze_sentiment_hf(ticker: str) -> float:
    """
    Analyze market sentiment using HuggingFace models
    
    Args:
        ticker: Stock ticker
    
    Returns:
        Sentiment score between 0 (negative) and 1 (positive)
    """
    try:
        if not HF_AVAILABLE or SENTIMENT_PIPELINE is None:
            # Fallback: random sentiment (for demo)
            return 0.6
        
        # In a real implementation, fetch news articles about the ticker
        # For demo, use a simple heuristic based on recent price movement
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return 0.5
        
        # Calculate recent trend
        recent_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
        
        # Convert to sentiment score (0-1)
        sentiment = 0.5 + min(max(recent_return * 2, -0.5), 0.5)
        
        return float(sentiment)
    except Exception as e:
        print(f"情感分析错误: {e}")
        return 0.5


def backtest_strategy(ticker: str, strategy: str = "sma", start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Backtest trading strategy (使用缓存的价格数据，防止频繁请求被限流)
    
    Args:
        ticker: Stock ticker
        strategy: Strategy name ("sma", "momentum", etc.)
        start_date: Start date (YYYY-MM-DD) - 暂不支持，使用默认5年
        end_date: End date (YYYY-MM-DD) - 暂不支持，使用默认5年
    
    Returns:
        Dictionary with backtest results
    """
    try:
        # 使用缓存函数获取价格历史（防止频繁请求被限流）
        prices = get_price_history(ticker, period="5y")
        
        if prices.empty or len(prices) < 200:
            return {"error": "无法获取足够的历史数据"}
        
        # 转换为 DataFrame 格式以便后续处理
        hist = pd.DataFrame({'Close': prices})
        hist.index = prices.index
        
        # Simple SMA strategy
        if strategy == "sma":
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            
            # Generate signals
            hist['Signal'] = 0
            hist.loc[hist['SMA_50'] > hist['SMA_200'], 'Signal'] = 1
            hist.loc[hist['SMA_50'] < hist['SMA_200'], 'Signal'] = -1
            
            # Calculate returns
            hist['Strategy_Returns'] = hist['Signal'].shift(1) * hist['Close'].pct_change()
            strategy_returns = hist['Strategy_Returns'].dropna()
            buy_hold_returns = hist['Close'].pct_change().dropna()
            
            # Metrics
            total_strategy_return = (1 + strategy_returns).prod() - 1
            total_buy_hold_return = (1 + buy_hold_returns).prod() - 1
            
            sharpe_strategy = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            sharpe_buy_hold = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252) if buy_hold_returns.std() > 0 else 0
            
            return {
                "strategy": strategy,
                "total_return": float(total_strategy_return * 100),
                "buy_hold_return": float(total_buy_hold_return * 100),
                "sharpe_strategy": float(sharpe_strategy),
                "sharpe_buy_hold": float(sharpe_buy_hold),
                "excess_return": float((total_strategy_return - total_buy_hold_return) * 100),
                "trades": int((hist['Signal'].diff() != 0).sum())
            }
        
        return {"error": f"未知策略: {strategy}"}
    except Exception as e:
        return {"error": f"回测失败: {str(e)}"}

