"""
Tool-Enforced Guardrail Validator
Ensures every agent message MUST contain <tool_call> before reasoning
This is the core innovation: Tool-Enforced Debate
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class ToolCallValidator:
    """
    Validator that enforces tool calls before agent reasoning
    Prevents hallucination by requiring evidence grounding
    """
    
    # Tool call patterns
    TOOL_CALL_PATTERNS = [
        r'<tool_call>.*?</tool_call>',
        r'\[tool_call\].*?\[/tool_call\]',
        r'fetch_stock_data|calculate_financial_metrics|analyze_sentiment|backtest_strategy',
        r'yfinance|LSTM|VectorBT',
    ]
    
    # Evidence patterns (numbers, metrics)
    EVIDENCE_PATTERNS = [
        r'ROE[:\s]+[\d.]+',
        r'Sharpe[:\s]+[\d.]+',
        r'ROA[:\s]+[\d.]+',
        r'[\d.]+%',  # Percentages
        r'[\d.]+元',  # Prices
    ]
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, reject messages without tool calls
        """
        self.strict_mode = strict_mode
        self.validation_log = []
    
    def validate_agent_message(self, agent_name: str, message: str, 
                               tool_calls: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that agent message contains tool calls and evidence
        
        Args:
            agent_name: Name of the agent
            message: Agent's message content
            tool_calls: List of tool calls made by agent
        
        Returns:
            Tuple of (is_valid, validation_result)
        """
        result = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "has_tool_call": False,
            "has_evidence": False,
            "tool_call_count": len(tool_calls),
            "evidence_count": 0,
            "violations": [],
            "score": 0.0
        }
        
        # Check for tool calls in message
        has_tool_call_in_text = any(
            re.search(pattern, message, re.IGNORECASE | re.DOTALL)
            for pattern in self.TOOL_CALL_PATTERNS
        )
        
        # Check if actual tool calls were made
        has_actual_tool_call = len(tool_calls) > 0
        
        result["has_tool_call"] = has_tool_call_in_text or has_actual_tool_call
        
        # Check for evidence (numbers, metrics)
        evidence_matches = []
        for pattern in self.EVIDENCE_PATTERNS:
            matches = re.findall(pattern, message, re.IGNORECASE)
            evidence_matches.extend(matches)
        
        result["evidence_count"] = len(evidence_matches)
        result["has_evidence"] = result["evidence_count"] > 0
        
        # Calculate score
        score = 0.0
        if result["has_tool_call"]:
            score += 0.5
        if result["has_evidence"]:
            score += 0.3
        if result["tool_call_count"] > 0:
            score += 0.2 * min(result["tool_call_count"] / 3.0, 1.0)
        
        result["score"] = score
        
        # Check violations
        if self.strict_mode:
            if not result["has_tool_call"]:
                result["violations"].append("缺少工具调用：消息必须包含<tool_call>或实际调用工具")
            if not result["has_evidence"]:
                result["violations"].append("缺少证据：消息必须包含具体数字或指标")
        
        is_valid = len(result["violations"]) == 0 or score >= 0.5
        
        # Log validation
        self.validation_log.append(result)
        
        return is_valid, result
    
    def enforce_tool_call(self, agent_name: str, message: str, 
                          tool_calls: List[Dict[str, Any]]) -> str:
        """
        Enforce tool call requirement - reject or modify message
        
        Args:
            agent_name: Agent name
            message: Original message
            tool_calls: Tool calls made
        
        Returns:
            Modified message or rejection notice
        """
        is_valid, validation = self.validate_agent_message(agent_name, message, tool_calls)
        
        if not is_valid and self.strict_mode:
            # Reject message and require tool call
            rejection = f"""<rejection>
原因: {', '.join(validation['violations'])}
要求: 请先调用工具获取数据，然后再进行推理。
工具调用示例: <tool_call>fetch_stock_data(ticker)</tool_call>
</rejection>"""
            return rejection
        
        # Add tool call annotation if missing
        if not validation["has_tool_call"] and tool_calls:
            tool_call_summary = f"\n<tool_calls_made>\n"
            for tc in tool_calls:
                tool_call_summary += f"- {tc.get('tool', 'unknown')}: {tc.get('result', 'N/A')[:100]}\n"
            tool_call_summary += "</tool_calls_made>\n"
            message = tool_call_summary + message
        
        return message
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get overall validation statistics"""
        if not self.validation_log:
            return {"total": 0, "avg_score": 0.0}
        
        total = len(self.validation_log)
        avg_score = sum(v["score"] for v in self.validation_log) / total
        violations = sum(len(v["violations"]) for v in self.validation_log)
        
        return {
            "total": total,
            "avg_score": avg_score,
            "total_violations": violations,
            "compliance_rate": (total - violations) / total if total > 0 else 0.0
        }


# Global validator instance
guardrail_validator = ToolCallValidator(strict_mode=True)

