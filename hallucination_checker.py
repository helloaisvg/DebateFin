"""
Hallucination checker: Compare LLM output vs tool data
Tool-forced grounding: Every debate turn must call tools before reasoning
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class HallucinationChecker:
    """Check for hallucinations by comparing LLM output with tool data"""
    
    def __init__(self):
        self.tool_data_cache = {}
        self.llm_outputs = []
    
    def register_tool_data(self, ticker: str, data_type: str, data: Any):
        """Register tool-fetched data"""
        key = f"{ticker}:{data_type}"
        self.tool_data_cache[key] = {
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_hallucination(self, llm_output: str, ticker: str, data_types: List[str]) -> Dict[str, Any]:
        """
        Check if LLM output contains hallucinations
        
        Args:
            llm_output: LLM generated text
            ticker: Stock ticker
            data_types: List of data types to check against
        
        Returns:
            Dictionary with hallucination check results
        """
        issues = []
        confidence_scores = []
        
        for data_type in data_types:
            key = f"{ticker}:{data_type}"
            tool_data = self.tool_data_cache.get(key, {}).get("data", {})
            
            if not tool_data:
                issues.append(f"缺少{data_type}的工具数据")
                continue
            
            # Extract numbers from LLM output
            llm_numbers = self._extract_numbers(llm_output)
            
            # Compare with tool data
            if isinstance(tool_data, dict):
                tool_numbers = self._extract_from_dict(tool_data)
            else:
                tool_numbers = []
            
            # Check for mismatches
            mismatches = self._compare_numbers(llm_numbers, tool_numbers, data_type)
            if mismatches:
                issues.extend(mismatches)
            
            # Calculate confidence
            if tool_numbers:
                match_ratio = len([n for n in llm_numbers if any(abs(n - t) < abs(t) * 0.1 for t in tool_numbers)]) / max(len(llm_numbers), 1)
                confidence_scores.append(match_ratio)
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            "has_hallucination": len(issues) > 0,
            "issues": issues,
            "confidence": overall_confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        # Match percentages, decimals, integers
        patterns = [
            r'(\d+\.?\d*)\s*%',  # Percentages
            r'(\d+\.\d+)',       # Decimals
            r'\b(\d+)\b'         # Integers
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.extend([float(m) for m in matches])
        
        return numbers
    
    def _extract_from_dict(self, data: Dict[str, Any]) -> List[float]:
        """Extract numeric values from dictionary"""
        numbers = []
        for value in data.values():
            if isinstance(value, (int, float)):
                numbers.append(float(value))
            elif isinstance(value, dict):
                numbers.extend(self._extract_from_dict(value))
        return numbers
    
    def _compare_numbers(self, llm_numbers: List[float], tool_numbers: List[float], data_type: str) -> List[str]:
        """Compare LLM numbers with tool numbers"""
        mismatches = []
        
        if not tool_numbers:
            return mismatches
        
        # Check if LLM numbers are within reasonable range of tool numbers
        for llm_num in llm_numbers:
            # Find closest tool number
            closest = min(tool_numbers, key=lambda x: abs(x - llm_num))
            diff_ratio = abs(llm_num - closest) / (abs(closest) + 1e-6)
            
            # If difference > 20%, flag as potential hallucination
            if diff_ratio > 0.2:
                mismatches.append(f"{data_type}: LLM输出{llm_num:.2f}与工具数据{closest:.2f}差异较大")
        
        return mismatches
    
    def force_tool_grounding(self, ticker: str, required_tools: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Force tool grounding: Ensure all required tools are called
        
        Args:
            ticker: Stock ticker
            required_tools: List of required tool names
        
        Returns:
            Tuple of (all_tools_called, tool_results)
        """
        tool_results = {}
        all_called = True
        
        for tool_name in required_tools:
            key = f"{ticker}:{tool_name}"
            if key not in self.tool_data_cache:
                all_called = False
                tool_results[tool_name] = None
            else:
                tool_results[tool_name] = self.tool_data_cache[key]["data"]
        
        return all_called, tool_results


# Global instance
hallucination_checker = HallucinationChecker()

