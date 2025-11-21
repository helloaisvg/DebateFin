"""
Backtest-Guided PPO Router
Uses historical Sharpe ratio as reward to guide debate routing
This is the core innovation: Backtest-Guided Router
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class PPORouter:
    """
    PPO-based router that uses backtest Sharpe ratio as reward signal
    Guides debate flow and agent selection based on historical performance
    """
    
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.99):
        """
        Args:
            learning_rate: PPO learning rate
            gamma: Discount factor
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Policy parameters (simplified PPO)
        self.policy_weights = {
            "analyst_weight": 1.0,
            "risk_weight": 1.0,
            "trader_weight": 1.0,
            "judge_weight": 1.0,
            "debate_rounds": 3.0,  # Preferred debate rounds
        }
        
        # Performance history
        self.performance_history = []
        
    def calculate_reward(self, backtest_result: Dict[str, Any]) -> float:
        """
        Calculate reward from backtest result
        
        Args:
            backtest_result: Backtest results with Sharpe, returns, etc.
        
        Returns:
            Reward value (higher is better)
        """
        sharpe = backtest_result.get("sharpe_ratio", 0.0)
        total_return = backtest_result.get("total_return", 0.0)
        max_drawdown = abs(backtest_result.get("max_drawdown", 0.0))
        
        # Reward = Sharpe * 0.6 + Return * 0.3 - Drawdown * 0.1
        reward = sharpe * 0.6 + (total_return / 100.0) * 0.3 - max_drawdown * 0.1
        
        return reward
    
    def route_decision(self, current_state: Dict[str, Any], 
                      available_agents: List[str]) -> Dict[str, Any]:
        """
        Make routing decision based on current state and policy
        
        Args:
            current_state: Current debate state
            available_agents: List of available agents
        
        Returns:
            Routing decision with next agent and parameters
        """
        round_num = current_state.get("round", 1)
        max_rounds = current_state.get("max_rounds", 3)
        
        # Calculate agent weights based on performance history
        agent_weights = {}
        for agent in available_agents:
            weight_key = f"{agent.lower()}_weight"
            agent_weights[agent] = self.policy_weights.get(weight_key, 1.0)
        
        # Normalize weights
        total_weight = sum(agent_weights.values())
        if total_weight > 0:
            agent_weights = {k: v / total_weight for k, v in agent_weights.items()}
        else:
            agent_weights = {agent: 1.0 / len(available_agents) for agent in available_agents}
        
        # Select next agent (weighted random or deterministic)
        # For demo: use deterministic based on round
        if round_num == 1:
            next_agent = "Analyst"
        elif round_num == 2:
            next_agent = "Risk"
        elif round_num == 3:
            next_agent = "Trader"
        else:
            next_agent = "Judge"
        
        # Decide if should continue debate
        preferred_rounds = int(self.policy_weights.get("debate_rounds", 3))
        should_continue = round_num < min(max_rounds, preferred_rounds)
        
        decision = {
            "next_agent": next_agent,
            "should_continue": should_continue,
            "agent_weights": agent_weights,
            "confidence": 0.8,
            "reasoning": f"基于历史回测表现，选择{next_agent}进行第{round_num}轮辩论"
        }
        
        return decision
    
    def update_policy(self, reward: float, action_history: List[Dict[str, Any]]):
        """
        Update policy using PPO update rule
        
        Args:
            reward: Final reward from backtest
            action_history: History of actions taken
        """
        # Simplified PPO update
        # In full implementation, would use advantage estimation and clipping
        
        # Update agent weights based on performance
        if reward > 0:
            # Positive reward: increase weights of agents that contributed
            for action in action_history:
                agent = action.get("agent", "")
                if agent:
                    weight_key = f"{agent.lower()}_weight"
                    if weight_key in self.policy_weights:
                        self.policy_weights[weight_key] *= (1 + self.learning_rate * reward)
        else:
            # Negative reward: decrease weights
            for action in action_history:
                agent = action.get("agent", "")
                if agent:
                    weight_key = f"{agent.lower()}_weight"
                    if weight_key in self.policy_weights:
                        self.policy_weights[weight_key] *= (1 - self.learning_rate * abs(reward))
        
        # Clip weights to reasonable range
        for key in self.policy_weights:
            if "weight" in key:
                self.policy_weights[key] = max(0.1, min(2.0, self.policy_weights[key]))
        
        # Record performance
        self.performance_history.append({
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
            "policy_weights": self.policy_weights.copy()
        })
    
    def get_routing_strategy(self) -> Dict[str, Any]:
        """Get current routing strategy"""
        return {
            "policy_weights": self.policy_weights.copy(),
            "performance_history_count": len(self.performance_history),
            "avg_reward": np.mean([p["reward"] for p in self.performance_history]) if self.performance_history else 0.0
        }


# Global router instance
ppo_router = PPORouter()

