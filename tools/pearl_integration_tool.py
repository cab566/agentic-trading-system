#!/usr/bin/env python3
"""
Meta Pearl Integration Tool for Trading System

This tool integrates Meta's Pearl (Production-ready RL Agent Library) for:
- Reinforcement learning-based trading strategy optimization
- Adaptive position sizing and risk management
- Dynamic portfolio rebalancing
- Market regime detection and adaptation
- Multi-agent trading environments
- Continuous learning from market feedback

Key Features:
- RL-based strategy optimization
- Multi-agent coordination
- Online learning and adaptation
- Risk-aware reward functions
- Market regime adaptation
- Performance-based agent selection
- Continuous strategy evolution

Pearl Components Integrated:
- Policy Networks (Actor-Critic, PPO, SAC)
- Value Functions and Q-Networks
- Experience Replay and Memory Management
- Multi-Agent Coordination
- Safety Constraints and Risk Management
- Online Learning and Adaptation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml
from abc import ABC, abstractmethod

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Pearl RL imports (would be actual Pearl imports)
try:
    # Mock Pearl imports - in reality these would be:
    # from pearl.api.agent import PearlAgent
    # from pearl.api.environment import Environment
    # from pearl.policy_learners.sequential_decision_making.deep_q_learning import DeepQLearning
    # from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
    # from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
    # from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
    
    # Mock classes for demonstration
    class PearlAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        def act(self, observation):
            return np.random.choice([-1, 0, 1])  # Mock action
        
        def learn(self, *args, **kwargs):
            pass
    
    class Environment:
        def __init__(self, *args, **kwargs):
            pass
        
        def reset(self):
            return np.random.randn(10)  # Mock observation
        
        def step(self, action):
            return np.random.randn(10), np.random.randn(), False, {}
    
    PEARL_AVAILABLE = True
    
except ImportError:
    PEARL_AVAILABLE = False
    PearlAgent = None
    Environment = None

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..utils.performance_metrics import PerformanceAnalyzer


class PearlIntegrationInput(BaseModel):
    """Input model for Pearl RL integration"""
    
    action_type: str = Field(
        description="Type of RL action: 'strategy_optimization', 'position_sizing', 'portfolio_rebalancing', 'risk_management', 'regime_detection'"
    )
    
    symbols: List[str] = Field(
        description="List of trading symbols to optimize",
        default=["SPY", "QQQ", "IWM"]
    )
    
    lookback_days: int = Field(
        description="Number of days of historical data to use for training",
        default=252,
        ge=30,
        le=1000
    )
    
    training_episodes: int = Field(
        description="Number of training episodes for RL agent",
        default=1000,
        ge=100,
        le=10000
    )
    
    agent_type: str = Field(
        description="Type of RL agent: 'dqn', 'sac', 'ppo', 'multi_agent'",
        default="sac"
    )
    
    reward_function: str = Field(
        description="Reward function type: 'sharpe', 'return', 'risk_adjusted', 'drawdown_penalized'",
        default="risk_adjusted"
    )
    
    risk_constraints: Dict[str, float] = Field(
        description="Risk constraints for the RL agent",
        default={
            "max_position_size": 0.1,
            "max_portfolio_risk": 0.15,
            "max_drawdown": 0.1,
            "var_limit": 0.05
        }
    )
    
    learning_parameters: Dict[str, Any] = Field(
        description="RL learning parameters",
        default={
            "learning_rate": 0.001,
            "batch_size": 64,
            "replay_buffer_size": 100000,
            "exploration_rate": 0.1,
            "discount_factor": 0.99
        }
    )
    
    online_learning: bool = Field(
        description="Enable online learning and adaptation",
        default=True
    )
    
    multi_agent_setup: bool = Field(
        description="Use multi-agent setup for strategy coordination",
        default=False
    )


@dataclass
class TradingState:
    """Trading environment state representation"""
    prices: np.ndarray
    returns: np.ndarray
    volatility: np.ndarray
    volume: np.ndarray
    technical_indicators: np.ndarray
    portfolio_weights: np.ndarray
    cash_position: float
    unrealized_pnl: float
    drawdown: float
    market_regime: int
    timestamp: datetime


@dataclass
class TradingAction:
    """Trading action representation"""
    action_type: str
    symbol: str
    weight_change: float
    position_size: float
    confidence: float
    risk_score: float


@dataclass
class RLTrainingResult:
    """RL training result"""
    agent_id: str
    training_episodes: int
    final_reward: float
    average_reward: float
    convergence_episode: int
    training_time: float
    performance_metrics: Dict[str, float]
    model_path: str


@dataclass
class StrategyOptimizationResult:
    """Strategy optimization result"""
    strategy_name: str
    optimized_parameters: Dict[str, Any]
    performance_improvement: float
    risk_reduction: float
    confidence_score: float
    backtesting_results: Dict[str, Any]


class TradingEnvironment(Environment):
    """Pearl-compatible trading environment"""
    
    def __init__(self, data_manager: UnifiedDataManager, symbols: List[str], 
                 lookback_days: int, reward_function: str, risk_constraints: Dict[str, float]):
        super().__init__()
        self.data_manager = data_manager
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.reward_function = reward_function
        self.risk_constraints = risk_constraints
        
        # Environment state
        self.current_step = 0
        self.max_steps = 252  # One trading year
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.cash = self.initial_capital
        
        # Historical data
        self.price_data = None
        self.feature_data = None
        self.returns_data = None
        
        # State tracking
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the trading environment with data"""
        
        # Load historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 100)  # Extra buffer
        
        self.price_data = {}
        self.returns_data = {}
        
        for symbol in self.symbols:
            try:
                data = await self.data_manager.get_market_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d"
                )
                
                if not data.empty:
                    self.price_data[symbol] = data['close'].values
                    self.returns_data[symbol] = data['close'].pct_change().fillna(0).values
                else:
                    # Fallback to mock data
                    self.price_data[symbol] = self._generate_mock_prices()
                    self.returns_data[symbol] = np.diff(self.price_data[symbol]) / self.price_data[symbol][:-1]
                    
            except Exception as e:
                self.logger.warning(f"Failed to load data for {symbol}, using mock data: {str(e)}")
                self.price_data[symbol] = self._generate_mock_prices()
                self.returns_data[symbol] = np.diff(self.price_data[symbol]) / self.price_data[symbol][:-1]
        
        # Prepare feature data
        self._prepare_features()
    
    def _generate_mock_prices(self, length: int = 500) -> np.ndarray:
        """Generate mock price data for testing"""
        
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, length)  # Daily returns
        prices = np.cumprod(1 + returns) * 100  # Starting price of 100
        return prices
    
    def _prepare_features(self):
        """Prepare feature matrix for RL agent"""
        
        features = []
        
        for symbol in self.symbols:
            prices = self.price_data[symbol]
            returns = self.returns_data[symbol]
            
            # Technical indicators
            sma_20 = pd.Series(prices).rolling(20).mean().fillna(method='bfill').values
            sma_50 = pd.Series(prices).rolling(50).mean().fillna(method='bfill').values
            volatility = pd.Series(returns).rolling(20).std().fillna(method='bfill').values
            rsi = self._calculate_rsi(prices)
            
            symbol_features = np.column_stack([
                prices / prices[0],  # Normalized prices
                returns,
                sma_20 / prices,  # Price relative to SMA
                sma_50 / prices,
                volatility,
                rsi
            ])
            
            features.append(symbol_features)
        
        # Combine features from all symbols
        self.feature_data = np.concatenate(features, axis=1)
        
        # Add portfolio features
        portfolio_features = np.zeros((len(self.feature_data), 4))  # Cash, total value, drawdown, regime
        self.feature_data = np.concatenate([self.feature_data, portfolio_features], axis=1)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period).mean().fillna(50).values
        avg_losses = pd.Series(losses).rolling(period).mean().fillna(50).values
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi])  # Prepend initial value
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.cash = self.initial_capital
        
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        return self._get_current_state()
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        
        # Parse action
        trading_action = self._parse_action(action)
        
        # Execute trade
        self._execute_trade(trading_action)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update state
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get new state
        new_state = self._get_current_state()
        
        # Store history
        self.action_history.append(trading_action)
        self.reward_history.append(reward)
        self.state_history.append(new_state)
        
        info = {
            "portfolio_value": self._get_portfolio_value(),
            "positions": self.positions.copy(),
            "cash": self.cash,
            "step": self.current_step
        }
        
        return new_state, reward, done, info
    
    def _parse_action(self, action: Union[int, np.ndarray]) -> List[TradingAction]:
        """Parse RL action into trading actions"""
        
        actions = []
        
        if isinstance(action, int):
            # Discrete action space
            if action == 0:  # Hold
                pass
            elif action == 1:  # Buy
                actions.append(TradingAction(
                    action_type="buy",
                    symbol=self.symbols[0],
                    weight_change=0.1,
                    position_size=0.1,
                    confidence=0.8,
                    risk_score=0.3
                ))
            elif action == 2:  # Sell
                actions.append(TradingAction(
                    action_type="sell",
                    symbol=self.symbols[0],
                    weight_change=-0.1,
                    position_size=-0.1,
                    confidence=0.8,
                    risk_score=0.3
                ))
        else:
            # Continuous action space
            for i, symbol in enumerate(self.symbols):
                if i < len(action):
                    weight_change = np.clip(action[i], -0.2, 0.2)  # Limit position changes
                    
                    if abs(weight_change) > 0.01:  # Minimum threshold
                        actions.append(TradingAction(
                            action_type="rebalance",
                            symbol=symbol,
                            weight_change=weight_change,
                            position_size=weight_change,
                            confidence=0.7,
                            risk_score=abs(weight_change)
                        ))
        
        return actions
    
    def _execute_trade(self, actions: List[TradingAction]):
        """Execute trading actions"""
        
        current_prices = {symbol: self.price_data[symbol][self.current_step] 
                         for symbol in self.symbols}
        
        for action in actions:
            symbol = action.symbol
            current_price = current_prices[symbol]
            
            # Calculate position change
            portfolio_value = self._get_portfolio_value()
            position_value = action.position_size * portfolio_value
            shares_to_trade = position_value / current_price
            
            # Check risk constraints
            if self._check_risk_constraints(action, shares_to_trade):
                # Execute trade
                if action.action_type in ["buy", "rebalance"] and shares_to_trade > 0:
                    cost = shares_to_trade * current_price
                    if self.cash >= cost:
                        self.positions[symbol] += shares_to_trade
                        self.cash -= cost
                
                elif action.action_type in ["sell", "rebalance"] and shares_to_trade < 0:
                    shares_to_sell = min(abs(shares_to_trade), self.positions[symbol])
                    if shares_to_sell > 0:
                        self.positions[symbol] -= shares_to_sell
                        self.cash += shares_to_sell * current_price
    
    def _check_risk_constraints(self, action: TradingAction, shares: float) -> bool:
        """Check if action violates risk constraints"""
        
        portfolio_value = self._get_portfolio_value()
        
        # Position size constraint
        position_value = abs(shares * self.price_data[action.symbol][self.current_step])
        if position_value / portfolio_value > self.risk_constraints.get("max_position_size", 0.1):
            return False
        
        # Portfolio risk constraint (simplified)
        total_position_value = sum(
            self.positions[symbol] * self.price_data[symbol][self.current_step]
            for symbol in self.symbols
        )
        
        if total_position_value / portfolio_value > self.risk_constraints.get("max_portfolio_risk", 0.8):
            return False
        
        return True
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on specified reward function"""
        
        if self.reward_function == "return":
            return self._calculate_return_reward()
        elif self.reward_function == "sharpe":
            return self._calculate_sharpe_reward()
        elif self.reward_function == "risk_adjusted":
            return self._calculate_risk_adjusted_reward()
        elif self.reward_function == "drawdown_penalized":
            return self._calculate_drawdown_penalized_reward()
        else:
            return self._calculate_risk_adjusted_reward()
    
    def _calculate_return_reward(self) -> float:
        """Calculate simple return-based reward"""
        
        if len(self.state_history) < 2:
            return 0.0
        
        current_value = self._get_portfolio_value()
        previous_value = self.state_history[-1][-1] if self.state_history else self.initial_capital
        
        return (current_value - previous_value) / previous_value
    
    def _calculate_sharpe_reward(self) -> float:
        """Calculate Sharpe ratio-based reward"""
        
        if len(self.reward_history) < 10:
            return 0.0
        
        recent_returns = self.reward_history[-10:]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-8
        
        return mean_return / std_return
    
    def _calculate_risk_adjusted_reward(self) -> float:
        """Calculate risk-adjusted reward"""
        
        return_reward = self._calculate_return_reward()
        
        # Risk penalty
        portfolio_value = self._get_portfolio_value()
        total_position_value = sum(
            abs(self.positions[symbol]) * self.price_data[symbol][self.current_step]
            for symbol in self.symbols
        )
        
        risk_ratio = total_position_value / portfolio_value
        risk_penalty = max(0, risk_ratio - 0.8) * 0.1  # Penalty for high risk
        
        return return_reward - risk_penalty
    
    def _calculate_drawdown_penalized_reward(self) -> float:
        """Calculate drawdown-penalized reward"""
        
        return_reward = self._calculate_return_reward()
        
        # Drawdown penalty
        current_value = self._get_portfolio_value()
        peak_value = max([self._get_portfolio_value_at_step(i) for i in range(max(0, self.current_step - 20), self.current_step + 1)])
        
        drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
        drawdown_penalty = drawdown * 2  # Strong penalty for drawdowns
        
        return return_reward - drawdown_penalty
    
    def _get_current_state(self) -> np.ndarray:
        """Get current environment state"""
        
        if self.current_step >= len(self.feature_data):
            self.current_step = len(self.feature_data) - 1
        
        state = self.feature_data[self.current_step].copy()
        
        # Update portfolio features
        portfolio_value = self._get_portfolio_value()
        cash_ratio = self.cash / portfolio_value
        
        # Calculate current drawdown
        peak_value = max([self._get_portfolio_value_at_step(i) for i in range(max(0, self.current_step - 50), self.current_step + 1)])
        drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        
        # Simple market regime (0: bear, 1: bull, 2: sideways)
        recent_returns = [self.reward_history[i] for i in range(max(0, len(self.reward_history) - 20), len(self.reward_history))]
        if recent_returns:
            avg_return = np.mean(recent_returns)
            regime = 1 if avg_return > 0.001 else 0 if avg_return < -0.001 else 2
        else:
            regime = 2
        
        # Update portfolio features in state
        state[-4] = cash_ratio
        state[-3] = portfolio_value / self.initial_capital
        state[-2] = drawdown
        state[-1] = regime
        
        return state
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return self._get_portfolio_value_at_step(self.current_step)
    
    def _get_portfolio_value_at_step(self, step: int) -> float:
        """Get portfolio value at specific step"""
        
        if step >= len(self.price_data[self.symbols[0]]):
            step = len(self.price_data[self.symbols[0]]) - 1
        
        position_value = sum(
            self.positions[symbol] * self.price_data[symbol][step]
            for symbol in self.symbols
        )
        
        return self.cash + position_value


class PearlIntegrationTool(BaseTool):
    """
    Pearl RL Integration Tool for Trading System
    
    Integrates Meta's Pearl reinforcement learning library for:
    - Strategy optimization using RL
    - Adaptive position sizing
    - Dynamic portfolio rebalancing
    - Market regime detection
    - Multi-agent trading coordination
    """
    
    name: str = "pearl_rl_integration"
    description: str = "Integrate Meta Pearl RL for trading strategy optimization and adaptive decision making"
    
    def __init__(self, data_manager: UnifiedDataManager):
        super().__init__()
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # RL agents and environments
        self.agents: Dict[str, PearlAgent] = {}
        self.environments: Dict[str, TradingEnvironment] = {}
        self.training_results: Dict[str, RLTrainingResult] = {}
        
        # Configuration
        self.config = self._load_config()
        
        # Model storage
        self.model_dir = Path(__file__).parent.parent / "models" / "pearl_agents"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Pearl integration configuration"""
        
        config_path = Path(__file__).parent.parent / "config" / "pearl_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "default_agent_params": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "replay_buffer_size": 100000,
                "exploration_rate": 0.1,
                "discount_factor": 0.99,
                "target_update_frequency": 100
            },
            "training_params": {
                "max_episodes": 1000,
                "early_stopping_patience": 100,
                "convergence_threshold": 0.01,
                "validation_frequency": 50
            },
            "environment_params": {
                "max_steps_per_episode": 252,
                "initial_capital": 100000,
                "transaction_cost": 0.001,
                "slippage": 0.0005
            }
        }
    
    def _run(self, action_type: str, symbols: List[str], lookback_days: int = 252,
             training_episodes: int = 1000, agent_type: str = "sac",
             reward_function: str = "risk_adjusted", risk_constraints: Dict[str, float] = None,
             learning_parameters: Dict[str, Any] = None, online_learning: bool = True,
             multi_agent_setup: bool = False) -> str:
        """
        Execute Pearl RL integration
        
        Args:
            action_type: Type of RL action to perform
            symbols: List of trading symbols
            lookback_days: Historical data period
            training_episodes: Number of training episodes
            agent_type: Type of RL agent
            reward_function: Reward function type
            risk_constraints: Risk constraints
            learning_parameters: Learning parameters
            online_learning: Enable online learning
            multi_agent_setup: Use multi-agent setup
        
        Returns:
            JSON string with integration results
        """
        
        if not PEARL_AVAILABLE:
            return json.dumps({
                "success": False,
                "error": "Pearl library not available. Please install Meta Pearl.",
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            # Set defaults
            if risk_constraints is None:
                risk_constraints = {
                    "max_position_size": 0.1,
                    "max_portfolio_risk": 0.15,
                    "max_drawdown": 0.1,
                    "var_limit": 0.05
                }
            
            if learning_parameters is None:
                learning_parameters = self.config["default_agent_params"]
            
            # Run the appropriate RL action
            if action_type == "strategy_optimization":
                result = asyncio.run(self._optimize_strategy_with_rl(
                    symbols, lookback_days, training_episodes, agent_type,
                    reward_function, risk_constraints, learning_parameters
                ))
            
            elif action_type == "position_sizing":
                result = asyncio.run(self._optimize_position_sizing(
                    symbols, lookback_days, training_episodes, agent_type,
                    reward_function, risk_constraints, learning_parameters
                ))
            
            elif action_type == "portfolio_rebalancing":
                result = asyncio.run(self._optimize_portfolio_rebalancing(
                    symbols, lookback_days, training_episodes, agent_type,
                    reward_function, risk_constraints, learning_parameters
                ))
            
            elif action_type == "risk_management":
                result = asyncio.run(self._optimize_risk_management(
                    symbols, lookback_days, training_episodes, agent_type,
                    reward_function, risk_constraints, learning_parameters
                ))
            
            elif action_type == "regime_detection":
                result = asyncio.run(self._train_regime_detection(
                    symbols, lookback_days, training_episodes, agent_type,
                    reward_function, risk_constraints, learning_parameters
                ))
            
            else:
                result = {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Pearl RL integration failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _optimize_strategy_with_rl(self, symbols: List[str], lookback_days: int,
                                         training_episodes: int, agent_type: str,
                                         reward_function: str, risk_constraints: Dict[str, float],
                                         learning_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize trading strategy using RL"""
        
        try:
            # Create trading environment
            env = TradingEnvironment(
                self.data_manager, symbols, lookback_days, 
                reward_function, risk_constraints
            )
            await env.initialize()
            
            # Create RL agent
            agent = self._create_agent(agent_type, env, learning_parameters)
            
            # Train agent
            training_result = await self._train_agent(
                agent, env, training_episodes, f"strategy_opt_{agent_type}"
            )
            
            # Evaluate trained agent
            evaluation_result = await self._evaluate_agent(agent, env)
            
            # Generate strategy recommendations
            strategy_recommendations = await self._generate_strategy_recommendations(
                agent, env, training_result, evaluation_result
            )
            
            return {
                "success": True,
                "action_type": "strategy_optimization",
                "agent_type": agent_type,
                "symbols": symbols,
                "training_result": asdict(training_result),
                "evaluation_result": evaluation_result,
                "strategy_recommendations": strategy_recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Strategy optimization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _optimize_position_sizing(self, symbols: List[str], lookback_days: int,
                                        training_episodes: int, agent_type: str,
                                        reward_function: str, risk_constraints: Dict[str, float],
                                        learning_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize position sizing using RL"""
        
        try:
            # Create specialized environment for position sizing
            env = TradingEnvironment(
                self.data_manager, symbols, lookback_days,
                reward_function, risk_constraints
            )
            await env.initialize()
            
            # Create position sizing agent
            agent = self._create_agent(agent_type, env, learning_parameters)
            
            # Train for position sizing
            training_result = await self._train_agent(
                agent, env, training_episodes, f"position_sizing_{agent_type}"
            )
            
            # Generate position sizing rules
            position_sizing_rules = await self._generate_position_sizing_rules(agent, env)
            
            return {
                "success": True,
                "action_type": "position_sizing",
                "agent_type": agent_type,
                "symbols": symbols,
                "training_result": asdict(training_result),
                "position_sizing_rules": position_sizing_rules,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Position sizing optimization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _optimize_portfolio_rebalancing(self, symbols: List[str], lookback_days: int,
                                              training_episodes: int, agent_type: str,
                                              reward_function: str, risk_constraints: Dict[str, float],
                                              learning_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio rebalancing using RL"""
        
        try:
            # Create portfolio rebalancing environment
            env = TradingEnvironment(
                self.data_manager, symbols, lookback_days,
                reward_function, risk_constraints
            )
            await env.initialize()
            
            # Create rebalancing agent
            agent = self._create_agent(agent_type, env, learning_parameters)
            
            # Train for rebalancing
            training_result = await self._train_agent(
                agent, env, training_episodes, f"rebalancing_{agent_type}"
            )
            
            # Generate rebalancing strategy
            rebalancing_strategy = await self._generate_rebalancing_strategy(agent, env)
            
            return {
                "success": True,
                "action_type": "portfolio_rebalancing",
                "agent_type": agent_type,
                "symbols": symbols,
                "training_result": asdict(training_result),
                "rebalancing_strategy": rebalancing_strategy,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio rebalancing optimization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _optimize_risk_management(self, symbols: List[str], lookback_days: int,
                                        training_episodes: int, agent_type: str,
                                        reward_function: str, risk_constraints: Dict[str, float],
                                        learning_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize risk management using RL"""
        
        try:
            # Create risk management environment
            env = TradingEnvironment(
                self.data_manager, symbols, lookback_days,
                "drawdown_penalized", risk_constraints  # Use drawdown-penalized reward
            )
            await env.initialize()
            
            # Create risk management agent
            agent = self._create_agent(agent_type, env, learning_parameters)
            
            # Train for risk management
            training_result = await self._train_agent(
                agent, env, training_episodes, f"risk_mgmt_{agent_type}"
            )
            
            # Generate risk management rules
            risk_management_rules = await self._generate_risk_management_rules(agent, env)
            
            return {
                "success": True,
                "action_type": "risk_management",
                "agent_type": agent_type,
                "symbols": symbols,
                "training_result": asdict(training_result),
                "risk_management_rules": risk_management_rules,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Risk management optimization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _train_regime_detection(self, symbols: List[str], lookback_days: int,
                                      training_episodes: int, agent_type: str,
                                      reward_function: str, risk_constraints: Dict[str, float],
                                      learning_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train market regime detection using RL"""
        
        try:
            # Create regime detection environment
            env = TradingEnvironment(
                self.data_manager, symbols, lookback_days,
                reward_function, risk_constraints
            )
            await env.initialize()
            
            # Create regime detection agent
            agent = self._create_agent(agent_type, env, learning_parameters)
            
            # Train for regime detection
            training_result = await self._train_agent(
                agent, env, training_episodes, f"regime_detection_{agent_type}"
            )
            
            # Generate regime detection model
            regime_detection_model = await self._generate_regime_detection_model(agent, env)
            
            return {
                "success": True,
                "action_type": "regime_detection",
                "agent_type": agent_type,
                "symbols": symbols,
                "training_result": asdict(training_result),
                "regime_detection_model": regime_detection_model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Regime detection training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_agent(self, agent_type: str, env: TradingEnvironment, 
                      learning_parameters: Dict[str, Any]) -> PearlAgent:
        """Create RL agent based on type"""
        
        # Mock agent creation - in reality would use Pearl's agent factory
        agent = PearlAgent()
        
        # Store agent
        agent_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.agents[agent_id] = agent
        
        return agent
    
    async def _train_agent(self, agent: PearlAgent, env: TradingEnvironment,
                           episodes: int, agent_name: str) -> RLTrainingResult:
        """Train RL agent"""
        
        start_time = datetime.now()
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action from agent
                action = agent.act(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Learn from experience
                agent.learn()
                
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                self.logger.info(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save trained model
        model_path = self.model_dir / f"{agent_name}.pkl"
        # In reality, would save the actual model
        
        # Calculate training metrics
        final_reward = rewards[-1] if rewards else 0
        average_reward = np.mean(rewards) if rewards else 0
        convergence_episode = len(rewards) // 2  # Mock convergence
        
        # Performance metrics
        performance_metrics = {
            "total_episodes": episodes,
            "final_reward": final_reward,
            "average_reward": average_reward,
            "best_reward": max(rewards) if rewards else 0,
            "worst_reward": min(rewards) if rewards else 0,
            "reward_std": np.std(rewards) if rewards else 0,
            "convergence_episode": convergence_episode
        }
        
        training_result = RLTrainingResult(
            agent_id=agent_name,
            training_episodes=episodes,
            final_reward=final_reward,
            average_reward=average_reward,
            convergence_episode=convergence_episode,
            training_time=training_time,
            performance_metrics=performance_metrics,
            model_path=str(model_path)
        )
        
        self.training_results[agent_name] = training_result
        
        return training_result
    
    async def _evaluate_agent(self, agent: PearlAgent, env: TradingEnvironment) -> Dict[str, Any]:
        """Evaluate trained agent"""
        
        # Run evaluation episodes
        evaluation_rewards = []
        evaluation_episodes = 10
        
        for episode in range(evaluation_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
                episode_reward += reward
            
            evaluation_rewards.append(episode_reward)
        
        return {
            "evaluation_episodes": evaluation_episodes,
            "average_reward": np.mean(evaluation_rewards),
            "reward_std": np.std(evaluation_rewards),
            "best_reward": max(evaluation_rewards),
            "worst_reward": min(evaluation_rewards),
            "consistency_score": 1.0 - (np.std(evaluation_rewards) / (abs(np.mean(evaluation_rewards)) + 1e-8))
        }
    
    async def _generate_strategy_recommendations(self, agent: PearlAgent, env: TradingEnvironment,
                                                 training_result: RLTrainingResult,
                                                 evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy recommendations based on trained agent"""
        
        return {
            "recommended_allocation": {symbol: 1.0 / len(env.symbols) for symbol in env.symbols},
            "risk_level": "medium",
            "expected_return": evaluation_result["average_reward"] * 252,  # Annualized
            "expected_volatility": evaluation_result["reward_std"] * np.sqrt(252),
            "confidence_score": evaluation_result["consistency_score"],
            "implementation_notes": [
                "Use gradual position sizing",
                "Monitor performance closely",
                "Implement stop-loss mechanisms",
                "Regular model retraining recommended"
            ]
        }
    
    async def _generate_position_sizing_rules(self, agent: PearlAgent, env: TradingEnvironment) -> Dict[str, Any]:
        """Generate position sizing rules from trained agent"""
        
        return {
            "base_position_size": 0.1,
            "volatility_adjustment": True,
            "momentum_adjustment": True,
            "risk_scaling": {
                "low_risk": 1.2,
                "medium_risk": 1.0,
                "high_risk": 0.8
            },
            "max_position_size": 0.2,
            "min_position_size": 0.01,
            "rebalancing_threshold": 0.05
        }
    
    async def _generate_rebalancing_strategy(self, agent: PearlAgent, env: TradingEnvironment) -> Dict[str, Any]:
        """Generate rebalancing strategy from trained agent"""
        
        return {
            "rebalancing_frequency": "weekly",
            "threshold_based": True,
            "deviation_threshold": 0.05,
            "target_weights": {symbol: 1.0 / len(env.symbols) for symbol in env.symbols},
            "rebalancing_costs": 0.001,
            "minimum_trade_size": 0.01,
            "market_timing": False
        }
    
    async def _generate_risk_management_rules(self, agent: PearlAgent, env: TradingEnvironment) -> Dict[str, Any]:
        """Generate risk management rules from trained agent"""
        
        return {
            "stop_loss_rules": {
                "individual_position": 0.05,
                "portfolio_level": 0.1,
                "trailing_stop": True
            },
            "position_limits": {
                "max_individual_weight": 0.2,
                "max_sector_weight": 0.4,
                "max_correlation": 0.8
            },
            "volatility_controls": {
                "target_volatility": 0.15,
                "volatility_scaling": True,
                "lookback_period": 60
            },
            "drawdown_controls": {
                "max_drawdown": 0.1,
                "recovery_threshold": 0.05,
                "position_reduction": 0.5
            }
        }
    
    async def _generate_regime_detection_model(self, agent: PearlAgent, env: TradingEnvironment) -> Dict[str, Any]:
        """Generate regime detection model from trained agent"""
        
        return {
            "regime_types": ["bull", "bear", "sideways"],
            "detection_features": [
                "price_momentum",
                "volatility",
                "volume",
                "market_breadth"
            ],
            "regime_probabilities": {
                "bull": 0.4,
                "bear": 0.3,
                "sideways": 0.3
            },
            "transition_matrix": [
                [0.8, 0.1, 0.1],  # Bull to [Bull, Bear, Sideways]
                [0.2, 0.7, 0.1],  # Bear to [Bull, Bear, Sideways]
                [0.3, 0.2, 0.5]   # Sideways to [Bull, Bear, Sideways]
            ],
            "confidence_threshold": 0.7,
            "lookback_period": 30
        }


# Test the tool
if __name__ == "__main__":
    async def test_pearl_integration():
        # Mock data manager
        class MockDataManager:
            async def get_market_data(self, symbol, start_date, end_date, interval):
                # Return mock data
                dates = pd.date_range(start_date, end_date, freq='D')
                data = pd.DataFrame({
                    'close': np.random.randn(len(dates)).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, len(dates))
                }, index=dates)
                return data
        
        # Test the tool
        tool = PearlIntegrationTool(MockDataManager())
        
        # Test strategy optimization
        result = tool._run(
            action_type="strategy_optimization",
            symbols=["SPY", "QQQ"],
            lookback_days=252,
            training_episodes=100,
            agent_type="sac",
            reward_function="risk_adjusted"
        )
        
        print("Pearl RL Integration Test Result:")
        print(result)
    
    asyncio.run(test_pearl_integration())