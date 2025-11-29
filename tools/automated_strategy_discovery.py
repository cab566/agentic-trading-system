#!/usr/bin/env python3
"""
Automated Strategy Discovery System

This tool uses machine learning, pattern recognition, and systematic backtesting
to discover and validate new trading strategies automatically.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib
from pathlib import Path

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import optuna
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

# Technical analysis
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.config_manager import ConfigManager


class StrategyType(Enum):
    """Types of trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PAIRS_TRADING = "pairs_trading"
    SEASONAL = "seasonal"
    NEWS_DRIVEN = "news_driven"
    TECHNICAL_PATTERN = "technical_pattern"
    MACHINE_LEARNING = "machine_learning"
    ARBITRAGE = "arbitrage"
    VOLATILITY = "volatility"


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    price: float
    strategy_name: str
    features: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    total_trades: int
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    backtest_period: Tuple[datetime, datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredStrategy:
    """Discovered trading strategy"""
    strategy_id: str
    name: str
    strategy_type: StrategyType
    description: str
    parameters: Dict[str, Any]
    features: List[str]
    performance: StrategyPerformance
    confidence_score: float
    discovery_timestamp: datetime
    validation_status: str
    code: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyDiscoveryConfig(BaseModel):
    """Configuration for strategy discovery"""
    min_sharpe_ratio: float = Field(default=1.0, description="Minimum Sharpe ratio")
    min_win_rate: float = Field(default=0.55, description="Minimum win rate")
    max_drawdown: float = Field(default=0.15, description="Maximum drawdown")
    min_trades: int = Field(default=50, description="Minimum number of trades")
    backtest_period_days: int = Field(default=252, description="Backtesting period in days")
    validation_period_days: int = Field(default=63, description="Validation period in days")
    feature_selection_threshold: float = Field(default=0.05, description="Feature importance threshold")
    optimization_trials: int = Field(default=100, description="Optuna optimization trials")
    symbols_universe: List[str] = Field(default=[], description="Universe of symbols to test")


class AutomatedStrategyDiscoveryTool(BaseTool):
    """Automated strategy discovery and validation system"""
    
    name: str = "automated_strategy_discovery"
    description: str = "Discovers and validates new trading strategies using machine learning and systematic backtesting"
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        super().__init__()
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Strategy storage
        self.discovered_strategies: List[DiscoveredStrategy] = []
        self.strategy_cache_path = Path("strategy_cache")
        self.strategy_cache_path.mkdir(exist_ok=True)
        
        # Feature engineering functions
        self.feature_functions = self._initialize_feature_functions()
        
        # Strategy templates
        self.strategy_templates = self._initialize_strategy_templates()
        
        # Performance tracking
        self.discovery_stats = {
            'total_strategies_tested': 0,
            'strategies_discovered': 0,
            'avg_performance': {},
            'best_strategy': None,
            'last_discovery_time': None
        }
        
        # Load existing strategies
        self._load_cached_strategies()
    
    def _load_config(self) -> StrategyDiscoveryConfig:
        """Load strategy discovery configuration"""
        try:
            config_dict = self.config_manager.get_config('strategy_discovery')
            return StrategyDiscoveryConfig(**config_dict)
        except Exception as e:
            self.logger.warning(f"Could not load strategy discovery config: {e}")
            return StrategyDiscoveryConfig(
                symbols_universe=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
            )
    
    def _initialize_feature_functions(self) -> Dict[str, Callable]:
        """Initialize feature engineering functions"""
        features = {}
        
        # Price-based features
        features['returns_1d'] = lambda df: df['Close'].pct_change(1)
        features['returns_5d'] = lambda df: df['Close'].pct_change(5)
        features['returns_20d'] = lambda df: df['Close'].pct_change(20)
        
        # Moving averages
        features['sma_10'] = lambda df: df['Close'].rolling(10).mean()
        features['sma_20'] = lambda df: df['Close'].rolling(20).mean()
        features['sma_50'] = lambda df: df['Close'].rolling(50).mean()
        
        # Volatility features
        features['volatility_10d'] = lambda df: df['Close'].pct_change().rolling(10).std()
        features['volatility_20d'] = lambda df: df['Close'].pct_change().rolling(20).std()
        
        # Volume features
        features['volume_sma_10'] = lambda df: df['Volume'].rolling(10).mean()
        features['volume_ratio'] = lambda df: df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Price position features
        features['price_position_20d'] = lambda df: (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
        
        # Momentum features
        features['roc_10d'] = lambda df: (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
        features['roc_20d'] = lambda df: (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
        
        # Add technical indicators if talib is available
        if HAS_TALIB:
            features['rsi_14'] = lambda df: talib.RSI(df['Close'].values, timeperiod=14)
            features['macd'] = lambda df: talib.MACD(df['Close'].values)[0]
            features['bb_upper'] = lambda df: talib.BBANDS(df['Close'].values)[0]
            features['bb_lower'] = lambda df: talib.BBANDS(df['Close'].values)[2]
            features['atr_14'] = lambda df: talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
        
        return features
    
    def _initialize_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategy templates"""
        templates = {}
        
        # Momentum strategy template
        templates['momentum'] = {
            'type': StrategyType.MOMENTUM,
            'features': ['returns_1d', 'returns_5d', 'volume_ratio', 'rsi_14'],
            'parameters': {
                'lookback_period': [5, 10, 20],
                'momentum_threshold': [0.02, 0.05, 0.1],
                'volume_threshold': [1.5, 2.0, 3.0]
            }
        }
        
        # Mean reversion strategy template
        templates['mean_reversion'] = {
            'type': StrategyType.MEAN_REVERSION,
            'features': ['price_position_20d', 'rsi_14', 'bb_upper', 'bb_lower'],
            'parameters': {
                'oversold_threshold': [20, 25, 30],
                'overbought_threshold': [70, 75, 80],
                'mean_reversion_period': [10, 20, 50]
            }
        }
        
        # Breakout strategy template
        templates['breakout'] = {
            'type': StrategyType.BREAKOUT,
            'features': ['volatility_20d', 'volume_ratio', 'atr_14', 'price_position_20d'],
            'parameters': {
                'breakout_threshold': [1.5, 2.0, 2.5],
                'volume_confirmation': [1.5, 2.0, 3.0],
                'lookback_period': [20, 50, 100]
            }
        }
        
        # Volatility strategy template
        templates['volatility'] = {
            'type': StrategyType.VOLATILITY,
            'features': ['volatility_10d', 'volatility_20d', 'atr_14', 'returns_1d'],
            'parameters': {
                'volatility_threshold': [0.02, 0.03, 0.05],
                'volatility_period': [10, 20, 30],
                'direction_bias': ['long', 'short', 'both']
            }
        }
        
        return templates
    
    def _run(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Synchronous strategy discovery execution"""
        return asyncio.run(self._arun(action, parameters))
    
    async def _arun(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Asynchronous strategy discovery execution"""
        try:
            parameters = parameters or {}
            
            if action == 'discover_strategies':
                return await self._discover_strategies(parameters)
            elif action == 'backtest_strategy':
                return await self._backtest_strategy(parameters)
            elif action == 'optimize_strategy':
                return await self._optimize_strategy(parameters)
            elif action == 'validate_strategy':
                return await self._validate_strategy(parameters)
            elif action == 'get_discovered_strategies':
                return await self._get_discovered_strategies(parameters)
            elif action == 'generate_signals':
                return await self._generate_signals(parameters)
            elif action == 'get_performance_report':
                return await self._get_performance_report(parameters)
            elif action == 'export_strategy':
                return await self._export_strategy(parameters)
            else:
                return json.dumps({
                    'error': f'Unknown action: {action}',
                    'available_actions': [
                        'discover_strategies', 'backtest_strategy', 'optimize_strategy',
                        'validate_strategy', 'get_discovered_strategies', 'generate_signals',
                        'get_performance_report', 'export_strategy'
                    ]
                })
                
        except Exception as e:
            self.logger.error(f"Strategy discovery failed: {e}")
            return json.dumps({'error': str(e)})
    
    async def _discover_strategies(self, parameters: Dict[str, Any]) -> str:
        """Discover new trading strategies"""
        if not HAS_ML_LIBS:
            return json.dumps({'error': 'Machine learning libraries not available'})
        
        symbols = parameters.get('symbols', self.config.symbols_universe[:5])  # Limit for demo
        strategy_types = parameters.get('strategy_types', list(self.strategy_templates.keys()))
        
        discovered_strategies = []
        
        for symbol in symbols:
            self.logger.info(f"Discovering strategies for {symbol}")
            
            # Get historical data
            data = await self._get_training_data(symbol)
            if data is None or len(data) < 100:
                continue
            
            # Generate features
            features_df = await self._generate_features(data)
            if features_df is None:
                continue
            
            # Test each strategy template
            for template_name, template in self.strategy_templates.items():
                if template_name not in strategy_types:
                    continue
                
                try:
                    strategy = await self._test_strategy_template(
                        symbol, template_name, template, features_df, data
                    )
                    
                    if strategy and self._meets_performance_criteria(strategy.performance):
                        discovered_strategies.append(strategy)
                        self.discovered_strategies.append(strategy)
                        self.logger.info(f"Discovered strategy: {strategy.name}")
                
                except Exception as e:
                    self.logger.error(f"Failed to test {template_name} for {symbol}: {e}")
        
        # Update statistics
        self.discovery_stats['total_strategies_tested'] += len(symbols) * len(strategy_types)
        self.discovery_stats['strategies_discovered'] += len(discovered_strategies)
        self.discovery_stats['last_discovery_time'] = datetime.now()
        
        # Save discovered strategies
        await self._save_strategies(discovered_strategies)
        
        return json.dumps({
            'discovered_strategies': len(discovered_strategies),
            'strategies': [
                {
                    'strategy_id': s.strategy_id,
                    'name': s.name,
                    'type': s.strategy_type.value,
                    'sharpe_ratio': s.performance.sharpe_ratio,
                    'total_return': s.performance.total_return,
                    'win_rate': s.performance.win_rate,
                    'confidence_score': s.confidence_score
                }
                for s in discovered_strategies
            ],
            'symbols_tested': symbols,
            'discovery_timestamp': datetime.now().isoformat()
        }, indent=2)
    
    async def _test_strategy_template(
        self, 
        symbol: str, 
        template_name: str, 
        template: Dict[str, Any], 
        features_df: pd.DataFrame, 
        price_data: pd.DataFrame
    ) -> Optional[DiscoveredStrategy]:
        """Test a strategy template and return the best variant"""
        
        # Prepare features for this template
        template_features = [f for f in template['features'] if f in features_df.columns]
        if len(template_features) < 2:
            return None
        
        X = features_df[template_features].dropna()
        
        # Generate target variable (future returns)
        y = self._generate_target_variable(price_data, X.index)
        
        if len(X) != len(y) or len(X) < 50:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        
        # Train model
        model = self._train_model(X_train, y_train)
        if model is None:
            return None
        
        # Generate predictions
        predictions = model.predict(X_test)
        
        # Convert predictions to signals
        signals = self._predictions_to_signals(predictions, X_test.index, price_data)
        
        # Backtest the strategy
        performance = await self._backtest_signals(signals, price_data)
        
        if performance is None:
            return None
        
        # Create strategy
        strategy_id = hashlib.md5(f"{symbol}_{template_name}_{datetime.now()}".encode()).hexdigest()[:8]
        
        strategy = DiscoveredStrategy(
            strategy_id=strategy_id,
            name=f"{template_name.title()}_{symbol}_{strategy_id}",
            strategy_type=template['type'],
            description=f"ML-based {template_name} strategy for {symbol}",
            parameters=self._extract_model_parameters(model),
            features=template_features,
            performance=performance,
            confidence_score=self._calculate_strategy_confidence(performance, model, X_test, y_test),
            discovery_timestamp=datetime.now(),
            validation_status='discovered',
            code=self._generate_strategy_code(template_name, template_features, model),
            metadata={
                'symbol': symbol,
                'template': template_name,
                'model_type': type(model).__name__,
                'feature_importance': dict(zip(template_features, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
            }
        )
        
        return strategy
    
    async def _backtest_strategy(self, parameters: Dict[str, Any]) -> str:
        """Backtest a specific strategy"""
        strategy_id = parameters.get('strategy_id')
        if not strategy_id:
            return json.dumps({'error': 'strategy_id parameter required'})
        
        strategy = next((s for s in self.discovered_strategies if s.strategy_id == strategy_id), None)
        if not strategy:
            return json.dumps({'error': f'Strategy {strategy_id} not found'})
        
        symbol = parameters.get('symbol', strategy.metadata.get('symbol', 'AAPL'))
        start_date = parameters.get('start_date')
        end_date = parameters.get('end_date')
        
        # Get data for backtesting
        if start_date and end_date:
            start_date = datetime.fromisoformat(start_date)
            end_date = datetime.fromisoformat(end_date)
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.backtest_period_days)
        
        data = await self.data_manager.get_price_data(symbol, '1d', start_date, end_date)
        if data.data.empty:
            return json.dumps({'error': f'No data available for {symbol}'})
        
        # Generate signals using the strategy
        signals = await self._apply_strategy(strategy, data.data)
        
        # Backtest
        performance = await self._backtest_signals(signals, data.data)
        
        return json.dumps({
            'strategy_id': strategy_id,
            'symbol': symbol,
            'backtest_period': [start_date.isoformat(), end_date.isoformat()],
            'performance': {
                'total_return': performance.total_return,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate,
                'total_trades': performance.total_trades,
                'profit_factor': performance.profit_factor
            },
            'signals_generated': len(signals)
        }, indent=2)
    
    async def _generate_signals(self, parameters: Dict[str, Any]) -> str:
        """Generate trading signals using discovered strategies"""
        strategy_id = parameters.get('strategy_id')
        symbols = parameters.get('symbols', ['AAPL'])
        
        if strategy_id:
            strategies = [s for s in self.discovered_strategies if s.strategy_id == strategy_id]
        else:
            # Use top performing strategies
            strategies = sorted(
                self.discovered_strategies, 
                key=lambda x: x.performance.sharpe_ratio, 
                reverse=True
            )[:5]
        
        if not strategies:
            return json.dumps({'error': 'No strategies available'})
        
        all_signals = []
        
        for symbol in symbols:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = await self.data_manager.get_price_data(symbol, '1d', start_date, end_date)
            if data.data.empty:
                continue
            
            for strategy in strategies:
                try:
                    signals = await self._apply_strategy(strategy, data.data)
                    
                    # Get the latest signal
                    if signals:
                        latest_signal = signals[-1]
                        all_signals.append({
                            'symbol': symbol,
                            'strategy_id': strategy.strategy_id,
                            'strategy_name': strategy.name,
                            'signal_type': latest_signal.signal_type.value,
                            'confidence': latest_signal.confidence,
                            'timestamp': latest_signal.timestamp.isoformat(),
                            'price': latest_signal.price,
                            'features': latest_signal.features
                        })
                
                except Exception as e:
                    self.logger.error(f"Failed to generate signals for {symbol} with {strategy.name}: {e}")
        
        return json.dumps({
            'signals': all_signals,
            'total_signals': len(all_signals),
            'symbols_analyzed': symbols,
            'strategies_used': len(strategies),
            'generation_timestamp': datetime.now().isoformat()
        }, indent=2)
    
    async def _get_discovered_strategies(self, parameters: Dict[str, Any]) -> str:
        """Get list of discovered strategies"""
        limit = parameters.get('limit', 20)
        min_sharpe = parameters.get('min_sharpe_ratio', 0)
        strategy_type = parameters.get('strategy_type')
        
        # Filter strategies
        filtered_strategies = []
        for strategy in self.discovered_strategies:
            if strategy.performance.sharpe_ratio < min_sharpe:
                continue
            if strategy_type and strategy.strategy_type.value != strategy_type:
                continue
            filtered_strategies.append(strategy)
        
        # Sort by performance
        filtered_strategies.sort(key=lambda x: x.performance.sharpe_ratio, reverse=True)
        filtered_strategies = filtered_strategies[:limit]
        
        # Convert to JSON format
        strategies_data = []
        for strategy in filtered_strategies:
            strategies_data.append({
                'strategy_id': strategy.strategy_id,
                'name': strategy.name,
                'type': strategy.strategy_type.value,
                'description': strategy.description,
                'performance': {
                    'total_return': strategy.performance.total_return,
                    'sharpe_ratio': strategy.performance.sharpe_ratio,
                    'max_drawdown': strategy.performance.max_drawdown,
                    'win_rate': strategy.performance.win_rate,
                    'total_trades': strategy.performance.total_trades
                },
                'confidence_score': strategy.confidence_score,
                'discovery_timestamp': strategy.discovery_timestamp.isoformat(),
                'validation_status': strategy.validation_status,
                'features': strategy.features,
                'metadata': strategy.metadata
            })
        
        return json.dumps({
            'strategies': strategies_data,
            'total_discovered': len(self.discovered_strategies),
            'filtered_count': len(filtered_strategies),
            'discovery_stats': self.discovery_stats
        }, indent=2)
    
    # Helper methods
    
    async def _get_training_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get training data for strategy discovery"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.backtest_period_days + 100)
        
        try:
            data = await self.data_manager.get_price_data(symbol, '1d', start_date, end_date)
            return data.data if not data.data.empty else None
        except Exception as e:
            self.logger.error(f"Failed to get training data for {symbol}: {e}")
            return None
    
    async def _generate_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate features from price data"""
        try:
            features_df = pd.DataFrame(index=data.index)
            
            for feature_name, feature_func in self.feature_functions.items():
                try:
                    feature_values = feature_func(data)
                    if isinstance(feature_values, np.ndarray):
                        features_df[feature_name] = feature_values
                    else:
                        features_df[feature_name] = feature_values.values
                except Exception as e:
                    self.logger.warning(f"Failed to generate feature {feature_name}: {e}")
            
            return features_df.dropna()
        
        except Exception as e:
            self.logger.error(f"Failed to generate features: {e}")
            return None
    
    def _generate_target_variable(self, price_data: pd.DataFrame, feature_index: pd.Index) -> np.ndarray:
        """Generate target variable for supervised learning"""
        # Use future returns as target
        returns = price_data['Close'].pct_change(5).shift(-5)  # 5-day forward returns
        
        # Align with feature index
        aligned_returns = returns.reindex(feature_index).dropna()
        
        # Convert to classification target (1 for positive returns, 0 for negative)
        return (aligned_returns > 0.02).astype(int).values  # 2% threshold
    
    def _train_model(self, X: pd.DataFrame, y: np.ndarray) -> Optional[Any]:
        """Train machine learning model"""
        if not HAS_ML_LIBS:
            return None
        
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Try different models
            models = [
                RandomForestClassifier(n_estimators=100, random_state=42),
                GradientBoostingClassifier(random_state=42),
                LogisticRegression(random_state=42)
            ]
            
            best_model = None
            best_score = 0
            
            for model in models:
                try:
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                    avg_score = scores.mean()
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_model.fit(X_scaled, y)
                        best_model.scaler = scaler  # Store scaler with model
                
                except Exception as e:
                    self.logger.warning(f"Failed to train {type(model).__name__}: {e}")
            
            return best_model if best_score > 0.55 else None
        
        except Exception as e:
            self.logger.error(f"Failed to train model: {e}")
            return None
    
    def _predictions_to_signals(self, predictions: np.ndarray, index: pd.Index, price_data: pd.DataFrame) -> List[TradingSignal]:
        """Convert model predictions to trading signals"""
        signals = []
        
        for i, (idx, pred) in enumerate(zip(index, predictions)):
            if idx not in price_data.index:
                continue
            
            price = price_data.loc[idx, 'Close']
            
            # Convert prediction to signal
            if pred == 1:
                signal_type = SignalType.BUY
                confidence = 0.7  # Base confidence
            else:
                signal_type = SignalType.SELL
                confidence = 0.6
            
            signal = TradingSignal(
                symbol=price_data.attrs.get('symbol', 'UNKNOWN'),
                signal_type=signal_type,
                confidence=confidence,
                timestamp=idx,
                price=price,
                strategy_name='ML_Strategy',
                features={}
            )
            
            signals.append(signal)
        
        return signals
    
    async def _backtest_signals(self, signals: List[TradingSignal], price_data: pd.DataFrame) -> Optional[StrategyPerformance]:
        """Backtest trading signals"""
        if not signals:
            return None
        
        try:
            # Simple backtesting logic
            portfolio_value = 10000  # Starting capital
            position = 0
            trades = []
            equity_curve = []
            
            for signal in signals:
                if signal.timestamp not in price_data.index:
                    continue
                
                price = price_data.loc[signal.timestamp, 'Close']
                
                if signal.signal_type == SignalType.BUY and position <= 0:
                    # Buy signal
                    shares = portfolio_value // price
                    position = shares
                    portfolio_value -= shares * price
                    trades.append({'type': 'buy', 'price': price, 'timestamp': signal.timestamp})
                
                elif signal.signal_type == SignalType.SELL and position > 0:
                    # Sell signal
                    portfolio_value += position * price
                    trades.append({'type': 'sell', 'price': price, 'timestamp': signal.timestamp})
                    position = 0
                
                # Calculate current portfolio value
                current_value = portfolio_value + (position * price if position > 0 else 0)
                equity_curve.append(current_value)
            
            if not equity_curve:
                return None
            
            # Calculate performance metrics
            total_return = (equity_curve[-1] - 10000) / 10000
            
            # Calculate Sharpe ratio (simplified)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            max_drawdown = abs(np.min(drawdown))
            
            # Calculate win rate
            winning_trades = sum(1 for i in range(0, len(trades)-1, 2) 
                               if i+1 < len(trades) and trades[i+1]['price'] > trades[i]['price'])
            total_trade_pairs = len(trades) // 2
            win_rate = winning_trades / total_trade_pairs if total_trade_pairs > 0 else 0
            
            performance = StrategyPerformance(
                strategy_name='Discovered_Strategy',
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_trade_duration=5.0,  # Simplified
                total_trades=len(trades),
                profit_factor=1.0,  # Simplified
                calmar_ratio=total_return / max_drawdown if max_drawdown > 0 else 0,
                sortino_ratio=sharpe_ratio,  # Simplified
                volatility=np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                beta=1.0,  # Simplified
                alpha=0.0,  # Simplified
                information_ratio=0.0,  # Simplified
                backtest_period=(signals[0].timestamp, signals[-1].timestamp)
            )
            
            return performance
        
        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            return None
    
    def _meets_performance_criteria(self, performance: StrategyPerformance) -> bool:
        """Check if strategy meets performance criteria"""
        return (
            performance.sharpe_ratio >= self.config.min_sharpe_ratio and
            performance.win_rate >= self.config.min_win_rate and
            performance.max_drawdown <= self.config.max_drawdown and
            performance.total_trades >= self.config.min_trades
        )
    
    def _calculate_strategy_confidence(self, performance: StrategyPerformance, model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> float:
        """Calculate confidence score for discovered strategy"""
        # Base confidence on multiple factors
        performance_score = min(1.0, performance.sharpe_ratio / 2.0)
        
        # Model accuracy
        if hasattr(model, 'predict'):
            try:
                X_scaled = model.scaler.transform(X_test) if hasattr(model, 'scaler') else X_test
                predictions = model.predict(X_scaled)
                accuracy = accuracy_score(y_test, predictions)
            except:
                accuracy = 0.5
        else:
            accuracy = 0.5
        
        # Trade frequency (not too high, not too low)
        trade_frequency_score = min(1.0, performance.total_trades / 100)
        
        # Combine scores
        confidence = (performance_score * 0.4 + accuracy * 0.4 + trade_frequency_score * 0.2)
        
        return min(0.95, max(0.1, confidence))
    
    def _extract_model_parameters(self, model: Any) -> Dict[str, Any]:
        """Extract parameters from trained model"""
        params = {}
        
        if hasattr(model, 'get_params'):
            params.update(model.get_params())
        
        if hasattr(model, 'feature_importances_'):
            params['feature_importances'] = model.feature_importances_.tolist()
        
        return params
    
    def _generate_strategy_code(self, template_name: str, features: List[str], model: Any) -> str:
        """Generate executable strategy code"""
        return f"""
# Auto-generated {template_name} strategy
def generate_signal(data):
    # Extract features: {features}
    # Model: {type(model).__name__}
    # This is a placeholder - implement actual strategy logic
    return 'hold'
"""
    
    async def _apply_strategy(self, strategy: DiscoveredStrategy, data: pd.DataFrame) -> List[TradingSignal]:
        """Apply a discovered strategy to generate signals"""
        # This is a simplified implementation
        # In practice, you would execute the strategy's code
        
        signals = []
        
        # Generate some sample signals based on strategy type
        if strategy.strategy_type == StrategyType.MOMENTUM:
            # Simple momentum signals
            returns = data['Close'].pct_change(5)
            for i, (idx, ret) in enumerate(returns.items()):
                if i < len(returns) - 1 and not pd.isna(ret):
                    if ret > 0.05:  # 5% momentum
                        signal = TradingSignal(
                            symbol=strategy.metadata.get('symbol', 'UNKNOWN'),
                            signal_type=SignalType.BUY,
                            confidence=0.7,
                            timestamp=idx,
                            price=data.loc[idx, 'Close'],
                            strategy_name=strategy.name,
                            features={'momentum': ret}
                        )
                        signals.append(signal)
        
        return signals
    
    async def _save_strategies(self, strategies: List[DiscoveredStrategy]):
        """Save discovered strategies to cache"""
        for strategy in strategies:
            cache_file = self.strategy_cache_path / f"{strategy.strategy_id}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(strategy, f)
            except Exception as e:
                self.logger.error(f"Failed to save strategy {strategy.strategy_id}: {e}")
    
    def _load_cached_strategies(self):
        """Load cached strategies"""
        if not self.strategy_cache_path.exists():
            return
        
        for cache_file in self.strategy_cache_path.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    strategy = pickle.load(f)
                    self.discovered_strategies.append(strategy)
            except Exception as e:
                self.logger.error(f"Failed to load cached strategy {cache_file}: {e}")


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from ..core.config_manager import ConfigManager
    from ..core.data_manager import UnifiedDataManager
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    discovery_tool = AutomatedStrategyDiscoveryTool(config_manager, data_manager)
    
    # Test strategy discovery
    result = discovery_tool._run('discover_strategies', {'symbols': ['AAPL']})
    print("Strategy Discovery:", result)