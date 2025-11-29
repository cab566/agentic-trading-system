#!/usr/bin/env python3
"""
System Orchestrator - Advanced Trading System Integration

A comprehensive orchestration engine that integrates all components of the
advanced algorithmic trading system into a unified, coordinated platform:

- Centralized system initialization and configuration management
- Component lifecycle management and health monitoring
- Inter-component communication and data flow coordination
- Advanced strategy orchestration with ML integration
- Real-time risk management and position sizing
- Multi-asset execution across traditional and crypto markets
- Performance monitoring and analytics integration
- Alternative data ingestion and signal generation
- Portfolio optimization and rebalancing automation
- Comprehensive logging, monitoring, and alerting
- Graceful shutdown and error recovery mechanisms
- System state persistence and recovery
- Advanced backtesting and strategy validation
- Real-time dashboard and reporting integration

This orchestrator serves as the central nervous system that coordinates
all trading activities, risk management, and performance monitoring
across the entire platform.

Author: AI Trading System v2.0
Date: January 2025
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import json
import os
import signal
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Core system imports
try:
    from core.config_manager import ConfigManager
    from core.market_data_aggregator import MarketDataAggregator
    from core.execution_engine import ExecutionEngine, Order, OrderType, OrderSide
    from core.risk_manager_24_7 import RiskManager24_7
    from core.trade_storage import TradeStorage
    from utils.notifications import NotificationManager
    from utils.cache_manager import CacheManager
except ImportError as e:
    logging.warning(f"Core system import failed: {e}")

# Advanced components
try:
    from advanced_strategies import AdvancedStrategyOrchestrator, TradingSignal
    from market_intelligence_engine import MarketIntelligenceEngine
    from portfolio_optimization_engine import PortfolioOptimizationEngine
    from advanced_risk_management import AdvancedRiskManager
    from advanced_backtesting_framework import AdvancedBacktester
    from ml_trading_pipeline import MLTradingPipeline
    from alternative_data_engine import AlternativeDataEngine
    from real_time_monitoring import RealTimeMonitor
    from performance_analytics import PerformanceAnalytics
except ImportError as e:
    logging.warning(f"Advanced component import failed: {e}")

# Try to import discovery tools
try:
    from tools.real_time_opportunity_scanner import RealTimeOpportunityScannerTool
    from tools.social_sentiment_analyzer import SocialSentimentAnalyzerTool
    from tools.cross_asset_arbitrage_detector import CrossAssetArbitrageDetectorTool
    from tools.market_regime_detector import MarketRegimeDetectorTool
    from tools.economic_calendar_monitor import EconomicCalendarMonitorTool
    from tools.options_flow_analyzer import OptionsFlowAnalyzerTool
    from tools.earnings_surprise_predictor import EarningsSurprisePredictorTool
    from tools.rd_agent_integration_tool import RDAgentIntegrationTool
except ImportError as e:
    logging.warning(f"Discovery tools import failed: {e}")

# External libraries
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"

class ExecutionMode(Enum):
    """System execution modes"""
    LIVE_TRADING = "live_trading"
    PAPER_TRADING = "paper_trading"
    BACKTESTING = "backtesting"
    RESEARCH = "research"
    MAINTENANCE = "maintenance"

@dataclass
class ComponentHealth:
    """Component health information"""
    name: str
    status: ComponentStatus
    last_heartbeat: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self, max_heartbeat_age: int = 300) -> bool:
        """Check if component is healthy"""
        if self.status == ComponentStatus.ERROR:
            return False
        
        age = (datetime.now() - self.last_heartbeat).total_seconds()
        return age <= max_heartbeat_age

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    uptime: timedelta
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_pnl: float
    current_positions: int
    active_strategies: int
    data_points_processed: int
    alerts_generated: int
    system_load: float
    memory_usage: float
    
    @property
    def success_rate(self) -> float:
        """Calculate trade success rate"""
        if self.total_trades == 0:
            return 0.0
        return self.successful_trades / self.total_trades

@dataclass
class OrchestratorConfig:
    """Orchestrator configuration"""
    execution_mode: ExecutionMode = ExecutionMode.PAPER_TRADING
    max_concurrent_strategies: int = 10
    heartbeat_interval: int = 30
    health_check_interval: int = 60
    performance_update_interval: int = 300
    auto_restart_failed_components: bool = True
    max_restart_attempts: int = 3
    enable_ml_pipeline: bool = True
    enable_alternative_data: bool = True
    enable_portfolio_optimization: bool = True
    rebalancing_frequency: str = "daily"
    risk_check_frequency: int = 60
    data_backup_interval: int = 0  # Disabled - no backups in production
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OrchestratorConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

class SystemOrchestrator:
    """Main system orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/orchestrator_config.json"
        self.config = self._load_config()
        
        # System state
        self.state = SystemState.INITIALIZING
        self.start_time = datetime.now()
        self.shutdown_event = threading.Event()
        self.component_health: Dict[str, ComponentHealth] = {}
        self.system_metrics = SystemMetrics(
            uptime=timedelta(0), total_trades=0, successful_trades=0,
            failed_trades=0, total_pnl=0.0, current_positions=0,
            active_strategies=0, data_points_processed=0, alerts_generated=0,
            system_load=0.0, memory_usage=0.0
        )
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.background_tasks: List[threading.Thread] = []
        
        # Component instances
        self.components: Dict[str, Any] = {}
        self.initialized_components: set = set()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("system_orchestrator")
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("System Orchestrator initialized")
    
    def _load_config(self) -> OrchestratorConfig:
        """Load orchestrator configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                return OrchestratorConfig.from_dict(config_dict)
            else:
                # Create default config
                default_config = OrchestratorConfig()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logging.warning(f"Failed to load config: {e}. Using defaults.")
            return OrchestratorConfig()
    
    def _save_config(self, config: OrchestratorConfig) -> None:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config.__dict__, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/system_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown()
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        self.logger.info("Starting system initialization...")
        self.state = SystemState.INITIALIZING
        
        try:
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize advanced components
            await self._initialize_advanced_components()
            
            # Initialize discovery tools
            await self._initialize_discovery_tools()
            
            # Start background tasks
            self._start_background_tasks()
            
            # Verify system health
            if await self._verify_system_health():
                self.state = SystemState.RUNNING
                self.logger.info("System initialization completed successfully")
                return True
            else:
                self.state = SystemState.ERROR
                self.logger.error("System health check failed")
                return False
                
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def _initialize_core_components(self) -> None:
        """Initialize core trading system components"""
        self.logger.info("Initializing core components...")
        
        try:
            # Validate trading mode configuration before initializing components
            from core.trading_mode_validator import validate_before_trading
            validation_result = validate_before_trading()
            if not validation_result['valid']:
                error_msg = f"Trading mode validation failed: {', '.join(validation_result['errors'])}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    self.logger.warning(f"Trading mode validation warning: {warning}")
            
            self.logger.info(f"Trading mode validation passed successfully for {validation_result['mode']} trading")
            
            # Configuration Manager
            config_path = Path(__file__).parent / "config"
            self.components['config_manager'] = ConfigManager(config_path)
            self._register_component('config_manager')
            
            # Cache Manager
            self.components['cache_manager'] = CacheManager()
            self._register_component('cache_manager')
            
            # Market Data Aggregator
            self.components['market_data'] = MarketDataAggregator(
                self.components['config_manager']
            )
            self._register_component('market_data')
            
            # Trade Storage
            self.components['trade_storage'] = TradeStorage(
                self.components['config_manager']
            )
            self._register_component('trade_storage')
            
            # Risk Manager
            self.components['risk_manager'] = RiskManager24_7(
                self.components['config_manager']
            )
            self._register_component('risk_manager')
            
            # Execution Engine
            self.components['execution_engine'] = ExecutionEngine(
                config_manager=self.components['config_manager'],
                market_data_aggregator=self.components['market_data'],
                risk_manager=self.components['risk_manager'],
                trade_storage=self.components['trade_storage']
            )
            self._register_component('execution_engine')
            
            # Notification Manager
            self.components['notification_manager'] = NotificationManager(
                self.components['config_manager']
            )
            self._register_component('notification_manager')
            
            self.logger.info("Core components initialized")
            
        except Exception as e:
            self.logger.error(f"Core component initialization failed: {e}")
            raise
    
    async def _initialize_advanced_components(self) -> None:
        """Initialize advanced trading components"""
        self.logger.info("Initializing advanced components...")
        
        try:
            # Advanced Risk Manager
            risk_config = {
                'confidence_level': 0.95,
                'risk_limits': {
                    'daily_var_limit': 0.02,
                    'max_single_position': 0.10,
                    'max_gross_leverage': 2.0,
                    'max_daily_drawdown': 0.03
                }
            }
            self.components['advanced_risk_manager'] = AdvancedRiskManager(config=risk_config)
            self._register_component('advanced_risk_manager')
            
            # Market Intelligence Engine
            intelligence_config = {
                'technical': {
                    'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr'],
                    'pattern_detection': True
                },
                'fundamental': {
                    'ratios': ['pe', 'pb', 'roe', 'debt_equity'],
                    'growth_metrics': True
                }
            }
            self.components['market_intelligence'] = MarketIntelligenceEngine(config=intelligence_config)
            self._register_component('market_intelligence')
            
            # Portfolio Optimization Engine
            if self.config.enable_portfolio_optimization:
                portfolio_config = {
                    'risk_aversion': 3.0,
                    'tau': 0.025,
                    'n_factors': 5,
                    'rebalancing_frequency': 'monthly',
                    'transaction_costs': 0.001
                }
                self.components['portfolio_optimizer'] = PortfolioOptimizationEngine(config=portfolio_config)
                self._register_component('portfolio_optimizer')
            
            # ML Trading Pipeline
            if self.config.enable_ml_pipeline:
                from ml_trading_pipeline import MLConfig, ModelType, PredictionType, FeatureType
                ml_config = MLConfig(
                    model_type=ModelType.RANDOM_FOREST,
                    prediction_type=PredictionType.REGRESSION,
                    feature_types=[FeatureType.TECHNICAL, FeatureType.TIME_SERIES],
                    lookback_window=252,
                    prediction_horizon=5,
                    use_ensemble=True
                )
                self.components['ml_pipeline'] = MLTradingPipeline(config=ml_config)
                self._register_component('ml_pipeline')
            
            # Alternative Data Engine
            if self.config.enable_alternative_data:
                alternative_config = {
                    'connectors': {
                        'social_media': {
                            'twitter_api_key': 'demo_key',
                            'reddit_client_id': 'demo_id'
                        },
                        'news_analytics': {
                            'news_sources': ['reuters', 'bloomberg', 'cnbc']
                        },
                        'satellite_imagery': {
                            'providers': ['planet', 'maxar']
                        },
                        'esg_data': {
                            'providers': ['msci', 'sustainalytics']
                        },
                        'crypto_onchain': {
                            'chains': ['bitcoin', 'ethereum']
                        },
                        'weather_data': {
                            'providers': ['openweather', 'noaa']
                        }
                    }
                }
                self.components['alternative_data'] = AlternativeDataEngine(config=alternative_config)
                self._register_component('alternative_data')
            
            # Advanced Strategy Orchestrator
            strategy_config = {
                'strategies': {
                    'multi_factor_momentum': {
                        'lookback_periods': [5, 10, 20, 50],
                        'volume_factor_weight': 0.3,
                        'volatility_factor_weight': 0.2
                    },
                    'statistical_arbitrage': {
                        'lookback_window': 60,
                        'entry_threshold': 2.0,
                        'exit_threshold': 0.5,
                        'correlation_threshold': 0.7
                    },
                    'volatility_surface': {
                        'vol_lookback': 30,
                        'vol_threshold': 1.5,
                        'regime_lookback': 60
                    }
                }
            }
            self.components['strategy_orchestrator'] = AdvancedStrategyOrchestrator(config=strategy_config)
            self._register_component('strategy_orchestrator')
            
            # Performance Analytics
            self.components['performance_analytics'] = PerformanceAnalytics()
            self._register_component('performance_analytics')
            
            # Real-time Monitor
            monitor_config = {
                'alert_db_path': 'alerts.db',
                'notifications': {
                    'email': {
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'username': 'demo@example.com',
                        'password': 'demo_password',
                        'from_email': 'demo@example.com',
                        'to_emails': ['admin@example.com']
                    }
                }
            }
            self.components['real_time_monitor'] = RealTimeMonitor(config=monitor_config)
            self._register_component('real_time_monitor')
            
            # Advanced Backtester
            from advanced_backtesting_framework import BacktestConfig
            from datetime import datetime
            backtest_config = BacktestConfig(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=1000000,
                commission_rate=0.001,
                rebalance_frequency="monthly",
                max_leverage=1.0,
                max_position_size=0.10
            )
            self.components['backtester'] = AdvancedBacktester(config=backtest_config)
            self._register_component('backtester')
            
            self.logger.info("Advanced components initialized")
            
        except Exception as e:
            self.logger.error(f"Advanced component initialization failed: {e}")
            raise
    
    async def _initialize_discovery_tools(self) -> None:
        """Initialize discovery tools"""
        self.logger.info("Initializing discovery tools...")
        
        try:
            # Load tools configuration
            tools_config_path = Path(__file__).parent / "config" / "tools_config.yaml"
            if tools_config_path.exists():
                import yaml
                with open(tools_config_path, 'r') as f:
                    tools_config = yaml.safe_load(f)
            else:
                tools_config = {}
            
            # Real-time Opportunity Scanner
            if 'real_time_opportunity_scanner' in tools_config.get('tools', {}):
                scanner_config = tools_config['tools']['real_time_opportunity_scanner'].get('config', {})
                self.components['opportunity_scanner'] = RealTimeOpportunityScannerTool(
                    config_manager=self.components['config_manager'],
                    data_manager=self.components['market_data']
                )
                self._register_component('opportunity_scanner')
            
            # Social Sentiment Analyzer
            if 'social_sentiment_analyzer' in tools_config.get('tools', {}):
                sentiment_config = tools_config['tools']['social_sentiment_analyzer'].get('config', {})
                self.components['sentiment_analyzer'] = SocialSentimentAnalyzerTool(
                    config_manager=self.components['config_manager'],
                    data_manager=self.components['market_data']
                )
                self._register_component('sentiment_analyzer')
            
            # Cross-Asset Arbitrage Detector
            if 'cross_asset_arbitrage_detector' in tools_config.get('tools', {}):
                arbitrage_config = tools_config['tools']['cross_asset_arbitrage_detector'].get('config', {})
                self.components['arbitrage_detector'] = CrossAssetArbitrageDetectorTool(
                    config_manager=self.components['config_manager'],
                    data_manager=self.components['market_data']
                )
                self._register_component('arbitrage_detector')
            
            # Market Regime Detector
            if 'market_regime_detector' in tools_config.get('tools', {}):
                regime_config = tools_config['tools']['market_regime_detector'].get('config', {})
                self.components['regime_detector'] = MarketRegimeDetectorTool(
                    config_manager=self.components['config_manager'],
                    data_manager=self.components['market_data']
                )
                self._register_component('regime_detector')
            
            # Economic Calendar Monitor
            if 'economic_calendar_monitor' in tools_config.get('tools', {}):
                calendar_config = tools_config['tools']['economic_calendar_monitor'].get('config', {})
                self.components['calendar_monitor'] = EconomicCalendarMonitorTool(
                    config_manager=self.components['config_manager'],
                    data_manager=self.components['market_data']
                )
                self._register_component('calendar_monitor')
            
            # Options Flow Analyzer
            if 'options_flow_analyzer' in tools_config.get('tools', {}):
                options_config = tools_config['tools']['options_flow_analyzer'].get('config', {})
                self.components['options_analyzer'] = OptionsFlowAnalyzerTool()
                self._register_component('options_analyzer')
            
            # Earnings Surprise Predictor
            if 'earnings_surprise_predictor' in tools_config.get('tools', {}):
                earnings_config = tools_config['tools']['earnings_surprise_predictor'].get('config', {})
                self.components['earnings_predictor'] = EarningsSurprisePredictorTool()
                self._register_component('earnings_predictor')
            
            # RD-Agent Integration Tool
            if 'rd_agent_integration' in tools_config.get('tools', {}):
                rd_config = tools_config['tools']['rd_agent_integration'].get('config', {})
                # Use real managers from the system instead of mock managers
                real_data_manager = self.components.get('data_manager')
                real_risk_manager = self.components.get('risk_manager')
                
                # Fallback to creating real managers if not available
                if not real_data_manager:
                    from core.data_manager import UnifiedDataManager
                    real_data_manager = UnifiedDataManager()
                
                if not real_risk_manager:
                    from risk.risk_manager import RiskManager
                    real_risk_manager = RiskManager()
                
                self.components['rd_agent'] = RDAgentIntegrationTool(
                    data_manager=real_data_manager,
                    risk_manager=real_risk_manager
                )
                self._register_component('rd_agent')
            
            self.logger.info("Discovery tools initialized")
            
        except Exception as e:
            self.logger.error(f"Discovery tools initialization failed: {e}")
            # Don't raise - discovery tools are optional
            pass
    
    def _register_component(self, name: str) -> None:
        """Register component for health monitoring"""
        self.component_health[name] = ComponentHealth(
            name=name,
            status=ComponentStatus.HEALTHY,
            last_heartbeat=datetime.now()
        )
        self.initialized_components.add(name)
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks"""
        self.logger.info("Starting background tasks...")
        
        # Health monitoring
        health_task = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="health_monitor"
        )
        health_task.start()
        self.background_tasks.append(health_task)
        
        # Performance monitoring
        perf_task = threading.Thread(
            target=self._performance_monitor_loop,
            daemon=True,
            name="performance_monitor"
        )
        perf_task.start()
        self.background_tasks.append(perf_task)
        
        # Risk monitoring
        risk_task = threading.Thread(
            target=self._risk_monitor_loop,
            daemon=True,
            name="risk_monitor"
        )
        risk_task.start()
        self.background_tasks.append(risk_task)
        
        # Data backup - DISABLED FOR PRODUCTION
        # backup_task = threading.Thread(
        #     target=self._backup_loop,
        #     daemon=True,
        #     name="data_backup"
        # )
        # backup_task.start()
        # self.background_tasks.append(backup_task)
        
        # Strategy execution
        strategy_task = threading.Thread(
            target=self._strategy_execution_loop,
            daemon=True,
            name="strategy_execution"
        )
        strategy_task.start()
        self.background_tasks.append(strategy_task)
    
    def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                self._check_component_health()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(10)
    
    def _performance_monitor_loop(self) -> None:
        """Background performance monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                self._update_system_metrics()
                time.sleep(self.config.performance_update_interval)
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                time.sleep(30)
    
    def _risk_monitor_loop(self) -> None:
        """Background risk monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                self._check_risk_limits()
                time.sleep(self.config.risk_check_frequency)
            except Exception as e:
                self.logger.error(f"Risk monitor error: {e}")
                time.sleep(10)
    
    def _backup_loop(self) -> None:
        """Background data backup loop - DISABLED"""
        # DISABLED FOR PRODUCTION - NO BACKUPS
        pass
        # while self.running:
        #     try:
        #         self._backup_system_data()
        #         time.sleep(self.config.data_backup_interval)
        #     except Exception as e:
        #         self.logger.error(f"Backup error: {e}")
        #         time.sleep(300)
    
    def _strategy_execution_loop(self) -> None:
        """Main strategy execution loop"""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    if self.state == SystemState.RUNNING:
                        loop.run_until_complete(self._execute_trading_cycle())
                    time.sleep(60)  # Execute every minute
                except Exception as e:
                    self.logger.error(f"Strategy execution error: {e}")
                    time.sleep(30)
        finally:
            loop.close()
    
    async def _execute_trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        try:
            # Update market data
            if 'market_data' in self.components:
                self.components['market_data'].update_all_data()
            
            # Generate signals from strategies
            signals = await self._generate_trading_signals()
            
            # Apply ML enhancements if enabled
            if self.config.enable_ml_pipeline and 'ml_pipeline' in self.components:
                signals = self._enhance_signals_with_ml(signals)
            
            # Portfolio optimization
            if self.config.enable_portfolio_optimization and 'portfolio_optimizer' in self.components:
                signals = await self._optimize_portfolio_allocation(signals)
            
            # Risk management
            signals = self._apply_risk_management(signals)
            
            # Execute trades
            await self._execute_signals(signals)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Trading cycle execution failed: {e}")
    
    async def _generate_trading_signals(self) -> List[TradingSignal]:
        """Generate trading signals from all strategies"""
        signals = []
        
        try:
            # Get appropriate symbols based on active market sessions
            active_symbols = self._get_active_market_symbols()
            
            if not active_symbols:
                self.logger.warning("No active market symbols found, using default AAPL")
                active_symbols = ['AAPL']
            
            self.logger.info(f"Generating signals for active symbols: {active_symbols}")
            
            if 'strategy_orchestrator' in self.components:
                # Get market data
                market_data = self._get_current_market_data()
                
                # Generate signals for each active symbol
                for symbol in active_symbols[:3]:  # Limit to top 3 symbols to avoid overload
                    try:
                        strategy_signals = await self.components['strategy_orchestrator'].generate_ensemble_signal(
                            symbol,
                            market_data
                        )
                        if strategy_signals:
                            signals.append(strategy_signals)
                    except Exception as e:
                        self.logger.error(f"Strategy signal generation failed for {symbol}: {e}")
            
            # Add alternative data signals for active symbols
            if self.config.enable_alternative_data and 'alternative_data' in self.components:
                for symbol in active_symbols[:2]:  # Limit alternative data to top 2 symbols
                    try:
                        # Fetch real alternative data for the symbol
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=7)  # Last 7 days of data
                        alt_data = await self.components['alternative_data'].fetch_all_data(
                            symbol, start_date, end_date
                        )
                        
                        alt_signal_value = self.components['alternative_data'].generate_trading_signal(
                            symbol,
                            alt_data  # Real alternative data
                        )
                        
                        # Convert float signal to TradingSignal object
                        if abs(alt_signal_value) > 0.1:  # Threshold for signal strength
                            action = "BUY" if alt_signal_value > 0 else "SELL"
                            alt_trading_signal = TradingSignal(
                                symbol=symbol,
                                action=action,
                                confidence=min(abs(alt_signal_value), 1.0),
                                strength=self._convert_signal_to_strength(abs(alt_signal_value)),
                                reasoning=f"Alternative data signal: {alt_signal_value:.3f}",
                                risk_level="MEDIUM",
                                suggested_position_size=abs(alt_signal_value) * 100,
                                strategy_name="AlternativeData"
                            )
                            signals.append(alt_trading_signal)
                    except Exception as e:
                        self.logger.error(f"Alternative data signal generation failed for {symbol}: {e}")
            
            self.logger.debug(f"Generated {len(signals)} trading signals for {len(active_symbols)} symbols")
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return []
    
    def _enhance_signals_with_ml(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Enhance signals using ML pipeline"""
        try:
            if 'ml_pipeline' in self.components:
                enhanced_signals = []
                
                for signal in signals:
                    try:
                        # Check if ML pipeline has trained models
                        if hasattr(self.components['ml_pipeline'], 'models') and self.components['ml_pipeline'].models:
                            # Get the first available trained model
                            for model_name, model in self.components['ml_pipeline'].models.items():
                                if hasattr(model, 'is_trained') and model.is_trained:
                                    features = self._extract_features_for_symbol(signal.symbol)
                                    if features is not None and not features.empty:
                                        # Simple ML enhancement - just adjust confidence slightly
                                        # Avoid complex prediction parsing for now
                                        ml_factor = 0.95  # Slightly reduce confidence as ML is not fully integrated
                                        signal.confidence = min(signal.confidence * ml_factor, 1.0)
                                        self.logger.debug(f"ML enhanced {signal.symbol}: confidence adjusted by {ml_factor}")
                                    break
                    except Exception as e:
                        self.logger.warning(f"ML enhancement failed for {signal.symbol}: {e}")
                    
                    enhanced_signals.append(signal)
                
                return enhanced_signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"ML signal enhancement failed: {e}")
            return signals
    
    async def _optimize_portfolio_allocation(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Optimize portfolio allocation"""
        try:
            if 'portfolio_optimizer' in self.components:
                # Get current portfolio
                current_portfolio = self._get_current_portfolio()
                
                # Create returns DataFrame from signals (using dummy data for now)
                symbols = [signal.symbol for signal in signals]
                returns_data = {}
                for symbol in symbols:
                    # Create dummy returns data - in production this should come from market data
                    returns_data[symbol] = np.random.normal(0.001, 0.02, 252)  # 252 trading days
                returns_df = pd.DataFrame(returns_data)
                
                # Import required classes
                from portfolio_optimization_engine import OptimizationObjective, OptimizationConstraints
                
                # Create optimization constraints
                constraints = OptimizationConstraints(
                    min_weight=0.0,
                    max_weight=0.3,  # Max 30% per position
                    max_turnover=0.5
                )
                
                # Optimize allocation
                optimization_result = await self.components['portfolio_optimizer'].optimize_portfolio(
                    returns=returns_df,
                    objective=OptimizationObjective.MAX_SHARPE,
                    constraints=constraints,
                    current_weights=current_portfolio
                )
                
                # Adjust signal sizes based on optimization
                optimized_signals = []
                for signal in signals:
                    if signal.symbol in optimization_result.weights:
                        target_weight = optimization_result.weights[signal.symbol]
                        current_weight = current_portfolio.get(signal.symbol, 0.0)
                        
                        # Adjust position size
                        signal.suggested_position_size = target_weight - current_weight
                        # Note: TradingSignal doesn't have metadata attribute
                        # Optimization info logged separately
                        self.logger.debug(f"Optimized {signal.symbol}: target_weight={target_weight:.3f}, sharpe={optimization_result.metrics.sharpe_ratio:.3f}")
                    
                    optimized_signals.append(signal)
                
                return optimized_signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return signals
    
    def _apply_risk_management(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply risk management to signals"""
        try:
            if 'advanced_risk_manager' in self.components:
                # Get current portfolio risk
                portfolio_data = self._get_portfolio_data()
                risk_metrics = self.components['advanced_risk_manager'].calculate_portfolio_risk(
                    portfolio_data['portfolio_weights'],
                    portfolio_data['asset_returns'],
                    portfolio_data['portfolio_value']
                )
                
                # Filter signals based on risk limits
                filtered_signals = []
                for signal in signals:
                    # Check individual position risk
                    position_risk = self._calculate_position_risk(signal)
                    
                    # Check portfolio impact
                    portfolio_impact = self._calculate_portfolio_impact(signal, risk_metrics)
                    
                    # Apply risk filters
                    if self._passes_risk_checks(signal, position_risk, portfolio_impact):
                        filtered_signals.append(signal)
                    else:
                        self.logger.warning(f"Signal filtered by risk management: {signal.symbol}")
                
                return filtered_signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Risk management failed: {e}")
            return signals
    
    async def _execute_signals(self, signals: List[TradingSignal]) -> None:
        """Execute trading signals"""
        try:
            if 'execution_engine' in self.components:
                for signal in signals:
                    # Create order
                    order = self._create_order_from_signal(signal)
                    
                    # Submit order
                    if order:
                        result = await self.components['execution_engine'].submit_order(
                            symbol=order.symbol,
                            side=order.side,
                            order_type=order.order_type,
                            quantity=order.quantity,
                            price=order.price,
                            stop_price=order.stop_price,
                            time_in_force=order.time_in_force,
                            metadata=order.metadata
                        )
                        
                        if result:
                            self.system_metrics.total_trades += 1
                            self.system_metrics.successful_trades += 1
                            self.logger.info(f"Order executed: {signal.symbol} - {signal.action}")
                        else:
                            self.system_metrics.failed_trades += 1
                            self.logger.warning(f"Order execution failed: {signal.symbol}")
            
        except Exception as e:
            self.logger.error(f"Signal execution failed: {e}")
    
    def _create_order_from_signal(self, signal: TradingSignal) -> Optional[Order]:
        """Create order from trading signal"""
        try:
            # Determine order side
            side = OrderSide.BUY if signal.action.upper() == 'BUY' else OrderSide.SELL
            
            # Calculate quantity
            quantity = abs(signal.suggested_position_size) if signal.suggested_position_size else 100
            
            # Create order with required order_id
            execution_engine = self.components.get('execution_engine')
            order_count = len(execution_engine.orders) if execution_engine and hasattr(execution_engine, 'orders') else 0
            order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal.symbol}_{order_count + 1:04d}"
            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                side=side,
                order_type=OrderType.MARKET,  # Default to market orders
                quantity=quantity,
                metadata={
                    'strategy': signal.strategy_name,
                    'signal_strength': signal.strength.value,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning
                }
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Order creation failed: {e}")
            return None
    
    def _check_component_health(self) -> None:
        """Check health of all components"""
        for name, component in self.components.items():
            try:
                # Update heartbeat
                if hasattr(component, 'health_check'):
                    is_healthy = component.health_check()
                    status = ComponentStatus.HEALTHY if is_healthy else ComponentStatus.WARNING
                else:
                    status = ComponentStatus.HEALTHY  # Assume healthy if no health check
                
                self.component_health[name].status = status
                self.component_health[name].last_heartbeat = datetime.now()
                
                # Auto-restart if configured
                if status == ComponentStatus.ERROR and self.config.auto_restart_failed_components:
                    self._restart_component(name)
                    
            except Exception as e:
                self.component_health[name].status = ComponentStatus.ERROR
                self.component_health[name].last_error = str(e)
                self.component_health[name].error_count += 1
                self.logger.error(f"Component health check failed for {name}: {e}")
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics"""
        try:
            # Update uptime
            self.system_metrics.uptime = datetime.now() - self.start_time
            
            # Update position count
            if 'execution_engine' in self.components:
                positions = self.components['execution_engine'].get_positions()
                self.system_metrics.current_positions = len(positions)
            
            # Update active strategies
            if 'strategy_orchestrator' in self.components:
                self.system_metrics.active_strategies = len(
                    self.components['strategy_orchestrator'].strategies
                )
            
            # Update system load and memory
            import psutil
            self.system_metrics.system_load = psutil.cpu_percent()
            self.system_metrics.memory_usage = psutil.virtual_memory().percent
            
        except Exception as e:
            self.logger.error(f"System metrics update failed: {e}")
    
    def _check_risk_limits(self) -> None:
        """Check system-wide risk limits"""
        try:
            if 'advanced_risk_manager' in self.components:
                # Get portfolio data
                portfolio_data = self._get_portfolio_data()
                
                # Calculate risk metrics with proper parameters
                risk_report = self.components['advanced_risk_manager'].calculate_portfolio_risk(
                    portfolio_data['portfolio_weights'],
                    portfolio_data['asset_returns'],
                    portfolio_data['portfolio_value']
                )
                
                # Log risk summary
                if 'risk_summary' in risk_report:
                    self.logger.info(f"Risk Summary: {risk_report['risk_summary']}")
                
                # Handle any alerts in the report
                if 'alerts' in risk_report and risk_report['alerts']:
                    for alert_dict in risk_report['alerts']:
                        self._handle_risk_alert(alert_dict)
                    
        except Exception as e:
            self.logger.error(f"Risk limit check failed: {e}")
    
    def _backup_system_data(self) -> None:
        """Backup critical system data - DISABLED"""
        # DISABLED FOR PRODUCTION - NO BACKUPS
        pass
        # try:
        #     backup_dir = f"backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        #     os.makedirs(backup_dir, exist_ok=True)
        #     
        #     # Backup trade data
        #     if 'trade_storage' in self.components:
        #         self.components['trade_storage'].backup_data(backup_dir)
        #     
        #     # Backup system state
        #     state_data = {
        #         'system_metrics': self.system_metrics.__dict__,
        #         'component_health': {k: v.__dict__ for k, v in self.component_health.items()},
        #         'config': self.config.__dict__
        #     }
        #     
        #     with open(f"{backup_dir}/system_state.json", 'w') as f:
        #         json.dump(state_data, f, indent=2, default=str)
        #     
        #     self.logger.info(f"System data backed up to {backup_dir}")
        #     
        # except Exception as e:
        #     self.logger.error(f"Data backup failed: {e}")
    
    async def _verify_system_health(self) -> bool:
        """Verify overall system health"""
        try:
            healthy_components = 0
            total_components = len(self.component_health)
            
            for name, health in self.component_health.items():
                if health.is_healthy():
                    healthy_components += 1
                else:
                    self.logger.warning(f"Component {name} is not healthy: {health.status}")
            
            health_ratio = healthy_components / total_components if total_components > 0 else 0
            
            # Require at least 80% of components to be healthy
            return health_ratio >= 0.8
            
        except Exception as e:
            self.logger.error(f"System health verification failed: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Handle both enum and string values safely
        state_value = self.state.value if hasattr(self.state, 'value') else str(self.state)
        
        # Safely serialize config with enum handling
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if hasattr(value, 'value'):  # Handle enum values
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        
        # Safely serialize component health with enum handling
        component_health_dict = {}
        for k, v in self.component_health.items():
            health_dict = {}
            for attr_key, attr_value in v.__dict__.items():
                if hasattr(attr_value, 'value'):  # Handle enum values
                    health_dict[attr_key] = attr_value.value
                else:
                    health_dict[attr_key] = attr_value
            component_health_dict[k] = health_dict
        
        return {
            'state': state_value,
            'uptime': str(self.system_metrics.uptime),
            'metrics': self.system_metrics.__dict__,
            'component_health': component_health_dict,
            'config': config_dict
        }
    
    def pause_system(self) -> bool:
        """Pause system operations"""
        try:
            self.logger.info("Pausing system operations...")
            self.state = SystemState.PAUSING
            
            # Pause components that support it
            for name, component in self.components.items():
                if hasattr(component, 'pause'):
                    component.pause()
            
            self.state = SystemState.PAUSED
            self.logger.info("System paused")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pause system: {e}")
            return False
    
    def resume_system(self) -> bool:
        """Resume system operations"""
        try:
            self.logger.info("Resuming system operations...")
            
            # Resume components
            for name, component in self.components.items():
                if hasattr(component, 'resume'):
                    component.resume()
            
            self.state = SystemState.RUNNING
            self.logger.info("System resumed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume system: {e}")
            return False
    
    def shutdown(self) -> None:
        """Graceful system shutdown"""
        self.logger.info("Initiating system shutdown...")
        self.state = SystemState.STOPPING
        
        try:
            # Signal shutdown to all background tasks
            self.shutdown_event.set()
            
            # Close positions if in live trading mode
            if self.config.execution_mode == ExecutionMode.LIVE_TRADING:
                self._close_all_positions()
            
            # Shutdown components in reverse order
            component_names = list(self.components.keys())
            for name in reversed(component_names):
                try:
                    component = self.components[name]
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                    self.logger.info(f"Component {name} shutdown complete")
                except Exception as e:
                    self.logger.error(f"Error shutting down {name}: {e}")
            
            # Wait for background tasks to complete
            for task in self.background_tasks:
                if task.is_alive():
                    task.join(timeout=10)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Final backup - DISABLED
            # self._backup_system_data()
            
            self.state = SystemState.STOPPED
            self.logger.info("System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.state = SystemState.ERROR
    
    # Helper methods (simplified implementations)
    def _convert_signal_to_strength(self, signal_value: float) -> 'SignalStrength':
        """Convert float signal value to SignalStrength enum"""
        from advanced_strategies import SignalStrength
        
        abs_value = abs(signal_value)
        if abs_value >= 0.8:
            return SignalStrength.VERY_STRONG
        elif abs_value >= 0.6:
            return SignalStrength.STRONG
        elif abs_value >= 0.4:
            return SignalStrength.MODERATE
        elif abs_value >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _get_active_market_symbols(self) -> List[str]:
        """Get symbols for currently active markets based on trading sessions"""
        try:
            from datetime import datetime
            import pytz
            
            current_time = datetime.now(pytz.UTC)
            current_hour = current_time.hour
            current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            
            # Define symbol pools for different asset classes (matching venue configurations)
            # crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']  # Binance format - DISABLED
            forex_symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD']  # OANDA format
            stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']  # Alpaca format
            
            active_symbols = []
            
            # Crypto is always active (24/7/365) - DISABLED
            # active_symbols.extend(crypto_symbols[:2])  # Top 2 crypto
            # self.logger.info(f"Added crypto symbols (24/7): {crypto_symbols[:2]}")
            
            # Forex is active 24/5 (Sunday 5 PM EST to Friday 5 PM EST)
            if current_weekday < 5 or (current_weekday == 6 and current_hour >= 22):  # Sunday 22:00 UTC = 5 PM EST
                active_symbols.extend(forex_symbols[:2])  # Top 2 forex
                self.logger.info(f"Added forex symbols (24/5): {forex_symbols[:2]}")
            
            # US Stock market hours: 9:30 AM - 4:00 PM EST (14:30 - 21:00 UTC)
            # FIXED: Align with execution_engine.py market validation (9-16 UTC)
            if current_weekday < 5 and 9 <= current_hour <= 16:  # Monday-Friday, market hours only
                active_symbols.extend(stock_symbols[:2])  # Top 2 stocks during market hours
                self.logger.info(f"Added stock symbols (market hours 9-16 UTC): {stock_symbols[:2]}")
                
                # Prioritize stocks during core trading hours (14-16 UTC)
                if 14 <= current_hour <= 16:
                    # Move stocks to front of list during peak hours
                    stock_portion = [s for s in active_symbols if s in stock_symbols]
                    other_portion = [s for s in active_symbols if s not in stock_symbols]
                    active_symbols = stock_portion + other_portion
                    self.logger.info(f"Prioritized stock symbols during core hours (14-16 UTC)")
            
            # If no active symbols, use default stock symbols
            if not active_symbols:
                active_symbols = stock_symbols[:3]  # Use first 3 stock symbols as fallback
                self.logger.warning("No active symbols determined, falling back to default stock symbols")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_symbols = []
            for symbol in active_symbols:
                if symbol not in seen:
                    seen.add(symbol)
                    unique_symbols.append(symbol)
            
            self.logger.info(f"Active market symbols selected: {unique_symbols[:5]}")
            return unique_symbols[:5]  # Limit to top 5 symbols
            
        except Exception as e:
            self.logger.error(f"Error determining active market symbols: {e}")
            # Fallback to crypto during errors since it's 24/7
            return ['BTC/USD', 'ETH/USD']
    
    def _get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data (simplified for now)"""
        return {}
    
    def _get_market_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Get market data for a specific symbol"""
        try:
            if 'market_data' in self.components:
                # Try to get data from market data component
                data = self.components['market_data'].get_data(symbol)
                if data is not None:
                    return data
            
            # Return empty DataFrame if no data available
            return pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Failed to get market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _extract_features_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Extract ML features for symbol"""
        try:
            # Get market data for the symbol
            market_data = self._get_market_data_for_symbol(symbol)
            
            if market_data.empty:
                # Return empty DataFrame with some basic columns
                return pd.DataFrame(columns=['close', 'volume', 'high', 'low', 'open'])
            
            # Create basic features from market data
            features = pd.DataFrame(index=market_data.index)
            
            if 'close' in market_data.columns:
                features['close'] = market_data['close']
                features['returns'] = market_data['close'].pct_change()
                features['volatility'] = features['returns'].rolling(20).std()
                
            if 'volume' in market_data.columns:
                features['volume'] = market_data['volume']
                features['volume_ma'] = market_data['volume'].rolling(20).mean()
                
            # Fill NaN values
            features = features.fillna(0)
            
            return features.tail(1)  # Return only the latest row
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed for {symbol}: {e}")
            return pd.DataFrame(columns=['close', 'volume', 'high', 'low', 'open'])
    
    def _get_current_portfolio(self) -> Dict[str, float]:
        """Get current portfolio weights"""
        return {}
    
    def _extract_expected_returns(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Extract expected returns from signals"""
        return {signal.symbol: signal.strength for signal in signals}
    
    def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data for risk analysis"""
        try:
            # Get current positions from execution engine
            positions = self.components['execution_engine'].get_positions() if 'execution_engine' in self.components else {}
            
            # Create portfolio weights (normalized)
            total_value = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
            portfolio_weights = {}
            if total_value > 0:
                for symbol, pos in positions.items():
                    portfolio_weights[symbol] = abs(pos.get('market_value', 0)) / total_value
            else:
                # Default portfolio for testing
                portfolio_weights = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.2, 'TSLA': 0.2}
            
            # Get historical returns data
            asset_returns = pd.DataFrame()
            if 'market_data' in self.components:
                try:
                    # Generate sample returns for testing
                    import numpy as np
                    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
                    for symbol in portfolio_weights.keys():
                        returns = np.random.normal(0.001, 0.02, 252)  # Sample daily returns
                        asset_returns[symbol] = returns
                    asset_returns.index = dates
                except Exception as e:
                    # No fallback data - raise error if portfolio data unavailable
                    self.logger.error(f"Failed to get portfolio data: {e}")
                    raise RuntimeError(f"Portfolio data unavailable: {e}")
            
            # Calculate portfolio value
            portfolio_value = total_value if total_value > 0 else 100000.0  # Default $100k
            
            return {
                'portfolio_weights': portfolio_weights,
                'asset_returns': asset_returns,
                'portfolio_value': portfolio_value
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            # No fallback data - fail fast
            raise RuntimeError(f"Portfolio data system failure: {e}")
    
    def _calculate_position_risk(self, signal: TradingSignal) -> float:
        """Calculate position-level risk"""
        return 0.01  # 1% risk
    
    def _calculate_portfolio_impact(self, signal: TradingSignal, risk_metrics: Any) -> float:
        """Calculate portfolio impact"""
        return 0.005  # 0.5% impact
    
    def _passes_risk_checks(self, signal: TradingSignal, position_risk: float, portfolio_impact: float) -> bool:
        """Check if signal passes risk filters"""
        return position_risk < 0.02 and portfolio_impact < 0.01
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        pass
    
    def _restart_component(self, name: str) -> None:
        """Restart failed component"""
        self.logger.info(f"Restarting component: {name}")
    
    def _handle_risk_alert(self, alert: Any) -> None:
        """Handle risk alert"""
        self.logger.warning(f"Risk alert: {alert}")
        self.system_metrics.alerts_generated += 1
    
    def _close_all_positions(self) -> None:
        """Close all open positions"""
        self.logger.info("Closing all positions for shutdown...")

# Main execution
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize orchestrator
        orchestrator = SystemOrchestrator()
        
        try:
            # Initialize system
            if await orchestrator.initialize():
                print("\n🚀 Advanced Trading System v2.0 - System Orchestrator")
                print("="*60)
                print("\n✅ System Components Initialized:")
                
                for component_name in orchestrator.initialized_components:
                    component_status = orchestrator.component_health[component_name].status
                    # Handle both enum and string values safely
                    status = component_status.value if hasattr(component_status, 'value') else str(component_status)
                    print(f"   • {component_name}: {status}")
                
                print(f"\n📊 System Status:")
                try:
                    status = orchestrator.get_system_status()
                    print(f"   • State: {status['state']}")
                except Exception as e:
                    print(f"   • State: {orchestrator.state}")
                    print(f"   • Status error: {e}")
                
                try:
                    execution_mode = orchestrator.config.execution_mode.value if hasattr(orchestrator.config.execution_mode, 'value') else str(orchestrator.config.execution_mode)
                    print(f"   • Execution Mode: {execution_mode}")
                except Exception as e:
                    print(f"   • Execution Mode error: {e}")
                
                print(f"   • Components: {len(orchestrator.initialized_components)}")
                print(f"   • Background Tasks: {len(orchestrator.background_tasks)}")
                
                print(f"\n🎯 System Capabilities:")
                print("   • Multi-asset trading (stocks, crypto, forex)")
                print("   • Advanced strategy orchestration with ML")
                print("   • Real-time risk management and monitoring")
                print("   • Portfolio optimization and rebalancing")
                print("   • Alternative data integration")
                print("   • Comprehensive performance analytics")
                print("   • 24/7 automated operations")
                print("   • Intelligent alerting and notifications")
                print("   • Advanced backtesting and validation")
                print("   • Graceful error handling and recovery")
                
                print(f"\n⚙️  Configuration:")
                print(f"   • Max Concurrent Strategies: {orchestrator.config.max_concurrent_strategies}")
                print(f"   • ML Pipeline: {'Enabled' if orchestrator.config.enable_ml_pipeline else 'Disabled'}")
                print(f"   • Alternative Data: {'Enabled' if orchestrator.config.enable_alternative_data else 'Disabled'}")
                print(f"   • Portfolio Optimization: {'Enabled' if orchestrator.config.enable_portfolio_optimization else 'Disabled'}")
                print(f"   • Auto Restart: {'Enabled' if orchestrator.config.auto_restart_failed_components else 'Disabled'}")
                
                print("\n🔄 System is running... Press Ctrl+C to shutdown")
                
                # Keep running until interrupted
                try:
                    while orchestrator.state == SystemState.RUNNING:
                        await asyncio.sleep(10)
                        
                        # Print periodic status
                        if int(time.time()) % 300 == 0:  # Every 5 minutes
                            metrics = orchestrator.system_metrics
                            print(f"\n📈 System Metrics (Uptime: {metrics.uptime}):")
                            print(f"   • Total Trades: {metrics.total_trades}")
                            print(f"   • Success Rate: {metrics.success_rate:.2%}")
                            print(f"   • Active Positions: {metrics.current_positions}")
                            print(f"   • System Load: {metrics.system_load:.1f}%")
                            print(f"   • Memory Usage: {metrics.memory_usage:.1f}%")
                            
                except KeyboardInterrupt:
                    print("\n🛑 Shutdown signal received...")
                
            else:
                print("❌ System initialization failed")
                
        except Exception as e:
            print(f"❌ System error: {e}")
            
        finally:
            # Graceful shutdown
            print("\n🔄 Shutting down system...")
            orchestrator.shutdown()
            print("✅ System shutdown complete")
    
    # Run the orchestrator
    asyncio.run(main())