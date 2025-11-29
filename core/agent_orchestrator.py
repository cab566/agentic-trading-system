#!/usr/bin/env python3
"""
Agent Orchestrator for Multi-Asset Trading System

Coordinates and manages all trading agents across:
- Traditional markets (stocks, bonds, options)
- Cryptocurrency markets (24/7)
- Forex markets (24/5)
- Multi-strategy execution
- Risk management coordination
- Performance monitoring
- Resource allocation
- Communication and collaboration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .config_manager import ConfigManager
from .market_data_aggregator import MarketDataAggregator
from .execution_engine import ExecutionEngine
from .risk_manager_24_7 import RiskManager24_7
from .backtesting_engine import BacktestingEngine
from .trading_orchestrator_24_7 import TradingOrchestrator24_7
# from tools.crypto_analysis_tool import CryptoAnalysisTool  # DISABLED - Crypto functionality commented out
from tools.forex_analysis_tool import ForexAnalysisTool
from tools.multi_asset_portfolio_tool import MultiAssetPortfolioTool
from tools.volume_spike_scanner import VolumeSpikeDiscoveryTool
from tools.news_driven_discovery import NewsDrivenDiscoveryTool
from tools.technical_breakout_scanner import TechnicalBreakoutDiscoveryTool
from tools.earnings_calendar_monitor import EarningsCalendarDiscoveryTool
from tools.sector_rotation_detector import SectorRotationDiscoveryTool

# New discovery tools
try:
    from tools.real_time_opportunity_scanner import RealTimeOpportunityScannerTool
    from tools.social_sentiment_analyzer import SocialSentimentAnalyzerTool
    from tools.cross_asset_arbitrage_detector import CrossAssetArbitrageDetectorTool
    from tools.market_regime_detector import MarketRegimeDetectorTool
    from tools.economic_calendar_monitor import EconomicCalendarMonitorTool
    from tools.options_flow_analyzer import OptionsFlowAnalyzerTool
    from tools.earnings_surprise_predictor import EarningsSurprisePredictorTool
    from tools.rd_agent_integration_tool import RDAgentIntegrationTool
    NEW_DISCOVERY_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some new discovery tools could not be imported: {e}")
    NEW_DISCOVERY_TOOLS_AVAILABLE = False

from utils.cache_manager import CacheManager
from utils.notifications import NotificationManager


class AgentType(Enum):
    """Types of trading agents."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    MONITORING = "monitoring"
    COORDINATION = "coordination"


class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class MarketSession(Enum):
    """Market session enumeration."""
    US_PRE_MARKET = "us_pre_market"
    US_REGULAR = "us_regular"
    US_AFTER_HOURS = "us_after_hours"
    EUROPE = "europe"
    ASIA = "asia"
    CRYPTO_24_7 = "crypto_24_7"
    FOREX_SYDNEY = "forex_sydney"
    FOREX_TOKYO = "forex_tokyo"
    FOREX_LONDON = "forex_london"
    FOREX_NEW_YORK = "forex_new_york"


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    agent_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    last_activity: Optional[datetime] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    uptime: float = 0.0
    
    def update_success_rate(self):
        """Update success rate based on completed and failed tasks."""
        total_tasks = self.tasks_completed + self.tasks_failed
        self.success_rate = self.tasks_completed / total_tasks if total_tasks > 0 else 0.0


@dataclass
class AgentInfo:
    """Agent information and state."""
    agent_id: str
    agent_type: AgentType
    agent: Agent
    status: AgentStatus = AgentStatus.IDLE
    market_sessions: List[MarketSession] = field(default_factory=list)
    supported_assets: List[str] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    priority: int = 1  # 1 = highest, 5 = lowest
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metrics: AgentMetrics = field(default_factory=lambda: AgentMetrics(""))
    
    def __post_init__(self):
        if not self.metrics.agent_id:
            self.metrics.agent_id = self.agent_id
    
    def can_accept_task(self) -> bool:
        """Check if agent can accept new tasks."""
        return (
            self.status in [AgentStatus.IDLE, AgentStatus.ACTIVE] and
            len(self.current_tasks) < self.max_concurrent_tasks
        )
    
    def is_available_for_session(self, session: MarketSession) -> bool:
        """Check if agent is available for market session."""
        return session in self.market_sessions or MarketSession.CRYPTO_24_7 in self.market_sessions


@dataclass
class TaskRequest:
    """Task request for agent execution."""
    task_id: str
    task_type: str
    agent_type: AgentType
    priority: int
    data: Dict[str, Any]
    deadline: Optional[datetime] = None
    required_sessions: List[MarketSession] = field(default_factory=list)
    required_assets: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    assigned_agent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class CollaborationRequest:
    """Request for agent collaboration."""
    collaboration_id: str
    initiating_agent: str
    target_agents: List[str]
    collaboration_type: str  # "data_sharing", "joint_analysis", "consensus_building"
    data: Dict[str, Any]
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    created_at: datetime = field(default_factory=datetime.now)
    responses: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False


class AgentOrchestrator:
    """
    Comprehensive agent orchestrator for multi-asset trading system.
    
    Features:
    - Agent lifecycle management
    - Task scheduling and distribution
    - Resource allocation and load balancing
    - Inter-agent communication and collaboration
    - Performance monitoring and optimization
    - Market session coordination
    - Risk management integration
    - Real-time decision making
    - Fault tolerance and recovery
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        market_data_aggregator: MarketDataAggregator,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager24_7,
        backtesting_engine: BacktestingEngine,
        trading_orchestrator: TradingOrchestrator24_7
    ):
        self.config_manager = config_manager
        self.market_data_aggregator = market_data_aggregator
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.backtesting_engine = backtesting_engine
        self.trading_orchestrator = trading_orchestrator
        
        self.cache_manager = CacheManager(config_manager)
        self.notification_manager = NotificationManager(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Agent management
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_crews: Dict[str, Crew] = {}
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.completed_tasks: Dict[str, TaskRequest] = {}
        self.collaboration_requests: Dict[str, CollaborationRequest] = {}
        
        # Scheduling and coordination
        self.scheduler_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.performance_monitor_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Market session tracking
        self.current_sessions: Set[MarketSession] = set()
        self.session_agents: Dict[MarketSession, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.system_metrics = {
            'total_tasks_processed': 0,
            'average_task_time': 0.0,
            'system_uptime': datetime.now(),
            'error_rate': 0.0,
            'throughput': 0.0
        }
        
        # Tools for agents
        # self.crypto_tool = CryptoAnalysisTool(config_manager)  # DISABLED - Crypto functionality commented out
        self.forex_tool = ForexAnalysisTool(config_manager)
        self.portfolio_tool = MultiAssetPortfolioTool(config_manager)
        
        # Dynamic discovery tools
        self.volume_spike_tool = VolumeSpikeDiscoveryTool()
        self.news_discovery_tool = NewsDrivenDiscoveryTool()
        self.technical_breakout_tool = TechnicalBreakoutDiscoveryTool()
        self.earnings_calendar_tool = EarningsCalendarDiscoveryTool()
        self.sector_rotation_tool = SectorRotationDiscoveryTool()
        
        # Initialize new discovery tools if available
        if NEW_DISCOVERY_TOOLS_AVAILABLE:
            try:
                self.real_time_opportunity_tool = RealTimeOpportunityScannerTool()
                self.social_sentiment_tool = SocialSentimentAnalyzerTool()
                self.cross_asset_arbitrage_tool = CrossAssetArbitrageDetectorTool()
                self.market_regime_tool = MarketRegimeDetectorTool()
                self.economic_calendar_monitor_tool = EconomicCalendarMonitorTool()
                self.options_flow_tool = OptionsFlowAnalyzerTool()
                self.earnings_surprise_tool = EarningsSurprisePredictorTool()
                self.rd_agent_tool = RDAgentIntegrationTool()
                self.logger.info("Successfully initialized new discovery tools")
            except Exception as e:
                self.logger.warning(f"Failed to initialize some new discovery tools: {e}")
                # Set tools to None if initialization fails
                self.real_time_opportunity_tool = None
                self.social_sentiment_tool = None
                self.cross_asset_arbitrage_tool = None
                self.market_regime_tool = None
                self.economic_calendar_monitor_tool = None
                self.options_flow_tool = None
                self.earnings_surprise_tool = None
                self.rd_agent_tool = None
        else:
            # Set all new tools to None if not available
            self.real_time_opportunity_tool = None
            self.social_sentiment_tool = None
            self.cross_asset_arbitrage_tool = None
            self.market_regime_tool = None
            self.economic_calendar_monitor_tool = None
            self.options_flow_tool = None
            self.earnings_surprise_tool = None
            self.rd_agent_tool = None
    
    async def initialize(self):
        """Initialize the agent orchestrator."""
        try:
            self.logger.info("Initializing Agent Orchestrator")
            
            # Create and register agents
            await self._create_agents()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Initialize market session tracking
            await self._initialize_market_sessions()
            
            self.logger.info(f"Agent Orchestrator initialized with {len(self.agents)} agents")
        
        except Exception as e:
            self.logger.error(f"Error initializing Agent Orchestrator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the agent orchestrator."""
        try:
            self.logger.info("Shutting down Agent Orchestrator")
            
            # Stop background tasks
            self.scheduler_running = False
            
            if self.scheduler_task:
                self.scheduler_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.performance_monitor_task:
                self.performance_monitor_task.cancel()
            
            # Wait for active tasks to complete (with timeout)
            await self._wait_for_active_tasks(timeout=30)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Agent Orchestrator shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error shutting down Agent Orchestrator: {e}")
    
    async def _create_agents(self):
        """Create and register all trading agents."""
        # Research Agents
        await self._create_research_agents()
        
        # Analysis Agents
        await self._create_analysis_agents()
        
        # Strategy Agents
        await self._create_strategy_agents()
        
        # Execution Agents
        await self._create_execution_agents()
        
        # Risk Management Agents
        await self._create_risk_agents()
        
        # Portfolio Management Agents
        await self._create_portfolio_agents()
        
        # Monitoring Agents
        await self._create_monitoring_agents()
        
        # Coordination Agents
        await self._create_coordination_agents()
    
    async def _create_research_agents(self):
        """Create research agents."""
        # Prepare tool list for market research agent
        research_tools = [
            # self.crypto_tool,  # DISABLED - Crypto functionality commented out
            self.forex_tool, 
            self.volume_spike_tool,
            self.news_discovery_tool,
            self.technical_breakout_tool,
            self.earnings_calendar_tool,
            self.sector_rotation_tool
        ]
        
        # Add new discovery tools if available
        if NEW_DISCOVERY_TOOLS_AVAILABLE:
            new_tools = [
                self.real_time_opportunity_tool,
                self.social_sentiment_tool,
                self.cross_asset_arbitrage_tool,
                self.market_regime_tool,
                self.economic_calendar_monitor_tool,
                self.options_flow_tool,
                self.earnings_surprise_tool,
                self.rd_agent_tool
            ]
            # Only add tools that were successfully initialized
            research_tools.extend([tool for tool in new_tools if tool is not None])
        
        # Market Research Agent with enhanced discovery capabilities
        market_research_agent = Agent(
            role="Market Research Analyst",
            goal="Conduct comprehensive market research and identify trading opportunities using dynamic discovery tools including real-time scanning, sentiment analysis, cross-asset arbitrage, market regime detection, and advanced analytics",
            backstory="Expert in financial markets with deep knowledge of fundamental and technical analysis, specializing in discovering emerging opportunities through volume analysis, news sentiment, technical breakouts, earnings events, sector rotation patterns, real-time opportunity scanning, social sentiment analysis, cross-asset arbitrage detection, market regime analysis, economic calendar monitoring, options flow analysis, earnings surprise prediction, and AI-powered research integration",
            tools=research_tools,
            verbose=True,
            allow_delegation=True
        )
        
        await self._register_agent(
            agent_id="market_research",
            agent_type=AgentType.RESEARCH,
            agent=market_research_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.EUROPE, MarketSession.ASIA],
            supported_assets=["stocks", "bonds", "etfs"]
        )
        
        # Crypto Research Agent - DISABLED
        # crypto_research_agent = Agent(
        #     role="Cryptocurrency Research Analyst",
        #     goal="Research cryptocurrency markets, DeFi protocols, and blockchain trends",
        #     backstory="Specialist in cryptocurrency markets with expertise in on-chain analysis and DeFi",
        #     tools=[self.crypto_tool],
        #     verbose=True,
        #     allow_delegation=True
        # )
        
        # await self._register_agent(
        #     agent_id="crypto_research",
        #     agent_type=AgentType.RESEARCH,
        #     agent=crypto_research_agent,
        #     market_sessions=[MarketSession.CRYPTO_24_7],
        #     supported_assets=["bitcoin", "ethereum", "altcoins", "defi"]
        # )
        
        # Forex Research Agent
        forex_research_agent = Agent(
            role="Foreign Exchange Research Analyst",
            goal="Analyze currency markets, central bank policies, and macroeconomic trends",
            backstory="Expert in forex markets with deep understanding of macroeconomics and central banking",
            tools=[self.forex_tool],
            verbose=True,
            allow_delegation=True
        )
        
        await self._register_agent(
            agent_id="forex_research",
            agent_type=AgentType.RESEARCH,
            agent=forex_research_agent,
            market_sessions=[MarketSession.FOREX_LONDON, MarketSession.FOREX_NEW_YORK, MarketSession.FOREX_TOKYO],
            supported_assets=["major_pairs", "minor_pairs", "exotic_pairs"]
        )
    
    async def _create_analysis_agents(self):
        """Create analysis agents."""
        # Technical Analysis Agent
        technical_agent = Agent(
            role="Technical Analysis Expert",
            goal="Perform advanced technical analysis across all asset classes",
            backstory="Master of technical analysis with expertise in chart patterns, indicators, and market structure",
            tools=[self.forex_tool, self.portfolio_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="technical_analysis",
            agent_type=AgentType.ANALYSIS,
            agent=technical_agent,
            market_sessions=[MarketSession.CRYPTO_24_7, MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],
            supported_assets=["all"]
        )
        
        # Fundamental Analysis Agent
        fundamental_agent = Agent(
            role="Fundamental Analysis Expert",
            goal="Conduct deep fundamental analysis of companies, economies, and market conditions",
            backstory="Expert in financial statement analysis, valuation models, and economic indicators",
            tools=[self.portfolio_tool],
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="fundamental_analysis",
            agent_type=AgentType.ANALYSIS,
            agent=fundamental_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.EUROPE],
            supported_assets=["stocks", "bonds", "commodities"]
        )
        
        # Sentiment Analysis Agent
        sentiment_agent = Agent(
            role="Market Sentiment Analyst",
            goal="Analyze market sentiment from news, social media, and market data",
            backstory="Specialist in sentiment analysis with expertise in NLP and social media monitoring",
            tools=[self.forex_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="sentiment_analysis",
            agent_type=AgentType.ANALYSIS,
            agent=sentiment_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],  # Removed CRYPTO_24_7
            supported_assets=["all"]
        )
    
    async def _create_strategy_agents(self):
        """Create strategy agents."""
        # Momentum Strategy Agent
        momentum_agent = Agent(
            role="Momentum Strategy Specialist",
            goal="Identify and execute momentum-based trading strategies",
            backstory="Expert in momentum trading with deep understanding of trend analysis and breakout patterns",
            tools=[self.forex_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="momentum_strategy",
            agent_type=AgentType.STRATEGY,
            agent=momentum_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],  # Removed CRYPTO_24_7
            supported_assets=["stocks", "forex"]
        )
        
        # Mean Reversion Strategy Agent
        mean_reversion_agent = Agent(
            role="Mean Reversion Strategy Specialist",
            goal="Identify and execute mean reversion trading strategies",
            backstory="Specialist in mean reversion strategies with expertise in statistical analysis and market cycles",
            tools=[self.forex_tool, self.portfolio_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="mean_reversion_strategy",
            agent_type=AgentType.STRATEGY,
            agent=mean_reversion_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],  # Removed CRYPTO_24_7
            supported_assets=["stocks", "forex"]
        )
        
        # Arbitrage Strategy Agent
        arbitrage_agent = Agent(
            role="Arbitrage Strategy Specialist",
            goal="Identify and execute arbitrage opportunities across markets",
            backstory="Expert in arbitrage strategies with deep knowledge of cross-market inefficiencies",
            tools=[self.forex_tool, self.portfolio_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="arbitrage_strategy",
            agent_type=AgentType.STRATEGY,
            agent=arbitrage_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],  # Removed CRYPTO_24_7
            supported_assets=["stocks", "forex"]
        )
        
        # Pairs Trading Strategy Agent
        pairs_trading_agent = Agent(
            role="Pairs Trading Strategy Specialist",
            goal="Identify and execute pairs trading strategies",
            backstory="Specialist in pairs trading with expertise in statistical arbitrage and correlation analysis",
            tools=[self.forex_tool, self.portfolio_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="pairs_trading_strategy",
            agent_type=AgentType.STRATEGY,
            agent=pairs_trading_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],  # Removed CRYPTO_24_7
            supported_assets=["stocks", "forex"]
        )
    
    async def _create_execution_agents(self):
        """Create execution agents."""
        # Smart Order Routing Agent
        execution_agent = Agent(
            role="Smart Execution Specialist",
            goal="Execute trades with optimal timing, routing, and minimal market impact",
            backstory="Expert in trade execution with deep knowledge of market microstructure",
            tools=[],
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="smart_execution",
            agent_type=AgentType.EXECUTION,
            agent=execution_agent,
            market_sessions=[MarketSession.CRYPTO_24_7, MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],
            supported_assets=["all"]
        )
        
        # Crypto Execution Agent - DISABLED (crypto functionality commented out)
        # crypto_execution_agent = Agent(
        #     role="Cryptocurrency Execution Specialist",
        #     goal="Execute cryptocurrency trades across multiple exchanges with optimal routing",
        #     backstory="Specialist in crypto execution with expertise in DEX and CEX trading",
        #     tools=[self.crypto_tool],
        #     verbose=True,
        #     allow_delegation=False
        # )
        
        # await self._register_agent(
        #     agent_id="crypto_execution",
        #     agent_type=AgentType.EXECUTION,
        #     agent=crypto_execution_agent,
        #     market_sessions=[MarketSession.CRYPTO_24_7],
        #     supported_assets=["crypto"]
        # )
    
    async def _create_risk_agents(self):
        """Create risk management agents."""
        # Portfolio Risk Agent
        portfolio_risk_agent = Agent(
            role="Portfolio Risk Manager",
            goal="Monitor and manage portfolio-wide risk exposure",
            backstory="Expert in portfolio risk management with deep understanding of risk metrics and hedging strategies",
            tools=[self.portfolio_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="portfolio_risk",
            agent_type=AgentType.RISK,
            agent=portfolio_risk_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],  # Removed CRYPTO_24_7
            supported_assets=["all"]
        )
        
        # Market Risk Agent
        market_risk_agent = Agent(
            role="Market Risk Analyst",
            goal="Analyze and monitor market-wide risk factors",
            backstory="Specialist in market risk analysis with expertise in volatility modeling and stress testing",
            tools=[self.forex_tool, self.portfolio_tool],  # Removed crypto_tool - crypto functionality disabled
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="market_risk",
            agent_type=AgentType.RISK,
            agent=market_risk_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.FOREX_LONDON],  # Removed CRYPTO_24_7
            supported_assets=["all"]
        )
    
    async def _create_portfolio_agents(self):
        """Create portfolio management agents."""
        # Portfolio Optimization Agent
        portfolio_agent = Agent(
            role="Portfolio Optimization Specialist",
            goal="Optimize portfolio allocation and rebalancing across all asset classes",
            backstory="Expert in portfolio theory with focus on multi-asset optimization",
            tools=[self.portfolio_tool],
            verbose=True,
            allow_delegation=True
        )
        
        await self._register_agent(
            agent_id="portfolio_optimization",
            agent_type=AgentType.PORTFOLIO,
            agent=portfolio_agent,
            market_sessions=[MarketSession.US_REGULAR, MarketSession.CRYPTO_24_7],
            supported_assets=["all"]
        )
    
    async def _create_monitoring_agents(self):
        """Create monitoring agents."""
        # Performance Monitor Agent
        performance_agent = Agent(
            role="Performance Monitor",
            goal="Monitor and analyze trading performance across all strategies and agents",
            backstory="Expert in performance analysis with focus on risk-adjusted returns",
            tools=[self.portfolio_tool],
            verbose=True,
            allow_delegation=False
        )
        
        await self._register_agent(
            agent_id="performance_monitor",
            agent_type=AgentType.MONITORING,
            agent=performance_agent,
            market_sessions=[MarketSession.CRYPTO_24_7],
            supported_assets=["all"]
        )
    
    async def _create_coordination_agents(self):
        """Create coordination agents."""
        # Master Coordinator Agent
        coordinator_agent = Agent(
            role="Master Coordinator",
            goal="Coordinate activities across all agents and ensure optimal system performance",
            backstory="Master coordinator with expertise in multi-agent systems and workflow optimization",
            tools=[self.portfolio_tool],
            verbose=True,
            allow_delegation=True
        )
        
        await self._register_agent(
            agent_id="master_coordinator",
            agent_type=AgentType.COORDINATION,
            agent=coordinator_agent,
            market_sessions=[MarketSession.CRYPTO_24_7],
            supported_assets=["all"],
            priority=1  # Highest priority
        )
    
    async def _register_agent(
        self,
        agent_id: str,
        agent_type: AgentType,
        agent: Agent,
        market_sessions: List[MarketSession],
        supported_assets: List[str],
        priority: int = 3,
        max_concurrent_tasks: int = 3
    ):
        """Register an agent with the orchestrator."""
        agent_info = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            agent=agent,
            market_sessions=market_sessions,
            supported_assets=supported_assets,
            priority=priority,
            max_concurrent_tasks=max_concurrent_tasks
        )
        
        self.agents[agent_id] = agent_info
        
        # Add to session mapping
        for session in market_sessions:
            self.session_agents[session].append(agent_id)
        
        self.logger.info(f"Registered agent: {agent_id} ({agent_type.value})")
    
    async def _start_background_tasks(self):
        """Start background monitoring and scheduling tasks."""
        self.scheduler_running = True
        
        # Task scheduler
        self.scheduler_task = asyncio.create_task(self._task_scheduler())
        
        # Agent heartbeat monitor
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        # Performance monitor
        self.performance_monitor_task = asyncio.create_task(self._performance_monitor())
    
    async def _initialize_market_sessions(self):
        """Initialize market session tracking."""
        # This would implement market session detection logic
        # For now, assume crypto is always active
        self.current_sessions.add(MarketSession.CRYPTO_24_7)
    
    async def submit_task(
        self,
        task_type: str,
        agent_type: AgentType,
        data: Dict[str, Any],
        priority: int = 3,
        deadline: Optional[datetime] = None,
        required_sessions: Optional[List[MarketSession]] = None,
        required_assets: Optional[List[str]] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a task for execution."""
        task_id = str(uuid.uuid4())
        
        task_request = TaskRequest(
            task_id=task_id,
            task_type=task_type,
            agent_type=agent_type,
            priority=priority,
            data=data,
            deadline=deadline,
            required_sessions=required_sessions or [],
            required_assets=required_assets or [],
            callback=callback
        )
        
        # Add to queue (priority queue would be better)
        self.task_queue.append(task_request)
        
        self.logger.info(f"Task submitted: {task_id} ({task_type})")
        
        return task_id
    
    async def _task_scheduler(self):
        """Main task scheduling loop."""
        while self.scheduler_running:
            try:
                # Process task queue
                await self._process_task_queue()
                
                # Check for completed tasks
                await self._check_completed_tasks()
                
                # Handle collaboration requests
                await self._process_collaboration_requests()
                
                # Update market sessions
                await self._update_market_sessions()
                
                # Sleep briefly
                await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {e}")
                await asyncio.sleep(5)
    
    async def _process_task_queue(self):
        """Process pending tasks in the queue."""
        if not self.task_queue:
            return
        
        # Sort tasks by priority and deadline
        sorted_tasks = sorted(
            self.task_queue,
            key=lambda t: (t.priority, t.deadline or datetime.max)
        )
        
        for task in sorted_tasks:
            # Find suitable agent
            agent_id = await self._find_suitable_agent(task)
            
            if agent_id:
                # Assign task to agent
                await self._assign_task_to_agent(task, agent_id)
                
                # Remove from queue
                self.task_queue.remove(task)
                
                # Add to active tasks
                self.active_tasks[task.task_id] = task
    
    async def _find_suitable_agent(self, task: TaskRequest) -> Optional[str]:
        """Find the most suitable agent for a task."""
        suitable_agents = []
        
        for agent_id, agent_info in self.agents.items():
            # Check agent type
            if agent_info.agent_type != task.agent_type:
                continue
            
            # Check availability
            if not agent_info.can_accept_task():
                continue
            
            # Check market session compatibility
            if task.required_sessions:
                if not any(agent_info.is_available_for_session(session) for session in task.required_sessions):
                    continue
            
            # Check asset compatibility
            if task.required_assets:
                if not ("all" in agent_info.supported_assets or 
                       any(asset in agent_info.supported_assets for asset in task.required_assets)):
                    continue
            
            suitable_agents.append((agent_id, agent_info))
        
        if not suitable_agents:
            return None
        
        # Select best agent based on priority, load, and performance
        best_agent = min(
            suitable_agents,
            key=lambda x: (
                x[1].priority,
                len(x[1].current_tasks),
                -x[1].metrics.success_rate
            )
        )
        
        return best_agent[0]
    
    async def _assign_task_to_agent(self, task: TaskRequest, agent_id: str):
        """Assign a task to a specific agent."""
        agent_info = self.agents[agent_id]
        
        # Update task
        task.assigned_agent = agent_id
        task.started_at = datetime.now()
        
        # Update agent
        agent_info.current_tasks.append(task.task_id)
        agent_info.status = AgentStatus.ACTIVE
        agent_info.last_heartbeat = datetime.now()
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task(task, agent_info))
        
        self.logger.info(f"Task {task.task_id} assigned to agent {agent_id}")
    
    async def _execute_task(self, task: TaskRequest, agent_info: AgentInfo):
        """Execute a task with the assigned agent."""
        try:
            start_time = datetime.now()
            
            # Create CrewAI task
            crew_task = Task(
                description=f"Execute {task.task_type} with data: {task.data}",
                agent=agent_info.agent,
                expected_output="Structured analysis and recommendations"
            )
            
            # Create crew and execute
            crew = Crew(
                agents=[agent_info.agent],
                tasks=[crew_task],
                verbose=True
            )
            
            # Execute in thread pool for CPU-intensive tasks
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                crew.kickoff
            )
            
            # Update task
            task.completed_at = datetime.now()
            task.result = result
            
            # Update agent metrics
            execution_time = (task.completed_at - start_time).total_seconds()
            agent_info.metrics.tasks_completed += 1
            agent_info.metrics.average_execution_time = (
                (agent_info.metrics.average_execution_time * (agent_info.metrics.tasks_completed - 1) + execution_time) /
                agent_info.metrics.tasks_completed
            )
            agent_info.metrics.update_success_rate()
            agent_info.metrics.last_activity = datetime.now()
            
            # Call callback if provided
            if task.callback:
                try:
                    await task.callback(task.result)
                except Exception as e:
                    self.logger.error(f"Error in task callback: {e}")
            
            self.logger.info(f"Task {task.task_id} completed successfully")
        
        except Exception as e:
            # Handle task failure
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # Update agent metrics
            agent_info.metrics.tasks_failed += 1
            agent_info.metrics.error_count += 1
            agent_info.metrics.update_success_rate()
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Clean up
            agent_info.current_tasks.remove(task.task_id)
            
            if not agent_info.current_tasks:
                agent_info.status = AgentStatus.IDLE
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _check_completed_tasks(self):
        """Check for completed tasks and handle cleanup."""
        # Clean up old completed tasks (keep last 1000)
        if len(self.completed_tasks) > 1000:
            oldest_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].completed_at or datetime.min
            )[:100]
            
            for task_id, _ in oldest_tasks:
                del self.completed_tasks[task_id]
    
    async def _process_collaboration_requests(self):
        """Process inter-agent collaboration requests."""
        for collab_id, request in list(self.collaboration_requests.items()):
            if request.completed:
                continue
            
            # Check timeout
            if datetime.now() - request.created_at > request.timeout:
                request.completed = True
                self.logger.warning(f"Collaboration request {collab_id} timed out")
                continue
            
            # Check if all responses received
            if len(request.responses) >= len(request.target_agents):
                request.completed = True
                self.logger.info(f"Collaboration request {collab_id} completed")
    
    async def _update_market_sessions(self):
        """Update current market sessions based on time."""
        # This would implement real market session detection
        # For now, keep crypto always active
        current_time = datetime.now()
        
        # Simple session detection (would be more sophisticated in practice)
        if 9 <= current_time.hour < 16:  # US market hours (simplified)
            self.current_sessions.add(MarketSession.US_REGULAR)
        else:
            self.current_sessions.discard(MarketSession.US_REGULAR)
        
        # Crypto is always active
        self.current_sessions.add(MarketSession.CRYPTO_24_7)
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and health."""
        while self.scheduler_running:
            try:
                current_time = datetime.now()
                
                for agent_id, agent_info in self.agents.items():
                    # Check heartbeat
                    time_since_heartbeat = current_time - agent_info.last_heartbeat
                    
                    if time_since_heartbeat > timedelta(minutes=5):
                        if agent_info.status != AgentStatus.OFFLINE:
                            agent_info.status = AgentStatus.ERROR
                            self.logger.warning(f"Agent {agent_id} heartbeat timeout")
                    
                    # Update uptime
                    uptime = (current_time - agent_info.created_at).total_seconds()
                    agent_info.metrics.uptime = uptime
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor(self):
        """Monitor system and agent performance."""
        while self.scheduler_running:
            try:
                # Update system metrics
                total_tasks = sum(agent.metrics.tasks_completed + agent.metrics.tasks_failed 
                                for agent in self.agents.values())
                
                self.system_metrics['total_tasks_processed'] = total_tasks
                
                # Calculate average task time
                if total_tasks > 0:
                    total_time = sum(agent.metrics.average_execution_time * 
                                   (agent.metrics.tasks_completed + agent.metrics.tasks_failed)
                                   for agent in self.agents.values())
                    self.system_metrics['average_task_time'] = total_time / total_tasks
                
                # Calculate error rate
                total_errors = sum(agent.metrics.tasks_failed for agent in self.agents.values())
                self.system_metrics['error_rate'] = total_errors / total_tasks if total_tasks > 0 else 0
                
                # Log performance summary
                if total_tasks > 0 and total_tasks % 100 == 0:  # Every 100 tasks
                    self.logger.info(f"Performance Summary: {total_tasks} tasks processed, "
                                   f"avg time: {self.system_metrics['average_task_time']:.2f}s, "
                                   f"error rate: {self.system_metrics['error_rate']:.2%}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
            
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(300)
    
    async def _wait_for_active_tasks(self, timeout: int = 30):
        """Wait for active tasks to complete with timeout."""
        start_time = datetime.now()
        
        while self.active_tasks and (datetime.now() - start_time).seconds < timeout:
            await asyncio.sleep(1)
        
        if self.active_tasks:
            self.logger.warning(f"{len(self.active_tasks)} tasks still active after timeout")
    
    # Public API methods
    
    async def request_analysis(
        self,
        analysis_type: str,
        symbols: List[str],
        timeframe: str = "1d",
        lookback_days: int = 30
    ) -> str:
        """Request market analysis from appropriate agents."""
        return await self.submit_task(
            task_type="market_analysis",
            agent_type=AgentType.ANALYSIS,
            data={
                "analysis_type": analysis_type,
                "symbols": symbols,
                "timeframe": timeframe,
                "lookback_days": lookback_days
            },
            priority=2
        )
    
    async def request_strategy_signals(
        self,
        strategy_name: str,
        symbols: List[str],
        parameters: Dict[str, Any]
    ) -> str:
        """Request trading signals from strategy agents."""
        return await self.submit_task(
            task_type="generate_signals",
            agent_type=AgentType.STRATEGY,
            data={
                "strategy_name": strategy_name,
                "symbols": symbols,
                "parameters": parameters
            },
            priority=1  # High priority for trading signals
        )
    
    async def request_risk_assessment(
        self,
        portfolio_data: Dict[str, Any],
        scenario: Optional[str] = None
    ) -> str:
        """Request risk assessment from risk management agents."""
        return await self.submit_task(
            task_type="risk_assessment",
            agent_type=AgentType.RISK,
            data={
                "portfolio_data": portfolio_data,
                "scenario": scenario
            },
            priority=1  # High priority for risk management
        )
    
    async def request_portfolio_optimization(
        self,
        current_positions: Dict[str, float],
        target_allocation: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> str:
        """Request portfolio optimization from portfolio agents."""
        return await self.submit_task(
            task_type="portfolio_optimization",
            agent_type=AgentType.PORTFOLIO,
            data={
                "current_positions": current_positions,
                "target_allocation": target_allocation,
                "constraints": constraints
            },
            priority=2
        )
    
    async def request_volume_spike_discovery(
        self,
        spike_threshold: float = 2.0,
        min_price_change: float = 0.02,
        sectors: Optional[List[str]] = None,
        market_cap_min: Optional[float] = None
    ) -> str:
        """Request volume spike discovery from research agents."""
        return await self.submit_task(
            task_type="volume_spike_discovery",
            agent_type=AgentType.RESEARCH,
            data={
                "spike_threshold": spike_threshold,
                "min_price_change": min_price_change,
                "sectors": sectors,
                "market_cap_min": market_cap_min
            },
            priority=2
        )
    
    async def request_news_driven_discovery(
        self,
        sentiment_threshold: float = 0.6,
        relevance_threshold: float = 0.7,
        max_age_hours: int = 24,
        event_types: Optional[List[str]] = None
    ) -> str:
        """Request news-driven stock discovery from research agents."""
        return await self.submit_task(
            task_type="news_driven_discovery",
            agent_type=AgentType.RESEARCH,
            data={
                "sentiment_threshold": sentiment_threshold,
                "relevance_threshold": relevance_threshold,
                "max_age_hours": max_age_hours,
                "event_types": event_types
            },
            priority=2
        )
    
    async def request_technical_breakout_discovery(
        self,
        pattern_types: Optional[List[str]] = None,
        min_strength: str = "moderate",
        timeframes: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None
    ) -> str:
        """Request technical breakout discovery from research agents."""
        return await self.submit_task(
            task_type="technical_breakout_discovery",
            agent_type=AgentType.RESEARCH,
            data={
                "pattern_types": pattern_types,
                "min_strength": min_strength,
                "timeframes": timeframes,
                "sectors": sectors
            },
            priority=2
        )
    
    async def request_earnings_calendar_discovery(
        self,
        days_ahead: int = 7,
        impact_threshold: str = "moderate",
        sectors: Optional[List[str]] = None,
        market_cap_min: Optional[float] = None
    ) -> str:
        """Request earnings calendar-based discovery."""
        return await self.submit_task(
            task_type="earnings_calendar_discovery",
            agent_type=AgentType.RESEARCH,
            data={
                "days_ahead": days_ahead,
                "impact_threshold": impact_threshold,
                "sectors": sectors,
                "market_cap_min": market_cap_min
            },
            priority=2
        )
    
    async def request_sector_rotation_discovery(
        self,
        lookback_days: int = 30,
        min_strength: str = "moderate",
        rotation_phases: Optional[List[str]] = None
    ) -> str:
        """Request sector rotation-based discovery."""
        return await self.submit_task(
            task_type="sector_rotation_discovery",
            agent_type=AgentType.RESEARCH,
            data={
                "lookback_days": lookback_days,
                "min_strength": min_strength,
                "rotation_phases": rotation_phases
            },
            priority=2
        )
    
    async def request_comprehensive_discovery(
        self,
        discovery_types: List[str] = ["volume_spike", "news_driven", "technical_breakout", "earnings_calendar", "sector_rotation", "real_time_opportunity", "social_sentiment", "cross_asset_arbitrage", "market_regime", "economic_calendar", "options_flow", "earnings_surprise", "rd_agent"],
        max_results_per_type: int = 10
    ) -> str:
        """Request comprehensive market discovery using all available tools."""
        # Filter discovery types based on tool availability
        available_types = ["volume_spike", "news_driven", "technical_breakout", "earnings_calendar", "sector_rotation"]
        
        if NEW_DISCOVERY_TOOLS_AVAILABLE:
            available_types.extend([
                "real_time_opportunity", "social_sentiment", "cross_asset_arbitrage", 
                "market_regime", "economic_calendar", "options_flow", 
                "earnings_surprise", "rd_agent"
            ])
        
        # Only include requested types that are available
        filtered_types = [dt for dt in discovery_types if dt in available_types]
        
        return await self.submit_task(
            task_type="comprehensive_discovery",
            agent_type=AgentType.RESEARCH,
            data={
                "discovery_types": filtered_types,
                "max_results_per_type": max_results_per_type
            },
            priority=1  # High priority for comprehensive discovery
        )
    
    async def request_real_time_opportunity_scan(
        self,
        scan_types: Optional[List[str]] = None,
        min_score: float = 0.7,
        max_results: int = 20,
        sectors: Optional[List[str]] = None
    ) -> str:
        """Request real-time opportunity scanning from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "Real-time opportunity scanning tool not available"
        
        return await self.submit_task(
            task_type="real_time_opportunity_scan",
            agent_type=AgentType.RESEARCH,
            data={
                "scan_types": scan_types,
                "min_score": min_score,
                "max_results": max_results,
                "sectors": sectors
            },
            priority=1
        )
    
    async def request_social_sentiment_analysis(
        self,
        symbols: List[str],
        sentiment_sources: Optional[List[str]] = None,
        time_window_hours: int = 24,
        min_confidence: float = 0.6
    ) -> str:
        """Request social sentiment analysis from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "Social sentiment analysis tool not available"
        
        return await self.submit_task(
            task_type="social_sentiment_analysis",
            agent_type=AgentType.RESEARCH,
            data={
                "symbols": symbols,
                "sentiment_sources": sentiment_sources,
                "time_window_hours": time_window_hours,
                "min_confidence": min_confidence
            },
            priority=2
        )
    
    async def request_cross_asset_arbitrage_detection(
        self,
        asset_pairs: Optional[List[tuple]] = None,
        min_spread_threshold: float = 0.01,
        max_execution_time: int = 300
    ) -> str:
        """Request cross-asset arbitrage detection from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "Cross-asset arbitrage detection tool not available"
        
        return await self.submit_task(
            task_type="cross_asset_arbitrage_detection",
            agent_type=AgentType.RESEARCH,
            data={
                "asset_pairs": asset_pairs,
                "min_spread_threshold": min_spread_threshold,
                "max_execution_time": max_execution_time
            },
            priority=1
        )
    
    async def request_market_regime_detection(
        self,
        lookback_days: int = 60,
        regime_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.75
    ) -> str:
        """Request market regime detection from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "Market regime detection tool not available"
        
        return await self.submit_task(
            task_type="market_regime_detection",
            agent_type=AgentType.RESEARCH,
            data={
                "lookback_days": lookback_days,
                "regime_types": regime_types,
                "confidence_threshold": confidence_threshold
            },
            priority=2
        )
    
    async def request_economic_calendar_monitoring(
        self,
        days_ahead: int = 14,
        impact_levels: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        event_categories: Optional[List[str]] = None
    ) -> str:
        """Request economic calendar monitoring from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "Economic calendar monitoring tool not available"
        
        return await self.submit_task(
            task_type="economic_calendar_monitoring",
            agent_type=AgentType.RESEARCH,
            data={
                "days_ahead": days_ahead,
                "impact_levels": impact_levels,
                "countries": countries,
                "event_categories": event_categories
            },
            priority=2
        )
    
    async def request_options_flow_analysis(
        self,
        symbols: List[str],
        flow_types: Optional[List[str]] = None,
        min_volume_threshold: int = 1000,
        time_window_hours: int = 24
    ) -> str:
        """Request options flow analysis from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "Options flow analysis tool not available"
        
        return await self.submit_task(
            task_type="options_flow_analysis",
            agent_type=AgentType.RESEARCH,
            data={
                "symbols": symbols,
                "flow_types": flow_types,
                "min_volume_threshold": min_volume_threshold,
                "time_window_hours": time_window_hours
            },
            priority=2
        )
    
    async def request_earnings_surprise_prediction(
        self,
        symbols: List[str],
        quarters_ahead: int = 1,
        confidence_threshold: float = 0.7,
        include_guidance: bool = True
    ) -> str:
        """Request earnings surprise prediction from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "Earnings surprise prediction tool not available"
        
        return await self.submit_task(
            task_type="earnings_surprise_prediction",
            agent_type=AgentType.RESEARCH,
            data={
                "symbols": symbols,
                "quarters_ahead": quarters_ahead,
                "confidence_threshold": confidence_threshold,
                "include_guidance": include_guidance
            },
            priority=2
        )
    
    async def request_rd_agent_integration(
        self,
        research_query: str,
        analysis_depth: str = "comprehensive",
        include_backtesting: bool = True,
        max_execution_time: int = 600
    ) -> str:
        """Request RD Agent integration analysis from research agents."""
        if not NEW_DISCOVERY_TOOLS_AVAILABLE:
            return "RD Agent integration tool not available"
        
        return await self.submit_task(
            task_type="rd_agent_integration",
            agent_type=AgentType.RESEARCH,
            data={
                "research_query": research_query,
                "analysis_depth": analysis_depth,
                "include_backtesting": include_backtesting,
                "max_execution_time": max_execution_time
            },
            priority=1
        )
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        return None
    
    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        if task_id in self.active_tasks:
            return "active"
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return "completed" if task.result else "failed"
        else:
            # Check queue
            for task in self.task_queue:
                if task.task_id == task_id:
                    return "queued"
        return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            agent_id: {
                "status": agent_info.status.value,
                "current_tasks": len(agent_info.current_tasks),
                "tasks_completed": agent_info.metrics.tasks_completed,
                "success_rate": agent_info.metrics.success_rate,
                "average_execution_time": agent_info.metrics.average_execution_time,
                "last_activity": agent_info.metrics.last_activity.isoformat() if agent_info.metrics.last_activity else None
            }
            for agent_id, agent_info in self.agents.items()
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            **self.system_metrics,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
            "current_sessions": [s.value for s in self.current_sessions]
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    async def test_agent_orchestrator():
        config_manager = ConfigManager(Path("../config"))
        market_data_aggregator = MarketDataAggregator(config_manager)
        execution_engine = ExecutionEngine(config_manager)
        risk_manager = RiskManager24_7(config_manager)
        backtesting_engine = BacktestingEngine(config_manager, market_data_aggregator)
        trading_orchestrator = TradingOrchestrator24_7(config_manager)
        
        orchestrator = AgentOrchestrator(
            config_manager,
            market_data_aggregator,
            execution_engine,
            risk_manager,
            backtesting_engine,
            trading_orchestrator
        )
        
        # Initialize
        await orchestrator.initialize()
        
        # Submit some test tasks
        task1 = await orchestrator.request_analysis(
            analysis_type="technical",
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="1h"
        )
        
        task2 = await orchestrator.request_strategy_signals(
            strategy_name="momentum",
            symbols=["AAPL", "GOOGL"],
            parameters={"lookback": 20, "threshold": 0.02}
        )
        
        # Wait a bit for processing
        await asyncio.sleep(10)
        
        # Check results
        result1 = await orchestrator.get_task_result(task1)
        result2 = await orchestrator.get_task_result(task2)
        
        print(f"Analysis result: {result1}")
        print(f"Strategy signals: {result2}")
        
        # Get system status
        agent_status = orchestrator.get_agent_status()
        system_metrics = orchestrator.get_system_metrics()
        
        print(f"Agent Status: {agent_status}")
        print(f"System Metrics: {system_metrics}")
        
        # Shutdown
        await orchestrator.shutdown()
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(test_agent_orchestrator())