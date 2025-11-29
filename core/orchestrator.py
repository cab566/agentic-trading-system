#!/usr/bin/env python3
"""
CrewAI Trading System Orchestrator

This module coordinates all trading agents using the CrewAI framework,
managing the complete trading workflow from market analysis to execution.
"""

import asyncio
import logging
import os
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI

from .config_manager import ConfigManager
from .data_manager import UnifiedDataManager
from .health_monitor import HealthMonitor
from tools.market_data_tool import MarketDataTool
from tools.technical_analysis_tool import TechnicalAnalysisTool
from tools.risk_analysis_tool import RiskAnalysisTool
from tools.order_management_tool import OrderManagementTool


class TradingOrchestrator:
    """
    Main orchestrator for the CrewAI-based trading system.
    
    Coordinates multiple specialized agents to execute trading strategies,
    manage risk, and optimize portfolio performance.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trading orchestrator.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent / "config"
        
        # Initialize core components
        self.config_manager = ConfigManager(self.config_path)
        self.data_manager = UnifiedDataManager(self.config_manager)
        self.health_monitor = HealthMonitor(self.config_manager)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize agents and crew
        self.agents: Dict[str, Agent] = {}
        self.crew: Optional[Crew] = None
        self.tools: Dict[str, Any] = {}
        
        # System state
        self.is_running = False
        self.last_cycle_time: Optional[datetime] = None
        self.cycle_count = 0
        
        # Initialize system
        self._initialize_system()
        
        self.logger.info("Trading Orchestrator initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/orchestrator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_system(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize tools
            self._initialize_tools()
            
            # Initialize agents
            self._initialize_agents()
            
            # Initialize crew
            self._initialize_crew()
            
            # Start health monitoring
            self.health_monitor.start_monitoring()
            
            self.logger.info("System initialization completed")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def _initialize_tools(self) -> None:
        """Initialize all tools for agents."""
        self.logger.info("Initializing agent tools...")
        
        # Market data tools
        self.tools['market_data'] = MarketDataTool(data_manager=self.data_manager)
        self.tools['technical_analysis'] = TechnicalAnalysisTool(data_manager=self.data_manager)
        self.tools['risk_analysis'] = RiskAnalysisTool(data_manager=self.data_manager)
        
        # Execution tools
        self.tools['order_management'] = OrderManagementTool(self.config_manager)
        
        self.logger.info(f"Initialized {len(self.tools)} tools")
    
    def _initialize_agents(self) -> None:
        """Initialize all trading agents."""
        self.logger.info("Initializing trading agents...")
        
        agent_configs = self.config_manager.get_agent_configs()
        llm_config = self.config_manager.get_llm_config()
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Create LLM instances
        default_llm = ChatOpenAI(
            model=llm_config['default']['model'],
            temperature=llm_config['default']['temperature'],
            max_tokens=llm_config['default']['max_tokens'],
            api_key=api_key
        )
        
        analysis_llm = ChatOpenAI(
            model=llm_config['analysis']['model'],
            temperature=llm_config['analysis']['temperature'],
            max_tokens=llm_config['analysis']['max_tokens'],
            api_key=api_key
        )
        
        execution_llm = ChatOpenAI(
            model=llm_config['execution']['model'],
            temperature=llm_config['execution']['temperature'],
            max_tokens=llm_config['execution']['max_tokens'],
            api_key=api_key
        )
        
        # Initialize each agent
        for agent_name, config in agent_configs.items():
            # Select appropriate LLM based on agent type
            if agent_name in ['market_analyst', 'research_agent']:
                llm = analysis_llm
            elif agent_name == 'trade_executor':
                llm = execution_llm
            else:
                llm = default_llm
            
            # Get tools for this agent
            agent_tools = [self.tools[tool_name] for tool_name in config.get('tools', []) if tool_name in self.tools]
            
            # Create agent
            agent = Agent(
                role=config['role'],
                goal=config['goal'],
                backstory=config['backstory'],
                tools=agent_tools,
                llm=llm,
                memory=config.get('memory', True),
                verbose=config.get('verbose', True),
                allow_delegation=config.get('allow_delegation', False),
                max_iter=config.get('max_iter', 5),
                max_execution_time=config.get('max_execution_time', 300)
            )
            
            self.agents[agent_name] = agent
            self.logger.info(f"Initialized agent: {agent_name}")
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def _initialize_crew(self) -> None:
        """Initialize the CrewAI crew."""
        self.logger.info("Initializing CrewAI crew...")
        
        crew_config = self.config_manager.get_crew_config()
        
        # Create crew with all agents
        agent_list = list(self.agents.values())
        
        # Determine process type
        process_type = Process.sequential
        if crew_config.get('process') == 'hierarchical':
            process_type = Process.hierarchical
        
        self.crew = Crew(
            agents=agent_list,
            tasks=[],  # Tasks will be created dynamically
            process=process_type,
            memory=crew_config.get('memory', True),
            cache=crew_config.get('cache', True),
            max_rpm=crew_config.get('max_rpm', 10),
            share_crew=crew_config.get('share_crew', False),
            verbose=True
        )
        
        self.logger.info("CrewAI crew initialized successfully")
    
    def _create_trading_tasks(self) -> List[Task]:
        """Create tasks for the current trading cycle."""
        tasks = []
        
        # Market Analysis Task
        market_analysis_task = Task(
            description="""
            Analyze current market conditions and identify trading opportunities.
            
            Your analysis should include:
            1. Overall market sentiment and trend direction
            2. Sector rotation and relative strength analysis
            3. Key technical levels and support/resistance
            4. Volume analysis and market breadth indicators
            5. Specific stock recommendations with entry/exit levels
            
            Focus on actionable insights that can drive trading decisions.
            """,
            agent=self.agents['market_analyst'],
            expected_output="Comprehensive market analysis with specific trading recommendations"
        )
        tasks.append(market_analysis_task)
        
        # Risk Assessment Task
        risk_assessment_task = Task(
            description="""
            Assess portfolio risk and provide position sizing recommendations.
            
            Your assessment should include:
            1. Current portfolio risk metrics (VaR, beta, correlation)
            2. Position sizing for new opportunities
            3. Risk-adjusted return expectations
            4. Stress test results under various market scenarios
            5. Recommendations for risk mitigation
            
            Ensure all recommendations align with risk management guidelines.
            """,
            agent=self.agents['risk_manager'],
            expected_output="Risk assessment report with position sizing recommendations",
            context=[market_analysis_task]
        )
        tasks.append(risk_assessment_task)
        
        # Research Task
        research_task = Task(
            description="""
            Conduct fundamental research on recommended securities.
            
            Your research should include:
            1. Fundamental analysis of recommended stocks
            2. Earnings and financial health assessment
            3. Industry and competitive analysis
            4. News and event impact analysis
            5. Long-term investment thesis validation
            
            Provide detailed research reports to support trading decisions.
            """,
            agent=self.agents['research_agent'],
            expected_output="Detailed research reports on recommended securities",
            context=[market_analysis_task]
        )
        tasks.append(research_task)
        
        # Portfolio Optimization Task
        portfolio_task = Task(
            description="""
            Optimize portfolio allocation and provide rebalancing recommendations.
            
            Your optimization should include:
            1. Current portfolio performance analysis
            2. Asset allocation optimization
            3. Rebalancing recommendations
            4. Performance attribution analysis
            5. Strategic adjustments based on market conditions
            
            Ensure recommendations align with investment objectives and risk tolerance.
            """,
            agent=self.agents['portfolio_manager'],
            expected_output="Portfolio optimization report with rebalancing recommendations",
            context=[market_analysis_task, risk_assessment_task, research_task]
        )
        tasks.append(portfolio_task)
        
        # Trade Execution Task
        execution_task = Task(
            description="""
            Execute approved trades with optimal timing and minimal market impact.
            
            Your execution should include:
            1. Order placement with appropriate order types
            2. Timing optimization to minimize slippage
            3. Market impact analysis
            4. Trade monitoring and adjustment
            5. Execution quality reporting
            
            Ensure all trades are executed efficiently and within risk parameters.
            """,
            agent=self.agents['trade_executor'],
            expected_output="Trade execution report with performance metrics",
            context=[portfolio_task, risk_assessment_task]
        )
        tasks.append(execution_task)
        
        return tasks
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Execute a complete trading cycle."""
        self.logger.info(f"Starting trading cycle #{self.cycle_count + 1}")
        
        try:
            # Update crew with new tasks
            tasks = self._create_trading_tasks()
            self.crew.tasks = tasks
            
            # Execute the crew
            start_time = datetime.now()
            result = await asyncio.to_thread(self.crew.kickoff)
            end_time = datetime.now()
            
            # Update cycle tracking
            self.cycle_count += 1
            self.last_cycle_time = end_time
            cycle_duration = (end_time - start_time).total_seconds()
            
            # Log results
            self.logger.info(f"Trading cycle #{self.cycle_count} completed in {cycle_duration:.2f} seconds")
            
            # Update health monitor
            self.health_monitor.record_cycle_completion(cycle_duration)
            
            return {
                'cycle_number': self.cycle_count,
                'start_time': start_time,
                'end_time': end_time,
                'duration': cycle_duration,
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")
            self.health_monitor.record_cycle_error(str(e))
            
            return {
                'cycle_number': self.cycle_count,
                'start_time': datetime.now(),
                'end_time': datetime.now(),
                'duration': 0,
                'result': None,
                'status': 'error',
                'error': str(e)
            }
    
    async def start_continuous_trading(self, cycle_interval: int = 300) -> None:
        """Start continuous trading with specified cycle interval."""
        self.logger.info(f"Starting continuous trading with {cycle_interval}s intervals")
        self.is_running = True
        
        while self.is_running:
            try:
                # Check if market is open (if configured)
                if self._should_trade():
                    await self.run_trading_cycle()
                else:
                    self.logger.info("Market closed or trading disabled, skipping cycle")
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, stopping trading")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        self.is_running = False
        self.logger.info("Continuous trading stopped")
    
    def _should_trade(self) -> bool:
        """Check if trading should occur based on market hours and configuration."""
        env_config = self.config_manager.get_environment_config()
        
        # Check if market hours only trading is enabled
        if env_config.get('market_hours_only', True):
            current_time = datetime.now().time()
            market_open = time(9, 30)  # 9:30 AM
            market_close = time(16, 0)  # 4:00 PM
            
            # Check if current time is within market hours
            if not (market_open <= current_time <= market_close):
                return False
            
            # Check if it's a weekday
            if datetime.now().weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
        
        return True
    
    def stop_trading(self) -> None:
        """Stop continuous trading."""
        self.logger.info("Stopping trading orchestrator")
        self.is_running = False
        self.health_monitor.stop_monitoring()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time,
            'agents_count': len(self.agents),
            'tools_count': len(self.tools),
            'health_status': self.health_monitor.get_status(),
            'data_manager_status': self.data_manager.get_status()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return self.health_monitor.get_performance_metrics()
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current trading positions."""
        try:
            # Get positions from data manager or portfolio tracker
            positions = []
            total_value = 0.0
            
            # This would integrate with your actual position tracking system
            # For now, return empty positions with proper structure
            return {
                "positions": positions,
                "total_value": total_value,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"positions": [], "total_value": 0.0, "timestamp": datetime.utcnow().isoformat()}
    
    async def get_recent_trades(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent trades."""
        try:
            # Get trades from data manager or trade history
            trades = []
            
            # This would integrate with your actual trade tracking system
            return {
                "trades": trades,
                "total_count": len(trades),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return {"trades": [], "total_count": 0, "timestamp": datetime.utcnow().isoformat()}
    
    async def get_portfolio_overview(self) -> Dict[str, Any]:
        """Get portfolio overview."""
        try:
            # Get portfolio data from data manager
            return {
                "total_value": 0.0,
                "cash_balance": 0.0,
                "positions_value": 0.0,
                "day_change": 0.0,
                "day_change_percent": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio: {e}")
            return {
                "total_value": 0.0,
                "cash_balance": 0.0,
                "positions_value": 0.0,
                "day_change": 0.0,
                "day_change_percent": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_orders(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Get orders with optional status filter."""
        try:
            # Get orders from order management system
            orders = []
            
            # This would integrate with your actual order tracking system
            return {
                "orders": orders,
                "total_count": len(orders),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return {"orders": [], "total_count": 0, "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = TradingOrchestrator()
        
        # Run a single trading cycle
        result = await orchestrator.run_trading_cycle()
        print(f"Trading cycle result: {result}")
        
        # Or start continuous trading
        # await orchestrator.start_continuous_trading(cycle_interval=300)
    
    # Commented out to prevent event loop conflicts when imported
    # asyncio.run(main())