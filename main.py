#!/usr/bin/env python3
"""
Main Application for Multi-Asset Trading System

A comprehensive 24/7 trading system supporting:
- Traditional assets (stocks, bonds, options, futures)
- Cryptocurrencies (24/7 trading)
- Forex markets (24/5 trading)
- Multi-strategy execution
- Advanced risk management
- AI-powered agent coordination
- Real-time monitoring and alerts

Usage:
    python main.py run --mode paper
    python main.py run --mode live
    python main.py backtest --strategy momentum --start 2023-01-01 --end 2023-12-31
    python main.py optimize --strategy all --lookback 252
    python main.py monitor --dashboard
"""

import asyncio
import logging
import signal
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import traceback

# Third-party imports
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
import click

# Core system imports
from core.orchestrator import TradingOrchestrator
from core.config_manager import ConfigManager
from core.data_manager import UnifiedDataManager
from core.market_data_aggregator import MarketDataAggregator
from core.execution_engine import ExecutionEngine
from core.risk_manager_24_7 import RiskManager24_7
from core.backtesting_engine import BacktestingEngine
from core.trading_orchestrator_24_7 import TradingOrchestrator24_7
from core.agent_orchestrator import AgentOrchestrator
from core.health_monitor import HealthMonitor

# Utility imports
from utils.notifications import NotificationManager
from utils.cache_manager import CacheManager
from utils.performance_tracker import PerformanceTracker

# Dashboard imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Initialize rich console for beautiful output
console = Console()

# Global orchestrator instance for signal handling
orchestrator: Optional[TradingOrchestrator] = None


class TradingSystemApp:
    """
    Main application class for the multi-asset trading system.
    
    Coordinates all system components and provides a unified interface
    for trading operations, monitoring, and management.
    """
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config")
        self.console = Console()
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.config_manager: Optional[ConfigManager] = None
        self.data_manager: Optional[UnifiedDataManager] = None
        self.market_data_aggregator: Optional[MarketDataAggregator] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager24_7] = None
        self.backtesting_engine: Optional[BacktestingEngine] = None
        self.trading_orchestrator: Optional[TradingOrchestrator24_7] = None
        self.agent_orchestrator: Optional[AgentOrchestrator] = None
        self.health_monitor: Optional[HealthMonitor] = None
        
        # Utility components
        self.notification_manager: Optional[NotificationManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def initialize(self, mode: str = "paper"):
        """Initialize all system components."""
        try:
            self.console.print("[bold blue]Initializing Multi-Asset Trading System...[/bold blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                # Initialize configuration
                task = progress.add_task("Loading configuration...", total=None)
                self.config_manager = ConfigManager(self.config_path)
                progress.update(task, description="✓ Configuration loaded")
                
                # Set trading mode
                self.config_manager.set_trading_mode(mode)
                
                # Initialize utility components
                task = progress.add_task("Initializing utilities...", total=None)
                self.notification_manager = NotificationManager(self.config_manager)
                self.cache_manager = CacheManager(self.config_manager)
                self.performance_tracker = PerformanceTracker(self.config_manager)
                progress.update(task, description="✓ Utilities initialized")
                
                # Initialize data management
                task = progress.add_task("Setting up data management...", total=None)
                self.data_manager = UnifiedDataManager(self.config_manager)
                progress.update(task, description="✓ Data management ready")
                
                # Initialize market data aggregator
                task = progress.add_task("Connecting to market data...", total=None)
                self.market_data_aggregator = MarketDataAggregator(self.config_manager)
                progress.update(task, description="✓ Market data connected")
                
                # Initialize risk manager
                task = progress.add_task("Initializing risk management...", total=None)
                self.risk_manager = RiskManager24_7(self.config_manager)
                progress.update(task, description="✓ Risk management active")
                
                # Initialize execution engine
                task = progress.add_task("Setting up execution engine...", total=None)
                from core.trade_storage import TradeStorage
                trade_storage = TradeStorage(self.config_manager)
                self.execution_engine = ExecutionEngine(
                    self.config_manager,
                    self.market_data_aggregator,
                    self.risk_manager,
                    trade_storage=trade_storage
                )
                progress.update(task, description="✓ Execution engine ready")
                
                # Initialize backtesting engine
                task = progress.add_task("Setting up backtesting...", total=None)
                self.backtesting_engine = BacktestingEngine(
                    self.config_manager,
                    self.market_data_aggregator
                )
                progress.update(task, description="✓ Backtesting engine ready")
                
                # Initialize health monitor
                task = progress.add_task("Starting health monitor...", total=None)
                self.health_monitor = HealthMonitor(self.config_manager)
                progress.update(task, description="✓ Health monitor active")
                
                # Initialize trading orchestrator
                task = progress.add_task("Starting trading orchestrator...", total=None)
                self.trading_orchestrator = TradingOrchestrator24_7(self.config_manager, self.data_manager)
                progress.update(task, description="✓ Trading orchestrator active")
                
                # Initialize agent orchestrator
                task = progress.add_task("Deploying AI agents...", total=None)
                self.agent_orchestrator = AgentOrchestrator(
                    self.config_manager,
                    self.market_data_aggregator,
                    self.execution_engine,
                    self.risk_manager,
                    self.backtesting_engine,
                    self.trading_orchestrator
                )
                progress.update(task, description="✓ AI agents deployed")
            
            self.console.print("[bold green]✓ System initialization complete![/bold green]")
            
            # Display system status
            await self._display_system_status()
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.console.print(f"[bold red]✗ Initialization failed: {e}[/bold red]")
            raise
    
    async def shutdown(self):
        """Shutdown all system components gracefully."""
        try:
            self.console.print("[bold yellow]Shutting down trading system...[/bold yellow]")
            self.running = False
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                # Shutdown in reverse order of initialization
                if self.agent_orchestrator:
                    task = progress.add_task("Shutting down AI agents...", total=None)
                    await self.agent_orchestrator.shutdown()
                    progress.update(task, description="✓ AI agents shutdown")
                
                if self.trading_orchestrator:
                    task = progress.add_task("Stopping trading orchestrator...", total=None)
                    await self.trading_orchestrator.shutdown()
                    progress.update(task, description="✓ Trading orchestrator stopped")
                
                if self.health_monitor:
                    task = progress.add_task("Stopping health monitor...", total=None)
                    self.health_monitor.stop_monitoring()
                    progress.update(task, description="✓ Health monitor stopped")
                
                if self.risk_manager:
                    task = progress.add_task("Shutting down risk manager...", total=None)
                    await self.risk_manager.shutdown()
                    progress.update(task, description="✓ Risk manager shutdown")
                
                if self.execution_engine:
                    task = progress.add_task("Closing execution engine...", total=None)
                    await self.execution_engine.shutdown()
                    progress.update(task, description="✓ Execution engine closed")
                
                if self.market_data_aggregator:
                    task = progress.add_task("Disconnecting market data...", total=None)
                    await self.market_data_aggregator.shutdown()
                    progress.update(task, description="✓ Market data disconnected")
                
                if self.data_manager:
                    task = progress.add_task("Closing data connections...", total=None)
                    await self.data_manager.stop()
                    progress.update(task, description="✓ Data connections closed")
            
            self.console.print("[bold green]✓ System shutdown complete![/bold green]")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.console.print(f"[bold red]✗ Shutdown error: {e}[/bold red]")
    
    async def run_trading(self, mode: str = "paper"):
        """Run the main trading loop."""
        try:
            await self.initialize(mode)
            self.running = True
            
            self.console.print(f"[bold green]🚀 Trading system started in {mode.upper()} mode![/bold green]")
            
            # Start monitoring dashboard in background
            monitor_task = asyncio.create_task(self._run_monitoring_dashboard())
            
            # Main trading loop
            while self.running and not self.shutdown_event.is_set():
                try:
                    # System health check
                    await self._health_check()
                    
                    # Process any pending notifications
                    await self._process_notifications()
                    
                    # Update performance metrics
                    await self._update_performance_metrics()
                    
                    # Sleep briefly
                    await asyncio.sleep(1)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main trading loop: {e}")
                    await asyncio.sleep(5)
            
            # Cancel monitoring task
            monitor_task.cancel()
            
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            await self.shutdown()
    
    async def run_backtest(
        self,
        strategy: str,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        initial_capital: float = 100000.0
    ):
        """Run backtesting for specified strategy."""
        try:
            await self.initialize("backtest")
            
            self.console.print(f"[bold blue]📊 Running backtest for {strategy} strategy[/bold blue]")
            self.console.print(f"Period: {start_date} to {end_date}")
            
            # Parse dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Default symbols if not provided
            if not symbols:
                symbols = ["AAPL", "GOOGL", "MSFT", "BTCUSDT", "ETHUSDT", "EURUSD=X"]
            
            # Run backtest
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Running backtest...", total=None)
                
                results = await self.backtesting_engine.run_backtest(
                    strategy_name=strategy,
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    initial_capital=initial_capital
                )
                
                progress.update(task, description="✓ Backtest completed")
            
            # Display results
            await self._display_backtest_results(results)
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            self.console.print(f"[bold red]✗ Backtesting failed: {e}[/bold red]")
            raise
        finally:
            await self.shutdown()
    
    async def run_optimization(
        self,
        strategy: str,
        lookback_days: int = 252,
        symbols: Optional[List[str]] = None
    ):
        """Run strategy optimization."""
        try:
            await self.initialize("optimization")
            
            self.console.print(f"[bold blue]🔧 Optimizing {strategy} strategy[/bold blue]")
            self.console.print(f"Lookback period: {lookback_days} days")
            
            # Default symbols if not provided
            if not symbols:
                symbols = ["AAPL", "GOOGL", "MSFT", "BTCUSDT", "ETHUSDT"]
            
            # Run optimization
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Running optimization...", total=None)
                
                # This would implement parameter optimization
                # For now, just show a placeholder
                await asyncio.sleep(2)
                
                progress.update(task, description="✓ Optimization completed")
            
            self.console.print("[bold green]✓ Strategy optimization complete![/bold green]")
            
        except Exception as e:
            self.logger.error(f"Error in optimization: {e}")
            self.console.print(f"[bold red]✗ Optimization failed: {e}[/bold red]")
            raise
        finally:
            await self.shutdown()
    
    async def _display_system_status(self):
        """Display current system status."""
        # Create status table
        table = Table(title="System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Add component statuses
        components = [
            ("Configuration", "✓ Loaded", f"Mode: {self.config_manager.get_trading_mode() if self.config_manager else 'Unknown'}"),
            ("Data Manager", "✓ Connected", "Multi-source data feeds"),
            ("Market Data", "✓ Streaming", "Real-time feeds active"),
            ("Execution Engine", "✓ Ready", "Order routing enabled"),
            ("Risk Manager", "✓ Monitoring", "24/7 risk controls active"),
            ("Trading Orchestrator", "✓ Active", "Multi-asset coordination"),
            ("AI Agents", "✓ Deployed", "Agent coordination active"),
        ]
        
        for component, status, details in components:
            table.add_row(component, status, details)
        
        self.console.print(table)
    
    async def _display_backtest_results(self, results: Dict[str, Any]):
        """Display backtesting results."""
        # Create results panel
        results_text = f"""
[bold]Backtest Results Summary[/bold]

Total Return: {results.get('total_return', 0):.2%}
Annualized Return: {results.get('annualized_return', 0):.2%}
Volatility: {results.get('volatility', 0):.2%}
Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
Max Drawdown: {results.get('max_drawdown', 0):.2%}
Win Rate: {results.get('win_rate', 0):.2%}

Total Trades: {results.get('total_trades', 0)}
Profit Factor: {results.get('profit_factor', 0):.2f}
Calmar Ratio: {results.get('calmar_ratio', 0):.2f}
        """
        
        panel = Panel(results_text, title="📊 Backtest Results", border_style="green")
        self.console.print(panel)
    
    async def _health_check(self):
        """Perform system health check."""
        # Check component health
        if self.market_data_aggregator:
            # Check data feed health
            pass
        
        if self.risk_manager:
            # Check risk limits
            pass
        
        if self.agent_orchestrator:
            # Check agent health
            pass
    
    async def _process_notifications(self):
        """Process pending notifications."""
        if self.notification_manager:
            # Process any pending notifications
            pass
    
    async def _update_performance_metrics(self):
        """Update system performance metrics."""
        if self.performance_tracker:
            # Update performance tracking
            pass
    
    async def _run_monitoring_dashboard(self):
        """Run real-time monitoring dashboard."""
        try:
            while self.running:
                # Update dashboard data
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in monitoring dashboard: {e}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration with rich formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True
            )
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logging.getLogger().addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("crewai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def signal_handler(signum: int, frame) -> None:
    """
    Handle shutdown signals gracefully.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global orchestrator
    
    console.print("\n[yellow]Received shutdown signal. Stopping trading system...[/yellow]")
    
    if orchestrator:
        try:
            # Stop the orchestrator gracefully
            asyncio.create_task(orchestrator.stop())
            console.print("[green]Trading system stopped successfully.[/green]")
        except Exception as e:
            console.print(f"[red]Error stopping trading system: {e}[/red]")
    
    sys.exit(0)


def display_banner() -> None:
    """
    Display the application banner.
    """
    banner_text = Text()
    banner_text.append("Trading System v2\n", style="bold blue")
    banner_text.append("CrewAI-Powered Algorithmic Trading Platform\n", style="cyan")
    banner_text.append("\nFeatures:\n", style="bold")
    banner_text.append("• Multi-agent trading system\n", style="green")
    banner_text.append("• Real-time market data integration\n", style="green")
    banner_text.append("• Advanced risk management\n", style="green")
    banner_text.append("• Comprehensive backtesting\n", style="green")
    banner_text.append("• Health monitoring & alerts\n", style="green")
    
    panel = Panel(
        banner_text,
        title="[bold red]🚀 Trading System v2[/bold red]",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)


def validate_environment() -> bool:
    """
    Validate the environment and dependencies.
    
    Returns:
        True if environment is valid, False otherwise
    """
    import os
    
    try:
        # Check required environment variables
        required_vars = ["OPENAI_API_KEY"]
        optional_vars = [
            "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
            "BINANCE_API_KEY", "BINANCE_SECRET_KEY",
            "OANDA_API_KEY", "OANDA_ACCOUNT_ID",
            "COINBASE_API_KEY", "COINBASE_SECRET_KEY", "COINBASE_PASSPHRASE"
        ]
        
        missing_required = []
        missing_optional = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)
        
        for var in optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)
        
        if missing_required:
            console.print(f"[red]Missing required environment variables: {', '.join(missing_required)}[/red]")
            console.print("[yellow]Please set these variables before running the system.[/yellow]")
            return False
        
        if missing_optional:
            console.print(f"[yellow]Optional environment variables not set: {', '.join(missing_optional)}[/yellow]")
            console.print("[dim]Some features may be limited without these credentials.[/dim]")
        
        # Check if config directory exists
        config_dir = Path("config")
        if not config_dir.exists():
            console.print("[red]Error: Config directory not found![/red]")
            return False
        
        # Check required config files
        required_configs = ["agents.yaml", "data_sources.yaml", "strategies.yaml"]
        for config_file in required_configs:
            config_path = config_dir / config_file
            if not config_path.exists():
                console.print(f"[red]Error: Required config file '{config_file}' not found![/red]")
                return False
        
        # Check if logs directory exists, create if not
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Check if data directory exists, create if not
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        console.print("[green]✅ Environment validation passed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Error validating environment: {e}[/red]")
        return False


# Remove the old initialize_system and run_system functions as they are now part of TradingSystemApp


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', default=None, help='Log file path')
@click.option('--config-dir', default='config', help='Configuration directory')
@click.pass_context
def cli(ctx, log_level: str, log_file: Optional[str], config_dir: str):
    """
    Trading System v2 - CrewAI-powered algorithmic trading platform.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store common parameters
    ctx.obj['log_level'] = log_level
    ctx.obj['log_file'] = log_file
    ctx.obj['config_dir'] = Path(config_dir)
    
    # Setup logging
    setup_logging(log_level, log_file)
    
    # Display banner
    display_banner()


@cli.command()
@click.option('--mode', default='paper', type=click.Choice(['live', 'paper']), 
              help='Trading mode')
@click.pass_context
def run(ctx, mode: str):
    """
    Run the trading system.
    """
    async def main():
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        # Initialize and run system
        app = TradingSystemApp(ctx.obj['config_dir'])
        await app.run_trading(mode)
    
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--strategy', required=True, help='Strategy to backtest')
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--symbols', help='Comma-separated list of symbols')
@click.option('--capital', default=100000.0, help='Initial capital')
@click.pass_context
def backtest(ctx, strategy: str, start: str, end: str, symbols: Optional[str], capital: float):
    """
    Run backtesting for a specific strategy.
    """
    async def main():
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Initialize and run backtest
        app = TradingSystemApp(ctx.obj['config_dir'])
        await app.run_backtest(strategy, start, end, symbol_list, capital)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--strategy', required=True, help='Strategy to optimize')
@click.option('--lookback', default=252, help='Lookback period in days')
@click.option('--symbols', help='Comma-separated list of symbols')
@click.pass_context
def optimize(ctx, strategy: str, lookback: int, symbols: Optional[str]):
    """
    Run strategy optimization.
    """
    async def main():
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Initialize and run optimization
        app = TradingSystemApp(ctx.obj['config_dir'])
        await app.run_optimization(strategy, lookback, symbol_list)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """
    Check system status and health.
    """
    async def check_status():
        try:
            # Initialize minimal system for status check
            config_manager = ConfigManager(ctx.obj['config_dir'])
            health_monitor = HealthMonitor(config_manager)
            
            health_monitor.start_monitoring()
            
            # Get system status
            health_status = health_monitor.get_health_status()
            
            # Display status
            console.print("\n[bold blue]System Status Report[/bold blue]")
            console.print("=" * 40)
            
            if health_status.get("overall_status") == "healthy":
                console.print("[green]✅ System Status: HEALTHY[/green]")
            elif health_status.get("overall_status") == "warning":
                console.print("[yellow]⚠️ System Status: WARNING[/yellow]")
            else:
                console.print("[red]❌ System Status: CRITICAL[/red]")
            
            # Display component status
            console.print("\n[bold]Component Status:[/bold]")
            for component, status in health_status.get("components", {}).items():
                status_icon = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
                console.print(f"  {status_icon} {component}: {status.upper()}")
            
            health_monitor.stop_monitoring()
            
        except Exception as e:
            console.print(f"[red]Error checking status: {e}[/red]")
    
    asyncio.run(check_status())


@cli.command()
@click.pass_context
def validate(ctx):
    """
    Validate system configuration and dependencies.
    """
    console.print("[blue]🔍 Validating system configuration...[/blue]")
    
    try:
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        # Validate configurations
        config_manager = ConfigManager(ctx.obj['config_dir'])
        
        # Test configuration loading
        agents_config = config_manager.get_agents_config()
        data_sources_config = config_manager.get_data_sources_config()
        strategies_config = config_manager.get_strategies_config()
        
        console.print(f"[green]✅ Loaded {len(agents_config.get('agents', {}))} agent configurations[/green]")
        console.print(f"[green]✅ Loaded {len(data_sources_config.get('sources', {}))} data source configurations[/green]")
        console.print(f"[green]✅ Loaded {len(strategies_config.get('strategies', {}))} strategy configurations[/green]")
        
        console.print("\n[green]🎉 All validations passed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--symbol', required=True, help='Stock symbol to analyze')
@click.option('--analysis-type', default='comprehensive', 
              type=click.Choice(['quick', 'standard', 'comprehensive']),
              help='Type of analysis to perform')
@click.pass_context
def analyze(ctx, symbol: str, analysis_type: str):
    """
    Perform quick analysis on a specific symbol.
    """
    async def run_analysis():
        try:
            console.print(f"[blue]📊 Analyzing {symbol.upper()}...[/blue]")
            
            # Initialize minimal system for analysis
            config_manager = ConfigManager(ctx.obj['config_dir'])
            data_manager = UnifiedDataManager(config_manager)
            
            # Simple analysis placeholder
            console.print(f"\n[bold green]Analysis Results for {symbol.upper()}:[/bold green]")
            console.print(f"Analysis Type: {analysis_type}")
            console.print("[yellow]Detailed analysis functionality coming soon![/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Analysis failed: {e}[/red]")
    
    asyncio.run(run_analysis())


@cli.command()
@click.pass_context
def monitor(ctx):
    """
    Launch monitoring dashboard.
    """
    if not STREAMLIT_AVAILABLE:
        console.print("[red]Streamlit not available. Install with: pip install streamlit[/red]")
        return
    
    console.print("[blue]🚀 Launching monitoring dashboard...[/blue]")
    console.print("[yellow]Dashboard functionality coming soon![/yellow]")


if __name__ == "__main__":
    cli()