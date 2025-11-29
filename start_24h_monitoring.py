#!/usr/bin/env python3
"""
24-Hour Trading System Monitoring Launcher

This script launches the complete 24-hour monitoring system including:
- Trading system orchestrator
- Comprehensive monitoring
- Unified dashboard
- Logging and alerting
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import subprocess
import time

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from monitoring.comprehensive_24h_monitor import ComprehensiveMonitor
    from monitoring.unified_dashboard import UnifiedDashboard
    from main import TradingSystemApp
    from core.orchestrator import TradingOrchestrator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the system is properly configured.")
    sys.exit(1)

class TradingSystem24HLauncher:
    """Launcher for 24-hour trading system monitoring"""
    
    def __init__(self, duration_hours: int = 24, dashboard_port: int = 5000):
        self.duration_hours = duration_hours
        self.dashboard_port = dashboard_port
        self.logger = self._setup_logging()
        
        # System components
        self.trading_system = None
        self.monitor = None
        self.dashboard = None
        
        # Process management
        self.running = False
        self.tasks = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Setup logging configuration
        log_filename = f"trading_system_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized - Log file: {log_filename}")
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        self.logger.info("Checking system prerequisites...")
        
        # Check environment variables
        required_env_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY',
            'ALPACA_BASE_URL'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            self.logger.error("Please check your .env file and ensure all Alpaca API credentials are set.")
            return False
        
        # Check if .env file exists
        env_file = Path('.env')
        if not env_file.exists():
            self.logger.warning(".env file not found. Creating template...")
            self._create_env_template()
            return False
        
        # Test API connectivity
        try:
            from core.alpaca_client import AlpacaClient
            # Initialize Alpaca client based on TRADING_MODE environment variable
            trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
            paper_trading = trading_mode == 'paper'
            alpaca_client = AlpacaClient(paper_trading=paper_trading)
            print(f"Alpaca client initialized in {'PAPER' if paper_trading else 'LIVE'} trading mode")
            await alpaca_client.test_connection()
            self.logger.info("✅ Alpaca API connection successful")
        except Exception as e:
            self.logger.error(f"❌ Alpaca API connection failed: {e}")
            return False
        
        # Check required packages
        required_packages = ['flask', 'flask-socketio', 'plotly', 'pandas', 'aiohttp', 'psutil']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.info(f"Installing missing packages: {missing_packages}")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
                self.logger.info("✅ Required packages installed successfully")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Failed to install packages: {e}")
                return False
        
        self.logger.info("✅ All prerequisites satisfied")
        return True
    
    def _create_env_template(self):
        """Create .env template file"""
        env_template = '''
# Alpaca API Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use https://api.alpaca.markets for live trading

# Trading Configuration
TRADING_MODE=PAPER  # PAPER or LIVE
MAX_POSITION_SIZE=1000
RISK_LIMIT=0.02

# Monitoring Configuration
DASHBOARD_PORT=5000
ALERT_EMAIL=your_email@example.com
SLACK_WEBHOOK_URL=your_slack_webhook_url_here

# Logging Configuration
LOG_LEVEL=INFO
LOG_RETENTION_DAYS=30
'''
        
        with open('.env', 'w') as f:
            f.write(env_template)
        
        self.logger.info("Created .env template file. Please update with your actual credentials.")
    
    async def _initialize_components(self) -> bool:
        """Initialize all system components"""
        self.logger.info("Initializing system components...")
        
        try:
            # Initialize trading system
            self.logger.info("Initializing trading system...")
            self.trading_system = TradingSystemApp()
            # Note: TradingSystemApp initialization might need to be adapted for async
            
            # Initialize monitoring system
            self.logger.info("Initializing monitoring system...")
            self.monitor = ComprehensiveMonitor()
            if not await self.monitor.initialize():
                self.logger.error("Failed to initialize monitoring system")
                return False
            
            # Initialize dashboard
            self.logger.info("Initializing dashboard...")
            self.dashboard = UnifiedDashboard(port=self.dashboard_port)
            if not await self.dashboard.initialize():
                self.logger.error("Failed to initialize dashboard")
                return False
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    async def _start_trading_system(self):
        """Start the trading system"""
        self.logger.info("🚀 Starting trading system...")
        
        try:
            # This would start the actual trading orchestrator
            # For now, we'll simulate trading activity
            while self.running:
                self.logger.info("Trading system heartbeat - monitoring for signals...")
                
                # Here you would integrate with the actual trading orchestrator
                # Example: await self.trading_system.run_trading_cycle()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
        except Exception as e:
            self.logger.error(f"Trading system error: {e}")
    
    async def _start_monitoring(self):
        """Start the monitoring system"""
        self.logger.info("📊 Starting monitoring system...")
        
        try:
            await self.monitor.start_monitoring()
        except Exception as e:
            self.logger.error(f"Monitoring system error: {e}")
    
    async def _start_dashboard(self):
        """Start the dashboard in a separate process"""
        self.logger.info(f"🌐 Starting dashboard on port {self.dashboard_port}...")
        
        try:
            # Run dashboard in a separate thread since it's Flask-based
            import threading
            
            def run_dashboard():
                self.dashboard.run(debug=False)
            
            dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            dashboard_thread.start()
            
            self.logger.info(f"✅ Dashboard started at http://localhost:{self.dashboard_port}")
            
            # Keep the dashboard running
            while self.running:
                await asyncio.sleep(60)
                
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
    
    async def _monitor_system_health(self):
        """Monitor overall system health"""
        self.logger.info("❤️ Starting system health monitoring...")
        
        start_time = datetime.now()
        
        while self.running:
            try:
                current_time = datetime.now()
                runtime = current_time - start_time
                
                # Log system status every hour
                if runtime.total_seconds() % 3600 < 60:  # Every hour (with 1-minute tolerance)
                    status_report = await self.monitor.get_status_report() if self.monitor else {}
                    
                    self.logger.info(f"""\n
🕐 HOURLY SYSTEM STATUS REPORT
{'='*50}
Runtime: {runtime}
Monitoring Active: {status_report.get('status', 'unknown')}
Total Trades: {status_report.get('total_trades', 0)}
Successful Trades: {status_report.get('successful_trades', 0)}
Failed Trades: {status_report.get('failed_trades', 0)}
Total Errors: {status_report.get('total_errors', 0)}
Total Alerts: {status_report.get('total_alerts', 0)}
Dashboard: http://localhost:{self.dashboard_port}
{'='*50}
""")
                
                # Check if we've reached the duration limit
                if runtime.total_seconds() >= self.duration_hours * 3600:
                    self.logger.info(f"⏰ {self.duration_hours}-hour monitoring period completed")
                    self.running = False
                    break
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def start(self):
        """Start the complete 24-hour monitoring system"""
        self.logger.info(f"🚀 Starting 24-hour trading system monitoring ({self.duration_hours} hours)")
        self.logger.info(f"Start time: {datetime.now()}")
        
        # Check prerequisites
        if not await self._check_prerequisites():
            self.logger.error("❌ Prerequisites not met. Exiting.")
            return False
        
        # Initialize components
        if not await self._initialize_components():
            self.logger.error("❌ Component initialization failed. Exiting.")
            return False
        
        # Start all systems
        self.running = True
        
        try:
            # Create tasks for all components
            self.tasks = [
                asyncio.create_task(self._start_trading_system()),
                asyncio.create_task(self._start_monitoring()),
                asyncio.create_task(self._start_dashboard()),
                asyncio.create_task(self._monitor_system_health())
            ]
            
            self.logger.info("✅ All systems started successfully")
            self.logger.info(f"🌐 Dashboard available at: http://localhost:{self.dashboard_port}")
            self.logger.info(f"⏰ Monitoring duration: {self.duration_hours} hours")
            self.logger.info("🛑 Press Ctrl+C to stop monitoring")
            
            # Wait for all tasks to complete or for shutdown signal
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            await self._shutdown()
        
        return True
    
    async def _shutdown(self):
        """Graceful shutdown of all components"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop monitoring and generate final report
        if self.monitor:
            try:
                final_report = await self.monitor.stop_monitoring()
                self.logger.info("📋 Final monitoring report generated")
            except Exception as e:
                self.logger.error(f"Error generating final report: {e}")
        
        self.logger.info("✅ Shutdown completed")

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="24-Hour Trading System Monitor")
    parser.add_argument("--duration", type=int, default=24, help="Monitoring duration in hours")
    parser.add_argument("--dashboard-port", type=int, default=5000, help="Dashboard port")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Print startup banner
    print(f"""
{'='*60}
🚀 24-HOUR TRADING SYSTEM MONITORING LAUNCHER
{'='*60}
Duration: {args.duration} hours
Dashboard Port: {args.dashboard_port}
Log Level: {args.log_level}
Start Time: {datetime.now()}
{'='*60}
""")
    
    async def main():
        launcher = TradingSystem24HLauncher(
            duration_hours=args.duration,
            dashboard_port=args.dashboard_port
        )
        
        success = await launcher.start()
        
        if success:
            print("\n✅ 24-hour monitoring session completed successfully")
        else:
            print("\n❌ 24-hour monitoring session failed")
            sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)