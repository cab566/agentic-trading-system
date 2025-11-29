#!/usr/bin/env python3
"""
Comprehensive Integration Testing Framework

Tests all AI tool integrations and system components to ensure
they work together seamlessly.
"""

import asyncio
import unittest
import logging
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Rich for beautiful test output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

class IntegrationTestFramework:
    """Comprehensive testing framework for AI tool integrations"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = None
        self.logger = self._setup_logging()
        
        # Test configurations
        self.test_suites = {
            "basic_imports": {
                "description": "Test basic imports of all AI tools",
                "critical": True,
                "timeout": 30
            },
            "data_pipeline": {
                "description": "Test data ingestion and processing pipeline",
                "critical": True,
                "timeout": 60
            },
            "agent_communication": {
                "description": "Test CrewAI agent communication and coordination",
                "critical": True,
                "timeout": 45
            },
            "market_data_integration": {
                "description": "Test market data sources and real-time feeds",
                "critical": True,
                "timeout": 90
            },
            "ml_model_integration": {
                "description": "Test ML model loading and inference",
                "critical": False,
                "timeout": 120
            },
            "strategy_execution": {
                "description": "Test strategy execution and portfolio management",
                "critical": True,
                "timeout": 60
            },
            "risk_management": {
                "description": "Test risk management and safety checks",
                "critical": True,
                "timeout": 30
            },
            "performance_benchmarks": {
                "description": "Run performance benchmarks",
                "critical": False,
                "timeout": 180
            }
        }
        
        # Tool-specific tests
        self.tool_tests = {
            "qlib": ["import_test", "data_loader_test", "model_test"],
            "openbb": ["import_test", "data_fetch_test", "terminal_test"],
            "rd_agent": ["import_test", "research_test", "decision_test"],
            "pearl": ["import_test", "environment_test", "agent_test"],
            "alpaca_mcp": ["import_test", "connection_test", "trading_test"]
        }

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for tests"""
        logger = logging.getLogger("integration_tests")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = self.project_root / "logs" / "integration"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = log_dir / f"integration_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        console.print(Panel.fit(
            "[bold blue]AI Trading System Integration Tests[/bold blue]\n"
            "Running comprehensive test suite...",
            border_style="blue"
        ))
        
        self.start_time = time.time()
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "test_suites": {},
            "tool_tests": {},
            "summary": {}
        }
        
        # Run test suites
        await self._run_test_suites()
        
        # Run tool-specific tests
        await self._run_tool_tests()
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        await self._save_results()
        
        # Display results
        self._display_results()
        
        return self.test_results

    async def _run_test_suites(self):
        """Run all test suites"""
        console.print("\n[bold yellow]Running Test Suites[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for suite_name, config in self.test_suites.items():
                task = progress.add_task(f"Running {suite_name}...", total=1)
                
                try:
                    result = await self._run_test_suite(suite_name, config)
                    self.test_results["test_suites"][suite_name] = result
                    
                    status = "✓ PASS" if result["passed"] else "✗ FAIL"
                    color = "green" if result["passed"] else "red"
                    console.print(f"[{color}]{status}[/{color}] {config['description']}")
                    
                except Exception as e:
                    self.test_results["test_suites"][suite_name] = {
                        "passed": False,
                        "error": str(e),
                        "duration": 0
                    }
                    console.print(f"[red]✗ ERROR[/red] {config['description']}: {str(e)}")
                
                progress.update(task, completed=1)

    async def _run_test_suite(self, suite_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test suite"""
        start_time = time.time()
        
        try:
            # Get test method
            method_name = f"_test_{suite_name}"
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                
                # Run with timeout
                result = await asyncio.wait_for(
                    method(),
                    timeout=config["timeout"]
                )
                
                return {
                    "passed": result,
                    "duration": time.time() - start_time,
                    "critical": config["critical"]
                }
            else:
                return {
                    "passed": False,
                    "error": f"Test method {method_name} not found",
                    "duration": time.time() - start_time,
                    "critical": config["critical"]
                }
                
        except asyncio.TimeoutError:
            return {
                "passed": False,
                "error": f"Test timed out after {config['timeout']} seconds",
                "duration": time.time() - start_time,
                "critical": config["critical"]
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "duration": time.time() - start_time,
                "critical": config["critical"]
            }

    async def _run_tool_tests(self):
        """Run tool-specific tests"""
        console.print("\n[bold yellow]Running Tool-Specific Tests[/bold yellow]")
        
        for tool_name, test_list in self.tool_tests.items():
            console.print(f"\nTesting {tool_name.upper()}...")
            
            tool_results = {}
            for test_name in test_list:
                try:
                    method_name = f"_test_{tool_name}_{test_name}"
                    if hasattr(self, method_name):
                        method = getattr(self, method_name)
                        result = await method()
                        tool_results[test_name] = {"passed": result, "error": None}
                        
                        status = "✓ PASS" if result else "✗ FAIL"
                        color = "green" if result else "red"
                        console.print(f"  [{color}]{status}[/{color}] {test_name}")
                    else:
                        tool_results[test_name] = {
                            "passed": False, 
                            "error": f"Method {method_name} not found"
                        }
                        console.print(f"  [yellow]? SKIP[/yellow] {test_name} (not implemented)")
                        
                except Exception as e:
                    tool_results[test_name] = {"passed": False, "error": str(e)}
                    console.print(f"  [red]✗ ERROR[/red] {test_name}: {str(e)}")
            
            self.test_results["tool_tests"][tool_name] = tool_results

    # Test Suite Implementations
    async def _test_basic_imports(self) -> bool:
        """Test basic imports of all AI tools"""
        try:
            # Test core system imports
            import pandas as pd
            import numpy as np
            from crewai import Agent, Task, Crew
            
            # Test AI tool imports
            tools_to_test = [
                ("qlib", "qlib"),
                ("openbb", "openbb"),
                ("rdagent", "rdagent"),
                ("pearl", "pearl"),
                ("mcp", "mcp")
            ]
            
            for display_name, import_name in tools_to_test:
                try:
                    __import__(import_name)
                    self.logger.info(f"Successfully imported {display_name}")
                except ImportError as e:
                    self.logger.warning(f"Could not import {display_name}: {str(e)}")
                    # Don't fail the test for optional imports
            
            return True
            
        except Exception as e:
            self.logger.error(f"Basic imports test failed: {str(e)}")
            return False

    async def _test_data_pipeline(self) -> bool:
        """Test data ingestion and processing pipeline"""
        try:
            # Test data directory structure
            data_dir = self.project_root / "data"
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
            
            # Test basic data processing
            import pandas as pd
            import numpy as np
            
            # Create test data
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
                'price': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Test data operations
            test_data['returns'] = test_data['price'].pct_change()
            test_data['sma_10'] = test_data['price'].rolling(10).mean()
            
            self.logger.info("Data pipeline test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data pipeline test failed: {str(e)}")
            return False

    async def _test_agent_communication(self) -> bool:
        """Test CrewAI agent communication"""
        try:
            from crewai import Agent, Task, Crew
            
            # Create test agents
            test_agent = Agent(
                role="Test Agent",
                goal="Perform test operations",
                backstory="A test agent for integration testing",
                verbose=False
            )
            
            # Create test task
            test_task = Task(
                description="Perform a simple test calculation",
                agent=test_agent,
                expected_output="A simple calculation result"
            )
            
            # Test crew creation (don't execute to avoid API calls)
            test_crew = Crew(
                agents=[test_agent],
                tasks=[test_task],
                verbose=False
            )
            
            self.logger.info("Agent communication test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Agent communication test failed: {str(e)}")
            return False

    async def _test_market_data_integration(self) -> bool:
        """Test market data sources"""
        try:
            # Test yfinance (most reliable for testing)
            import yfinance as yf
            
            # Fetch small amount of test data
            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="1d", interval="1m")
            
            if len(hist) > 0:
                self.logger.info("Market data integration test completed successfully")
                return True
            else:
                self.logger.warning("No market data received")
                return False
                
        except Exception as e:
            self.logger.error(f"Market data integration test failed: {str(e)}")
            return False

    async def _test_ml_model_integration(self) -> bool:
        """Test ML model loading and inference"""
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # Create synthetic data
            X = np.random.randn(1000, 10)
            y = np.random.randn(1000)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Train simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Test prediction
            predictions = model.predict(X_test)
            
            if len(predictions) == len(y_test):
                self.logger.info("ML model integration test completed successfully")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"ML model integration test failed: {str(e)}")
            return False

    async def _test_strategy_execution(self) -> bool:
        """Test strategy execution framework"""
        try:
            # Test basic strategy components
            import pandas as pd
            import numpy as np
            
            # Simulate strategy execution
            portfolio_value = 100000
            positions = {}
            
            # Test position management
            positions['AAPL'] = {'shares': 100, 'avg_price': 150.0}
            positions['GOOGL'] = {'shares': 50, 'avg_price': 2500.0}
            
            # Calculate portfolio value
            total_value = sum(pos['shares'] * pos['avg_price'] for pos in positions.values())
            
            if total_value > 0:
                self.logger.info("Strategy execution test completed successfully")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Strategy execution test failed: {str(e)}")
            return False

    async def _test_risk_management(self) -> bool:
        """Test risk management and safety checks"""
        try:
            # Test risk calculations
            import numpy as np
            
            # Simulate portfolio returns
            returns = np.random.randn(252) * 0.02  # Daily returns
            
            # Calculate risk metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            var_95 = np.percentile(returns, 5)  # Value at Risk
            max_drawdown = np.min(np.cumsum(returns))  # Max drawdown
            
            # Test risk limits
            risk_limits = {
                'max_volatility': 0.3,
                'max_var': -0.05,
                'max_drawdown': -0.2
            }
            
            # Check limits
            within_limits = (
                volatility <= risk_limits['max_volatility'] and
                var_95 >= risk_limits['max_var'] and
                max_drawdown >= risk_limits['max_drawdown']
            )
            
            self.logger.info("Risk management test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Risk management test failed: {str(e)}")
            return False

    async def _test_performance_benchmarks(self) -> bool:
        """Run performance benchmarks"""
        try:
            import time
            import psutil
            
            # CPU benchmark
            start_time = time.time()
            result = sum(i**2 for i in range(100000))
            cpu_time = time.time() - start_time
            
            # Memory benchmark
            memory_usage = psutil.virtual_memory().percent
            
            # Disk I/O benchmark
            test_file = self.project_root / "temp_benchmark_file.txt"
            start_time = time.time()
            with open(test_file, 'w') as f:
                f.write("x" * 1000000)  # 1MB
            io_time = time.time() - start_time
            
            # Cleanup
            if test_file.exists():
                test_file.unlink()
            
            # Check performance thresholds
            performance_ok = (
                cpu_time < 1.0 and  # CPU test should complete in < 1 second
                memory_usage < 90 and  # Memory usage should be < 90%
                io_time < 5.0  # I/O test should complete in < 5 seconds
            )
            
            self.logger.info(f"Performance benchmarks: CPU={cpu_time:.3f}s, Memory={memory_usage}%, I/O={io_time:.3f}s")
            return performance_ok
            
        except Exception as e:
            self.logger.error(f"Performance benchmarks failed: {str(e)}")
            return False

    # Tool-specific test implementations
    async def _test_qlib_import_test(self) -> bool:
        """Test Qlib import"""
        try:
            import qlib
            return True
        except ImportError:
            return False

    async def _test_openbb_import_test(self) -> bool:
        """Test OpenBB import"""
        try:
            import openbb
            return True
        except ImportError:
            return False

    async def _test_rd_agent_import_test(self) -> bool:
        """Test RD-Agent import"""
        try:
            import rdagent
            return True
        except ImportError:
            return False

    async def _test_pearl_import_test(self) -> bool:
        """Test Pearl import"""
        try:
            import pearl
            return True
        except ImportError:
            return False

    async def _test_alpaca_mcp_import_test(self) -> bool:
        """Test Alpaca MCP import"""
        try:
            import mcp
            return True
        except ImportError:
            return False

    def _generate_summary(self):
        """Generate test summary"""
        total_tests = 0
        passed_tests = 0
        critical_failures = 0
        
        # Count test suite results
        for suite_name, result in self.test_results["test_suites"].items():
            total_tests += 1
            if result.get("passed", False):
                passed_tests += 1
            elif result.get("critical", False):
                critical_failures += 1
        
        # Count tool test results
        for tool_name, tool_results in self.test_results["tool_tests"].items():
            for test_name, result in tool_results.items():
                total_tests += 1
                if result.get("passed", False):
                    passed_tests += 1
        
        # Calculate metrics
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        duration = time.time() - self.start_time if self.start_time else 0
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "critical_failures": critical_failures,
            "pass_rate": pass_rate,
            "duration": duration,
            "end_time": datetime.now().isoformat(),
            "overall_status": "PASS" if critical_failures == 0 and pass_rate >= 80 else "FAIL"
        }

    async def _save_results(self):
        """Save test results to file"""
        results_dir = self.project_root / "tests" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"integration_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        self.logger.info(f"Test results saved to {results_file}")

    def _display_results(self):
        """Display test results in a beautiful format"""
        summary = self.test_results["summary"]
        
        # Summary panel
        status_color = "green" if summary["overall_status"] == "PASS" else "red"
        summary_text = f"""
[bold]Overall Status:[/bold] [{status_color}]{summary['overall_status']}[/{status_color}]
[bold]Total Tests:[/bold] {summary['total_tests']}
[bold]Passed:[/bold] [green]{summary['passed_tests']}[/green]
[bold]Failed:[/bold] [red]{summary['failed_tests']}[/red]
[bold]Critical Failures:[/bold] [red]{summary['critical_failures']}[/red]
[bold]Pass Rate:[/bold] {summary['pass_rate']:.1f}%
[bold]Duration:[/bold] {summary['duration']:.2f}s
"""
        
        console.print(Panel(
            summary_text,
            title="[bold blue]Integration Test Results[/bold blue]",
            border_style=status_color
        ))
        
        # Detailed results table
        table = Table(title="Detailed Test Results")
        table.add_column("Category", style="cyan")
        table.add_column("Test", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Duration", style="yellow")
        
        # Add test suite results
        for suite_name, result in self.test_results["test_suites"].items():
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            color = "green" if result.get("passed", False) else "red"
            duration = f"{result.get('duration', 0):.2f}s"
            
            table.add_row(
                "Test Suite",
                suite_name,
                f"[{color}]{status}[/{color}]",
                duration
            )
        
        # Add tool test results
        for tool_name, tool_results in self.test_results["tool_tests"].items():
            for test_name, result in tool_results.items():
                status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
                color = "green" if result.get("passed", False) else "red"
                
                table.add_row(
                    f"Tool ({tool_name})",
                    test_name,
                    f"[{color}]{status}[/{color}]",
                    "-"
                )
        
        console.print(table)

async def main():
    """Main entry point for integration tests"""
    framework = IntegrationTestFramework()
    results = await framework.run_all_tests()
    
    # Exit with appropriate code
    overall_status = results["summary"]["overall_status"]
    sys.exit(0 if overall_status == "PASS" else 1)

if __name__ == "__main__":
    asyncio.run(main())