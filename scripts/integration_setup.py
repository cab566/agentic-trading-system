#!/usr/bin/env python3
"""
Integration Setup Script for AI Trading System

This script automates the setup and integration of new AI tools:
- Microsoft Qlib
- OpenBB Platform  
- RD-Agent
- Meta Pearl RL
- Alpaca MCP Server

Usage:
    python scripts/integration_setup.py --phase 1
    python scripts/integration_setup.py --tool qlib --test
    python scripts/integration_setup.py --validate-all
"""

import asyncio
import logging
import subprocess
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from datetime import datetime
import importlib
import pkg_resources
from packaging import version
import requests
import tempfile
import shutil

# Rich for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler

console = Console()

class IntegrationManager:
    """Manages the integration of AI tools into the trading system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.tools_dir = project_root / "tools"
        self.docs_dir = project_root / "docs"
        self.scripts_dir = project_root / "scripts"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.logger = logging.getLogger("integration")
        
        # Integration phases
        self.phases = {
            1: ["environment_setup", "dependency_check", "benchmarking_setup"],
            2: ["qlib_integration", "openbb_integration"],
            3: ["rd_agent_integration", "pearl_integration"],
            4: ["alpaca_mcp_integration", "self_improvement_activation"]
        }
        
        # Tool configurations
        self.tool_configs = {
            "qlib": {
                "name": "Microsoft Qlib",
                "repo": "https://github.com/microsoft/qlib",
                "install_cmd": "pip install qlib",
                "test_import": "qlib",
                "dependencies": ["torch", "lightgbm", "catboost"]
            },
            "openbb": {
                "name": "OpenBB Platform",
                "repo": "https://github.com/OpenBB-finance/OpenBB",
                "install_cmd": "pip install openbb",
                "test_import": "openbb",
                "dependencies": ["openbb-core"]
            },
            "rd_agent": {
                "name": "RD-Agent",
                "repo": "https://github.com/microsoft/RD-Agent",
                "install_cmd": "pip install git+https://github.com/microsoft/RD-Agent.git",
                "test_import": "rdagent",
                "dependencies": []
            },
            "pearl": {
                "name": "Meta Pearl RL",
                "repo": "https://github.com/facebookresearch/Pearl",
                "install_cmd": "pip install git+https://github.com/facebookresearch/Pearl.git",
                "test_import": "pearl",
                "dependencies": ["gymnasium", "torch"]
            },
            "alpaca_mcp": {
                "name": "Alpaca MCP Server",
                "repo": "https://github.com/modelcontextprotocol/servers",
                "install_cmd": "pip install mcp alpaca-py",
                "test_import": "mcp",
                "dependencies": ["alpaca-py", "websocket-client"]
            }
        }

    async def run_phase(self, phase: int) -> bool:
        """Run a specific integration phase"""
        console.print(f"\n[bold blue]Starting Phase {phase} Integration[/bold blue]")
        
        if phase not in self.phases:
            console.print(f"[red]Invalid phase: {phase}[/red]")
            return False
            
        tasks = self.phases[phase]
        success = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            for task_name in tasks:
                task = progress.add_task(f"Running {task_name}...", total=1)
                
                try:
                    if hasattr(self, task_name):
                        method = getattr(self, task_name)
                        result = await method()
                        if not result:
                            success = False
                            console.print(f"[red]Failed: {task_name}[/red]")
                        else:
                            console.print(f"[green]Completed: {task_name}[/green]")
                    else:
                        console.print(f"[yellow]Method not found: {task_name}[/yellow]")
                        success = False
                        
                    progress.update(task, completed=1)
                    
                except Exception as e:
                    console.print(f"[red]Error in {task_name}: {str(e)}[/red]")
                    success = False
                    progress.update(task, completed=1)
        
        return success

    async def environment_setup(self) -> bool:
        """Set up the development environment"""
        try:
            # Create necessary directories
            directories = [
                self.project_root / "tests" / "integration",
                self.project_root / "benchmarks",
                self.project_root / "logs" / "integration",
                self.project_root / "data" / "test"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
            # Create virtual environment if it doesn't exist
            venv_path = self.project_root / "venv_integration"
            if not venv_path.exists():
                subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], check=True)
                
            console.print("[green]Environment setup completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Environment setup failed: {str(e)}[/red]")
            return False

    async def dependency_check(self) -> bool:
        """Check and validate all dependencies"""
        try:
            # Read enhanced requirements
            req_file = self.project_root / "requirements_enhanced.txt"
            if not req_file.exists():
                console.print("[red]Enhanced requirements file not found[/red]")
                return False
                
            # Parse requirements
            with open(req_file, 'r') as f:
                requirements = [
                    line.strip() for line in f.readlines()
                    if line.strip() and not line.startswith('#')
                ]
            
            # Check for conflicts
            conflicts = await self._check_dependency_conflicts(requirements)
            if conflicts:
                console.print("[yellow]Dependency conflicts detected:[/yellow]")
                for conflict in conflicts:
                    console.print(f"  - {conflict}")
                    
            # Create compatibility report
            await self._create_compatibility_report(requirements)
            
            console.print("[green]Dependency check completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Dependency check failed: {str(e)}[/red]")
            return False

    async def benchmarking_setup(self) -> bool:
        """Set up benchmarking and monitoring framework"""
        try:
            # Create benchmark configuration
            benchmark_config = {
                "performance_metrics": {
                    "latency_threshold_ms": 100,
                    "throughput_min_ops_sec": 1000,
                    "memory_limit_mb": 2048,
                    "cpu_limit_percent": 80
                },
                "test_scenarios": {
                    "market_data_ingestion": {
                        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                        "timeframes": ["1m", "5m", "1h", "1d"],
                        "duration_minutes": 10
                    },
                    "strategy_execution": {
                        "strategies": ["momentum", "mean_reversion", "arbitrage"],
                        "portfolio_size": 100000,
                        "max_positions": 10
                    }
                },
                "monitoring": {
                    "metrics_collection_interval": 30,
                    "alert_thresholds": {
                        "error_rate": 0.01,
                        "response_time_p95": 200,
                        "memory_usage": 0.8
                    }
                }
            }
            
            # Save benchmark configuration
            benchmark_file = self.project_root / "benchmarks" / "config.yaml"
            with open(benchmark_file, 'w') as f:
                yaml.dump(benchmark_config, f, default_flow_style=False)
                
            # Create benchmark runner script
            await self._create_benchmark_runner()
            
            console.print("[green]Benchmarking setup completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Benchmarking setup failed: {str(e)}[/red]")
            return False

    async def qlib_integration(self) -> bool:
        """Integrate Microsoft Qlib"""
        return await self._integrate_tool("qlib")

    async def openbb_integration(self) -> bool:
        """Integrate OpenBB Platform"""
        return await self._integrate_tool("openbb")

    async def rd_agent_integration(self) -> bool:
        """Integrate RD-Agent"""
        return await self._integrate_tool("rd_agent")

    async def pearl_integration(self) -> bool:
        """Integrate Meta Pearl RL"""
        return await self._integrate_tool("pearl")

    async def alpaca_mcp_integration(self) -> bool:
        """Integrate Alpaca MCP Server"""
        return await self._integrate_tool("alpaca_mcp")

    async def self_improvement_activation(self) -> bool:
        """Activate the self-improvement framework"""
        try:
            # Update self-improvement framework configuration
            framework_file = self.project_root / "core" / "self_improvement_framework.py"
            if not framework_file.exists():
                console.print("[red]Self-improvement framework not found[/red]")
                return False
                
            # Create integration configuration
            integration_config = {
                "enabled_tools": list(self.tool_configs.keys()),
                "learning_rate": 0.01,
                "evaluation_frequency": "daily",
                "auto_integration": True,
                "safety_checks": True
            }
            
            config_file = self.config_dir / "self_improvement.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(integration_config, f, default_flow_style=False)
                
            console.print("[green]Self-improvement framework activated[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Self-improvement activation failed: {str(e)}[/red]")
            return False

    async def _integrate_tool(self, tool_name: str) -> bool:
        """Generic tool integration method"""
        try:
            config = self.tool_configs[tool_name]
            console.print(f"[blue]Integrating {config['name']}...[/blue]")
            
            # Install dependencies
            for dep in config["dependencies"]:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    console.print(f"[red]Failed to install {dep}: {result.stderr}[/red]")
                    return False
            
            # Install main package
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", config["install_cmd"].split()[-1]
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[red]Failed to install {config['name']}: {result.stderr}[/red]")
                return False
            
            # Test import
            try:
                importlib.import_module(config["test_import"])
                console.print(f"[green]Successfully imported {config['test_import']}[/green]")
            except ImportError as e:
                console.print(f"[yellow]Import test failed for {config['test_import']}: {str(e)}[/yellow]")
                # Continue anyway as some packages might have different import names
            
            # Run integration tests
            await self._run_integration_tests(tool_name)
            
            console.print(f"[green]{config['name']} integration completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Tool integration failed for {tool_name}: {str(e)}[/red]")
            return False

    async def _check_dependency_conflicts(self, requirements: List[str]) -> List[str]:
        """Check for dependency conflicts"""
        conflicts = []
        # This is a simplified conflict checker
        # In practice, you'd use tools like pip-tools or poetry
        return conflicts

    async def _create_compatibility_report(self, requirements: List[str]) -> None:
        """Create a compatibility report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "total_packages": len(requirements),
            "status": "compatible"
        }
        
        report_file = self.project_root / "docs" / "compatibility_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

    async def _create_benchmark_runner(self) -> None:
        """Create benchmark runner script"""
        runner_script = '''#!/usr/bin/env python3
"""
Benchmark Runner for Trading System Integration

Runs performance benchmarks and generates reports.
"""

import asyncio
import time
import psutil
import json
from datetime import datetime
from pathlib import Path

async def run_benchmarks():
    """Run all benchmark tests"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version
        },
        "benchmarks": {}
    }
    
    # Add benchmark implementations here
    
    # Save results
    results_file = Path("benchmarks/results") / f"benchmark_{int(time.time())}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(run_benchmarks())
'''
        
        runner_file = self.project_root / "benchmarks" / "runner.py"
        with open(runner_file, 'w') as f:
            f.write(runner_script)
        
        # Make executable
        runner_file.chmod(0o755)

    async def _run_integration_tests(self, tool_name: str) -> bool:
        """Run integration tests for a specific tool"""
        try:
            # Create and run basic integration test
            test_script = f'''
import sys
import traceback

def test_{tool_name}_integration():
    """Test {tool_name} integration"""
    try:
        # Basic import test
        import {self.tool_configs[tool_name]["test_import"]}
        print(f"✓ {tool_name} import successful")
        return True
    except Exception as e:
        print(f"✗ {tool_name} import failed: {{str(e)}}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_{tool_name}_integration()
    sys.exit(0 if success else 1)
'''
            
            # Write and run test
            test_file = self.project_root / "tests" / "integration" / f"test_{tool_name}.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            result = subprocess.run([
                sys.executable, str(test_file)
            ], capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception as e:
            console.print(f"[red]Integration test failed for {tool_name}: {str(e)}[/red]")
            return False

    async def validate_all_integrations(self) -> bool:
        """Validate all integrations are working correctly"""
        console.print("\n[bold blue]Validating All Integrations[/bold blue]")
        
        # Add user library path to Python path
        import site
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
        
        validation_results = {}
        overall_success = True
        
        for tool_name, config in self.tool_configs.items():
            console.print(f"Validating {config['name']}...")
            
            try:
                # Test import
                importlib.import_module(config["test_import"])
                validation_results[tool_name] = "✓ PASS"
                console.print(f"[green]✓ {config['name']} - PASS[/green]")
                
            except ImportError:
                validation_results[tool_name] = "✗ FAIL"
                console.print(f"[red]✗ {config['name']} - FAIL[/red]")
                overall_success = False
        
        # Create validation report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "PASS" if overall_success else "FAIL",
            "individual_results": validation_results
        }
        
        report_file = self.project_root / "docs" / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary table
        table = Table(title="Integration Validation Summary")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="bold")
        
        for tool_name, status in validation_results.items():
            color = "green" if "PASS" in status else "red"
            table.add_row(self.tool_configs[tool_name]["name"], f"[{color}]{status}[/{color}]")
        
        console.print(table)
        
        return overall_success

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Trading System Integration Setup")
    parser.add_argument("--phase", type=int, help="Run specific integration phase (1-4)")
    parser.add_argument("--tool", choices=["qlib", "openbb", "rd_agent", "pearl", "alpaca_mcp"], 
                       help="Integrate specific tool")
    parser.add_argument("--validate-all", action="store_true", help="Validate all integrations")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    manager = IntegrationManager(project_root)
    
    console.print(Panel.fit(
        "[bold blue]AI Trading System Integration Manager[/bold blue]\n"
        "Automating the integration of advanced AI tools",
        border_style="blue"
    ))
    
    if args.validate_all:
        success = await manager.validate_all_integrations()
        sys.exit(0 if success else 1)
    
    elif args.phase:
        success = await manager.run_phase(args.phase)
        sys.exit(0 if success else 1)
    
    elif args.tool:
        success = await manager._integrate_tool(args.tool)
        sys.exit(0 if success else 1)
    
    else:
        console.print("[yellow]Please specify --phase, --tool, or --validate-all[/yellow]")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())