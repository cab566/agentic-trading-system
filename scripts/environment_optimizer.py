#!/usr/bin/env python3
"""
Development Environment Optimizer

Prepares and optimizes the development environment for:
- AI tool integrations
- Performance optimization
- Resource management
- Development workflow enhancement
"""

import os
import sys
import subprocess
import json
import yaml
import shutil
import platform
import psutil
import docker
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from datetime import datetime

# Rich for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.tree import Tree
from rich.text import Text

console = Console()

@dataclass
class SystemRequirement:
    """System requirement specification"""
    name: str
    required_version: str
    current_version: Optional[str] = None
    installed: bool = False
    critical: bool = True
    install_command: Optional[str] = None

@dataclass
class OptimizationResult:
    """Optimization result"""
    component: str
    status: str  # 'success', 'warning', 'error', 'skipped'
    message: str
    details: Optional[Dict[str, Any]] = None

class EnvironmentOptimizer:
    """Main environment optimizer class"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.system_info = self._get_system_info()
        self.docker_client = None
        self.results: List[OptimizationResult] = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            console.print(f"[yellow]Docker not available: {e}[/yellow]")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "environment_optimizer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total,
            "disk_free": psutil.disk_usage('/').free
        }

    def display_system_info(self):
        """Display system information"""
        info_table = Table(title="System Information")
        info_table.add_column("Component", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Platform", self.system_info["platform"])
        info_table.add_row("System", self.system_info["system"])
        info_table.add_row("Architecture", self.system_info["machine"])
        info_table.add_row("Python Version", self.system_info["python_version"])
        info_table.add_row("CPU Cores", str(self.system_info["cpu_count"]))
        info_table.add_row("Total Memory", f"{self.system_info['memory_total'] / (1024**3):.1f} GB")
        info_table.add_row("Free Disk Space", f"{self.system_info['disk_free'] / (1024**3):.1f} GB")
        
        console.print(Panel(info_table, title="Environment Analysis", border_style="blue"))

    def check_system_requirements(self) -> List[SystemRequirement]:
        """Check system requirements for AI trading system"""
        requirements = [
            SystemRequirement("python", "3.9.0", critical=True),
            SystemRequirement("docker", "20.0.0", critical=True),
            SystemRequirement("docker-compose", "2.0.0", critical=True),
            SystemRequirement("git", "2.20.0", critical=True),
            SystemRequirement("node", "16.0.0", critical=False),
            SystemRequirement("npm", "8.0.0", critical=False),
            SystemRequirement("redis-cli", "6.0.0", critical=False),
            SystemRequirement("postgresql-client", "13.0", critical=False),
        ]
        
        console.print("[blue]Checking system requirements...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Checking requirements...", total=len(requirements))
            
            for req in requirements:
                progress.update(task, description=f"Checking {req.name}...")
                
                try:
                    if req.name == "python":
                        req.current_version = platform.python_version()
                        req.installed = True
                    elif req.name == "docker":
                        result = subprocess.run(["docker", "--version"], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            version_line = result.stdout.strip()
                            # Extract version from "Docker version 20.10.8, build 3967b7d"
                            version = version_line.split()[2].rstrip(',')
                            req.current_version = version
                            req.installed = True
                    elif req.name == "docker-compose":
                        result = subprocess.run(["docker-compose", "--version"], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            version_line = result.stdout.strip()
                            # Extract version from "docker-compose version 2.10.2, build a9339b"
                            version = version_line.split()[2].rstrip(',')
                            req.current_version = version
                            req.installed = True
                    else:
                        # Generic version check
                        result = subprocess.run([req.name, "--version"], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            req.installed = True
                            req.current_version = "installed"
                
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    req.installed = False
                
                progress.advance(task)
        
        return requirements

    def display_requirements_status(self, requirements: List[SystemRequirement]):
        """Display requirements check results"""
        req_table = Table(title="System Requirements Status")
        req_table.add_column("Component", style="cyan")
        req_table.add_column("Required", style="yellow")
        req_table.add_column("Current", style="green")
        req_table.add_column("Status", style="bold")
        req_table.add_column("Critical", style="red")
        
        for req in requirements:
            status = "[green]✓ OK[/green]" if req.installed else "[red]✗ Missing[/red]"
            critical = "[red]Yes[/red]" if req.critical else "[yellow]No[/yellow]"
            current = req.current_version or "[red]Not found[/red]"
            
            req_table.add_row(
                req.name,
                req.required_version,
                current,
                status,
                critical
            )
        
        console.print(Panel(req_table, title="Requirements Check", border_style="green"))
        
        # Show missing critical requirements
        missing_critical = [req for req in requirements if not req.installed and req.critical]
        if missing_critical:
            console.print(Panel(
                f"[red]Missing critical requirements: {', '.join(req.name for req in missing_critical)}[/red]\n"
                f"Please install these before proceeding.",
                title="Critical Issues",
                border_style="red"
            ))

    def optimize_python_environment(self) -> OptimizationResult:
        """Optimize Python environment"""
        try:
            console.print("[blue]Optimizing Python environment...[/blue]")
            
            # Check if virtual environment exists
            venv_path = self.project_root / "venv"
            if not venv_path.exists():
                console.print("[yellow]Creating virtual environment...[/yellow]")
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            
            # Determine activation script path
            if platform.system() == "Windows":
                activate_script = venv_path / "Scripts" / "activate"
                pip_path = venv_path / "Scripts" / "pip"
            else:
                activate_script = venv_path / "bin" / "activate"
                pip_path = venv_path / "bin" / "pip"
            
            # Upgrade pip
            console.print("[yellow]Upgrading pip...[/yellow]")
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            
            # Install wheel and setuptools
            subprocess.run([str(pip_path), "install", "--upgrade", "wheel", "setuptools"], check=True)
            
            # Install requirements if they exist
            requirements_files = [
                "requirements.txt",
                "requirements_enhanced.txt",
                "requirements-dev.txt"
            ]
            
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    console.print(f"[yellow]Installing {req_file}...[/yellow]")
                    subprocess.run([str(pip_path), "install", "-r", str(req_path)], check=True)
            
            return OptimizationResult(
                component="python_environment",
                status="success",
                message="Python environment optimized successfully",
                details={
                    "venv_path": str(venv_path),
                    "pip_path": str(pip_path)
                }
            )
            
        except subprocess.CalledProcessError as e:
            return OptimizationResult(
                component="python_environment",
                status="error",
                message=f"Failed to optimize Python environment: {e}",
                details={"error": str(e)}
            )

    def optimize_docker_environment(self) -> OptimizationResult:
        """Optimize Docker environment"""
        try:
            if not self.docker_client:
                return OptimizationResult(
                    component="docker_environment",
                    status="skipped",
                    message="Docker not available"
                )
            
            console.print("[blue]Optimizing Docker environment...[/blue]")
            
            # Check Docker daemon
            try:
                self.docker_client.ping()
            except Exception as e:
                return OptimizationResult(
                    component="docker_environment",
                    status="error",
                    message=f"Docker daemon not accessible: {e}"
                )
            
            # Clean up unused resources
            console.print("[yellow]Cleaning up Docker resources...[/yellow]")
            
            # Remove unused containers
            containers_removed = 0
            for container in self.docker_client.containers.list(all=True):
                if container.status == 'exited':
                    container.remove()
                    containers_removed += 1
            
            # Remove unused images
            images_removed = len(self.docker_client.images.prune()['ImagesDeleted'] or [])
            
            # Remove unused volumes
            volumes_removed = len(self.docker_client.volumes.prune()['VolumesDeleted'] or [])
            
            # Remove unused networks
            networks_removed = len(self.docker_client.networks.prune()['NetworksDeleted'] or [])
            
            # Pull required base images
            console.print("[yellow]Pulling required base images...[/yellow]")
            base_images = [
                "python:3.11-slim",
                "postgres:15-alpine",
                "redis:7-alpine",
                "nginx:alpine"
            ]
            
            for image in base_images:
                try:
                    self.docker_client.images.pull(image)
                    console.print(f"[green]✓ Pulled {image}[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠ Failed to pull {image}: {e}[/yellow]")
            
            return OptimizationResult(
                component="docker_environment",
                status="success",
                message="Docker environment optimized successfully",
                details={
                    "containers_removed": containers_removed,
                    "images_removed": images_removed,
                    "volumes_removed": volumes_removed,
                    "networks_removed": networks_removed
                }
            )
            
        except Exception as e:
            return OptimizationResult(
                component="docker_environment",
                status="error",
                message=f"Failed to optimize Docker environment: {e}",
                details={"error": str(e)}
            )

    def optimize_project_structure(self) -> OptimizationResult:
        """Optimize project directory structure"""
        try:
            console.print("[blue]Optimizing project structure...[/blue]")
            
            # Required directories
            required_dirs = [
                "logs",
                "data",
                "data/raw",
                "data/processed",
                "data/models",
                "config",
                "tests",
                "tests/unit",
                "tests/integration",
                "docs",
                "scripts",
                "monitoring",
                "deployment",
                "tools/integrations",
                "core/agents",
                "core/strategies",
                "core/data",
                "core/risk",
                "web/static",
                "web/templates",
                "notebooks"
            ]
            
            created_dirs = []
            for dir_path in required_dirs:
                full_path = self.project_root / dir_path
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_path)
            
            # Create .gitignore if it doesn't exist
            gitignore_path = self.project_root / ".gitignore"
            if not gitignore_path.exists():
                gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
data/models/*
!data/models/.gitkeep

# Environment variables
.env
.env.local
.env.production

# Database
*.db
*.sqlite3

# Docker
.dockerignore

# OS
.DS_Store
Thumbs.db

# Jupyter Notebooks
.ipynb_checkpoints/

# Temporary files
*.tmp
*.temp
"""
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content.strip())
            
            # Create .gitkeep files for empty directories
            for dir_path in ["data/raw", "data/processed", "data/models", "logs"]:
                gitkeep_path = self.project_root / dir_path / ".gitkeep"
                if not gitkeep_path.exists():
                    gitkeep_path.touch()
            
            return OptimizationResult(
                component="project_structure",
                status="success",
                message="Project structure optimized successfully",
                details={
                    "created_directories": created_dirs,
                    "total_directories": len(required_dirs)
                }
            )
            
        except Exception as e:
            return OptimizationResult(
                component="project_structure",
                status="error",
                message=f"Failed to optimize project structure: {e}",
                details={"error": str(e)}
            )

    def optimize_development_tools(self) -> OptimizationResult:
        """Optimize development tools and configurations"""
        try:
            console.print("[blue]Optimizing development tools...[/blue]")
            
            optimizations = []
            
            # Create VS Code settings
            vscode_dir = self.project_root / ".vscode"
            if not vscode_dir.exists():
                vscode_dir.mkdir()
                
                # Settings
                settings = {
                    "python.defaultInterpreterPath": "./venv/bin/python",
                    "python.linting.enabled": True,
                    "python.linting.pylintEnabled": True,
                    "python.formatting.provider": "black",
                    "python.sortImports.args": ["--profile", "black"],
                    "editor.formatOnSave": True,
                    "editor.codeActionsOnSave": {
                        "source.organizeImports": True
                    },
                    "files.exclude": {
                        "**/__pycache__": True,
                        "**/*.pyc": True,
                        "**/venv": True,
                        "**/node_modules": True
                    }
                }
                
                with open(vscode_dir / "settings.json", 'w') as f:
                    json.dump(settings, f, indent=2)
                
                # Launch configuration
                launch_config = {
                    "version": "0.2.0",
                    "configurations": [
                        {
                            "name": "Python: Current File",
                            "type": "python",
                            "request": "launch",
                            "program": "${file}",
                            "console": "integratedTerminal",
                            "cwd": "${workspaceFolder}"
                        },
                        {
                            "name": "Trading System",
                            "type": "python",
                            "request": "launch",
                            "program": "${workspaceFolder}/main.py",
                            "console": "integratedTerminal",
                            "args": ["--mode", "development"]
                        }
                    ]
                }
                
                with open(vscode_dir / "launch.json", 'w') as f:
                    json.dump(launch_config, f, indent=2)
                
                optimizations.append("VS Code configuration")
            
            # Create pre-commit configuration
            precommit_config = self.project_root / ".pre-commit-config.yaml"
            if not precommit_config.exists():
                config_content = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      - id: debug-statements
      
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]
"""
                with open(precommit_config, 'w') as f:
                    f.write(config_content.strip())
                
                optimizations.append("Pre-commit hooks")
            
            # Create pytest configuration
            pytest_config = self.project_root / "pytest.ini"
            if not pytest_config.exists():
                config_content = """
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=core
    --cov=tools
    --cov-report=term-missing
    --cov-report=html:htmlcov
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    api: API tests
"""
                with open(pytest_config, 'w') as f:
                    f.write(config_content.strip())
                
                optimizations.append("Pytest configuration")
            
            return OptimizationResult(
                component="development_tools",
                status="success",
                message="Development tools optimized successfully",
                details={
                    "optimizations": optimizations,
                    "total_optimizations": len(optimizations)
                }
            )
            
        except Exception as e:
            return OptimizationResult(
                component="development_tools",
                status="error",
                message=f"Failed to optimize development tools: {e}",
                details={"error": str(e)}
            )

    def optimize_performance_settings(self) -> OptimizationResult:
        """Optimize system performance settings"""
        try:
            console.print("[blue]Optimizing performance settings...[/blue]")
            
            optimizations = []
            
            # Create performance configuration
            perf_config = {
                "system": {
                    "max_workers": min(32, (os.cpu_count() or 1) + 4),
                    "memory_limit_mb": int(psutil.virtual_memory().total * 0.8 / (1024**2)),
                    "disk_cache_size_mb": 1024,
                    "network_timeout": 30
                },
                "database": {
                    "connection_pool_size": 20,
                    "max_overflow": 30,
                    "pool_timeout": 30,
                    "pool_recycle": 3600
                },
                "redis": {
                    "max_connections": 50,
                    "socket_timeout": 5,
                    "socket_connect_timeout": 5
                },
                "ai_models": {
                    "batch_size": 32,
                    "max_sequence_length": 512,
                    "model_cache_size": 3,
                    "inference_timeout": 30
                }
            }
            
            config_path = self.project_root / "config" / "performance.yaml"
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(perf_config, f, default_flow_style=False)
            
            optimizations.append("Performance configuration")
            
            # Set environment variables for optimization
            env_vars = {
                "PYTHONUNBUFFERED": "1",
                "PYTHONOPTIMIZE": "1",
                "OMP_NUM_THREADS": str(os.cpu_count() or 1),
                "NUMEXPR_MAX_THREADS": str(os.cpu_count() or 1)
            }
            
            env_file = self.project_root / ".env.performance"
            with open(env_file, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            optimizations.append("Environment variables")
            
            return OptimizationResult(
                component="performance_settings",
                status="success",
                message="Performance settings optimized successfully",
                details={
                    "optimizations": optimizations,
                    "config_path": str(config_path),
                    "env_file": str(env_file)
                }
            )
            
        except Exception as e:
            return OptimizationResult(
                component="performance_settings",
                status="error",
                message=f"Failed to optimize performance settings: {e}",
                details={"error": str(e)}
            )

    async def run_optimization(self) -> List[OptimizationResult]:
        """Run complete environment optimization"""
        console.print(Panel(
            "[bold blue]AI Trading System Environment Optimizer[/bold blue]\n"
            "Preparing and optimizing development environment...",
            border_style="blue"
        ))
        
        # Display system information
        self.display_system_info()
        
        # Check system requirements
        requirements = self.check_system_requirements()
        self.display_requirements_status(requirements)
        
        # Check for critical missing requirements
        missing_critical = [req for req in requirements if not req.installed and req.critical]
        if missing_critical:
            console.print(Panel(
                "[red]Cannot proceed with missing critical requirements.[/red]\n"
                "Please install the missing components and run again.",
                title="Optimization Blocked",
                border_style="red"
            ))
            return []
        
        # Run optimizations
        optimizations = [
            ("Python Environment", self.optimize_python_environment),
            ("Docker Environment", self.optimize_docker_environment),
            ("Project Structure", self.optimize_project_structure),
            ("Development Tools", self.optimize_development_tools),
            ("Performance Settings", self.optimize_performance_settings)
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Running optimizations...", total=len(optimizations))
            
            for name, optimization_func in optimizations:
                progress.update(task, description=f"Optimizing {name}...")
                
                try:
                    result = optimization_func()
                    results.append(result)
                    
                    status_color = {
                        "success": "green",
                        "warning": "yellow",
                        "error": "red",
                        "skipped": "blue"
                    }.get(result.status, "white")
                    
                    console.print(f"[{status_color}]{result.status.upper()}[/{status_color}]: {result.message}")
                    
                except Exception as e:
                    error_result = OptimizationResult(
                        component=name.lower().replace(" ", "_"),
                        status="error",
                        message=f"Unexpected error: {e}",
                        details={"error": str(e)}
                    )
                    results.append(error_result)
                    console.print(f"[red]ERROR[/red]: {error_result.message}")
                
                progress.advance(task)
        
        self.results = results
        return results

    def display_optimization_summary(self):
        """Display optimization results summary"""
        if not self.results:
            return
        
        # Summary table
        summary_table = Table(title="Optimization Summary")
        summary_table.add_column("Component", style="cyan")
        summary_table.add_column("Status", style="bold")
        summary_table.add_column("Message", style="white")
        
        status_counts = {"success": 0, "warning": 0, "error": 0, "skipped": 0}
        
        for result in self.results:
            status_color = {
                "success": "green",
                "warning": "yellow", 
                "error": "red",
                "skipped": "blue"
            }.get(result.status, "white")
            
            summary_table.add_row(
                result.component.replace("_", " ").title(),
                f"[{status_color}]{result.status.upper()}[/{status_color}]",
                result.message
            )
            
            status_counts[result.status] += 1
        
        console.print(Panel(summary_table, title="Optimization Results", border_style="green"))
        
        # Status summary
        status_text = f"""
[green]✓ Successful: {status_counts['success']}[/green]
[yellow]⚠ Warnings: {status_counts['warning']}[/yellow]
[red]✗ Errors: {status_counts['error']}[/red]
[blue]→ Skipped: {status_counts['skipped']}[/blue]
        """
        
        console.print(Panel(status_text.strip(), title="Status Summary", border_style="blue"))
        
        # Next steps
        if status_counts['error'] == 0:
            next_steps = """
[green]Environment optimization completed successfully![/green]

Next steps:
1. Run integration tests: `python scripts/integration_setup.py --test`
2. Start monitoring: `python monitoring/comprehensive_monitoring.py --dashboard`
3. Begin AI tool integration: `python scripts/integration_setup.py --phase 1`
4. Review performance settings in `config/performance.yaml`
            """
        else:
            next_steps = """
[yellow]Environment optimization completed with errors.[/yellow]

Please address the errors above before proceeding:
1. Check the logs in `logs/environment_optimizer.log`
2. Fix any configuration issues
3. Re-run the optimizer: `python scripts/environment_optimizer.py`
            """
        
        console.print(Panel(next_steps.strip(), title="Next Steps", border_style="cyan"))

    def generate_optimization_report(self) -> str:
        """Generate detailed optimization report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "optimization_results": [
                {
                    "component": result.component,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details
                }
                for result in self.results
            ],
            "summary": {
                "total_optimizations": len(self.results),
                "successful": len([r for r in self.results if r.status == "success"]),
                "warnings": len([r for r in self.results if r.status == "warning"]),
                "errors": len([r for r in self.results if r.status == "error"]),
                "skipped": len([r for r in self.results if r.status == "skipped"])
            }
        }
        
        report_path = self.project_root / "logs" / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(report_path)

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Environment Optimizer for AI Trading System")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--check-only", action="store_true", help="Only check requirements, don't optimize")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    optimizer = EnvironmentOptimizer(project_root)
    
    if args.check_only:
        requirements = optimizer.check_system_requirements()
        optimizer.display_requirements_status(requirements)
        return
    
    # Run optimization
    results = await optimizer.run_optimization()
    
    # Display summary
    optimizer.display_optimization_summary()
    
    # Generate report if requested
    if args.report:
        report_path = optimizer.generate_optimization_report()
        console.print(f"[green]Detailed report saved to: {report_path}[/green]")

if __name__ == "__main__":
    asyncio.run(main())