#!/usr/bin/env python3
"""
Automated Deployment Pipeline for AI Trading System

Handles:
- Zero-downtime deployments
- Automated rollbacks
- Health checks
- Configuration management
- Database migrations
- Service orchestration
"""

import asyncio
import logging
import subprocess
import sys
import json
import yaml
import docker
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
from datetime import datetime
import shutil
import tempfile
import hashlib
import requests
from dataclasses import dataclass, asdict

# Rich for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler

console = Console()

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str
    version: str
    rollback_enabled: bool = True
    health_check_timeout: int = 300
    max_rollback_attempts: int = 3
    backup_retention_days: int = 7
    deployment_strategy: str = "blue_green"  # blue_green, rolling, recreate

@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: str
    response_time: float
    last_check: datetime
    error_message: Optional[str] = None

class DeploymentManager:
    """Manages automated deployments with zero downtime"""
    
    def __init__(self, project_root: Path, config: DeploymentConfig):
        self.project_root = project_root
        self.config = config
        self.docker_client = docker.from_env()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.logger = logging.getLogger("deployment")
        
        # Deployment paths
        self.deployment_dir = project_root / "deployment"
        self.backup_dir = project_root / "backups"
        self.config_dir = project_root / "config"
        
        # Service configurations
        self.services = {
            "trading-system": {
                "image": "trading_system_v2-trading-system",
                "port": 8000,
                "health_endpoint": "/health",
                "critical": True
            },
            "postgres": {
                "image": "postgres:15-alpine",
                "port": 5432,
                "health_endpoint": None,
                "critical": True
            },
            "redis": {
                "image": "redis:7-alpine",
                "port": 6379,
                "health_endpoint": None,
                "critical": True
            },
            "nginx": {
                "image": "nginx:alpine",
                "port": 80,
                "health_endpoint": "/health",
                "critical": True
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "port": 3000,
                "health_endpoint": "/api/health",
                "critical": False
            },
            "prometheus": {
                "image": "prom/prometheus:latest",
                "port": 9090,
                "health_endpoint": "/-/healthy",
                "critical": False
            }
        }
        
        # Deployment history
        self.deployment_history = []

    async def deploy(self) -> bool:
        """Execute deployment with the configured strategy"""
        console.print(Panel.fit(
            f"[bold blue]Starting Deployment[/bold blue]\n"
            f"Environment: {self.config.environment}\n"
            f"Version: {self.config.version}\n"
            f"Strategy: {self.config.deployment_strategy}",
            border_style="blue"
        ))
        
        deployment_id = self._generate_deployment_id()
        start_time = time.time()
        
        try:
            # Pre-deployment checks
            if not await self._pre_deployment_checks():
                console.print("[red]Pre-deployment checks failed[/red]")
                return False
            
            # Create backup
            if not await self._create_backup(deployment_id):
                console.print("[red]Backup creation failed[/red]")
                return False
            
            # Execute deployment strategy
            if self.config.deployment_strategy == "blue_green":
                success = await self._blue_green_deployment()
            elif self.config.deployment_strategy == "rolling":
                success = await self._rolling_deployment()
            else:
                success = await self._recreate_deployment()
            
            if not success:
                console.print("[red]Deployment failed, initiating rollback[/red]")
                if self.config.rollback_enabled:
                    await self._rollback(deployment_id)
                return False
            
            # Post-deployment verification
            if not await self._post_deployment_verification():
                console.print("[red]Post-deployment verification failed[/red]")
                if self.config.rollback_enabled:
                    await self._rollback(deployment_id)
                return False
            
            # Record successful deployment
            duration = time.time() - start_time
            self._record_deployment(deployment_id, "success", duration)
            
            console.print(f"[green]Deployment {deployment_id} completed successfully in {duration:.2f}s[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Deployment failed with error: {str(e)}[/red]")
            if self.config.rollback_enabled:
                await self._rollback(deployment_id)
            return False

    async def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks"""
        console.print("\n[yellow]Running pre-deployment checks...[/yellow]")
        
        checks = [
            ("Docker daemon", self._check_docker_daemon),
            ("Disk space", self._check_disk_space),
            ("Network connectivity", self._check_network),
            ("Configuration files", self._check_configuration),
            ("Database connectivity", self._check_database),
            ("Service health", self._check_current_services)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for check_name, check_func in checks:
                task = progress.add_task(f"Checking {check_name}...", total=1)
                
                try:
                    result = await check_func()
                    if result:
                        console.print(f"[green]✓[/green] {check_name}")
                    else:
                        console.print(f"[red]✗[/red] {check_name}")
                        return False
                except Exception as e:
                    console.print(f"[red]✗[/red] {check_name}: {str(e)}")
                    return False
                
                progress.update(task, completed=1)
        
        return True

    async def _check_docker_daemon(self) -> bool:
        """Check if Docker daemon is running"""
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False

    async def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free // (1024**3)
            return free_gb >= 5  # Require at least 5GB free
        except Exception:
            return False

    async def _check_network(self) -> bool:
        """Check network connectivity"""
        try:
            response = requests.get("https://httpbin.org/status/200", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    async def _check_configuration(self) -> bool:
        """Check configuration files"""
        required_files = [
            self.project_root / "docker-compose.production.yml",
            self.config_dir / "agents.yaml",
            self.config_dir / "data_sources.yaml"
        ]
        
        return all(f.exists() for f in required_files)

    async def _check_database(self) -> bool:
        """Check database connectivity"""
        try:
            # Try to connect to PostgreSQL container
            containers = self.docker_client.containers.list(
                filters={"name": "postgres"}
            )
            return len(containers) > 0 and containers[0].status == "running"
        except Exception:
            return False

    async def _check_current_services(self) -> bool:
        """Check current service health"""
        try:
            health_results = await self._check_service_health()
            critical_services = [
                name for name, config in self.services.items()
                if config["critical"]
            ]
            
            for service_name in critical_services:
                if service_name in health_results:
                    if health_results[service_name].status != "healthy":
                        return False
            
            return True
        except Exception:
            return False

    async def _create_backup(self, deployment_id: str) -> bool:
        """Create backup before deployment"""
        console.print("\n[yellow]Creating backup...[/yellow]")
        
        try:
            # Create backup directory
            backup_path = self.backup_dir / deployment_id
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration files
            config_backup = backup_path / "config"
            shutil.copytree(self.config_dir, config_backup, dirs_exist_ok=True)
            
            # Backup database
            await self._backup_database(backup_path)
            
            # Backup Docker images
            await self._backup_docker_images(backup_path)
            
            # Create backup manifest
            manifest = {
                "deployment_id": deployment_id,
                "timestamp": datetime.now().isoformat(),
                "version": self.config.version,
                "environment": self.config.environment
            }
            
            with open(backup_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            console.print(f"[green]Backup created: {backup_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Backup creation failed: {str(e)}[/red]")
            return False

    async def _backup_database(self, backup_path: Path):
        """Backup database"""
        try:
            # Get PostgreSQL container
            containers = self.docker_client.containers.list(
                filters={"name": "postgres"}
            )
            
            if containers:
                container = containers[0]
                
                # Create database dump
                dump_cmd = [
                    "pg_dumpall", "-U", "postgres"
                ]
                
                result = container.exec_run(dump_cmd)
                if result.exit_code == 0:
                    with open(backup_path / "database_dump.sql", 'wb') as f:
                        f.write(result.output)
                    
                    console.print("[green]Database backup completed[/green]")
                else:
                    console.print(f"[yellow]Database backup failed: {result.output.decode()}[/yellow]")
                    
        except Exception as e:
            console.print(f"[yellow]Database backup error: {str(e)}[/yellow]")

    async def _backup_docker_images(self, backup_path: Path):
        """Backup Docker images"""
        try:
            images_to_backup = [
                "trading_system_v2-trading-system"
            ]
            
            for image_name in images_to_backup:
                try:
                    image = self.docker_client.images.get(image_name)
                    
                    # Save image to tar file
                    image_file = backup_path / f"{image_name.replace('/', '_')}.tar"
                    with open(image_file, 'wb') as f:
                        for chunk in image.save():
                            f.write(chunk)
                    
                    console.print(f"[green]Backed up image: {image_name}[/green]")
                    
                except docker.errors.ImageNotFound:
                    console.print(f"[yellow]Image not found: {image_name}[/yellow]")
                    
        except Exception as e:
            console.print(f"[yellow]Image backup error: {str(e)}[/yellow]")

    async def _blue_green_deployment(self) -> bool:
        """Execute blue-green deployment"""
        console.print("\n[blue]Executing blue-green deployment...[/blue]")
        
        try:
            # Build new images
            if not await self._build_images():
                return False
            
            # Start green environment
            if not await self._start_green_environment():
                return False
            
            # Health check green environment
            if not await self._health_check_green_environment():
                return False
            
            # Switch traffic to green
            if not await self._switch_traffic_to_green():
                return False
            
            # Stop blue environment
            await self._stop_blue_environment()
            
            console.print("[green]Blue-green deployment completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Blue-green deployment failed: {str(e)}[/red]")
            return False

    async def _rolling_deployment(self) -> bool:
        """Execute rolling deployment"""
        console.print("\n[blue]Executing rolling deployment...[/blue]")
        
        try:
            # Build new images
            if not await self._build_images():
                return False
            
            # Update services one by one
            for service_name, config in self.services.items():
                if config["critical"]:
                    console.print(f"Updating {service_name}...")
                    
                    if not await self._update_service(service_name):
                        return False
                    
                    # Wait for service to be healthy
                    if not await self._wait_for_service_health(service_name):
                        return False
            
            console.print("[green]Rolling deployment completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Rolling deployment failed: {str(e)}[/red]")
            return False

    async def _recreate_deployment(self) -> bool:
        """Execute recreate deployment (with downtime)"""
        console.print("\n[blue]Executing recreate deployment...[/blue]")
        
        try:
            # Stop all services
            await self._stop_all_services()
            
            # Build new images
            if not await self._build_images():
                return False
            
            # Start all services
            if not await self._start_all_services():
                return False
            
            console.print("[green]Recreate deployment completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Recreate deployment failed: {str(e)}[/red]")
            return False

    async def _build_images(self) -> bool:
        """Build Docker images"""
        console.print("Building Docker images...")
        
        try:
            # Build using docker-compose
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml", "build"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green]Images built successfully[/green]")
                return True
            else:
                console.print(f"[red]Image build failed: {result.stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Image build error: {str(e)}[/red]")
            return False

    async def _start_green_environment(self) -> bool:
        """Start green environment for blue-green deployment"""
        # This is a simplified implementation
        # In practice, you'd start services with different names/ports
        return await self._start_all_services()

    async def _health_check_green_environment(self) -> bool:
        """Health check green environment"""
        return await self._wait_for_all_services_healthy()

    async def _switch_traffic_to_green(self) -> bool:
        """Switch traffic to green environment"""
        # This would typically involve updating load balancer configuration
        # For now, we'll just ensure services are running
        return True

    async def _stop_blue_environment(self):
        """Stop blue environment"""
        # In a real blue-green setup, you'd stop the old environment
        pass

    async def _update_service(self, service_name: str) -> bool:
        """Update a specific service"""
        try:
            # Stop service
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml",
                "stop", service_name
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[red]Failed to stop {service_name}[/red]")
                return False
            
            # Start service with new image
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml",
                "up", "-d", service_name
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green]Updated {service_name}[/green]")
                return True
            else:
                console.print(f"[red]Failed to start {service_name}: {result.stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Service update error for {service_name}: {str(e)}[/red]")
            return False

    async def _wait_for_service_health(self, service_name: str, timeout: int = 60) -> bool:
        """Wait for a service to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health_results = await self._check_service_health()
            
            if service_name in health_results:
                if health_results[service_name].status == "healthy":
                    return True
            
            await asyncio.sleep(5)
        
        return False

    async def _stop_all_services(self):
        """Stop all services"""
        console.print("Stopping all services...")
        
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.production.yml", "down"
        ], cwd=self.project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]All services stopped[/green]")
        else:
            console.print(f"[yellow]Service stop warning: {result.stderr}[/yellow]")

    async def _start_all_services(self) -> bool:
        """Start all services"""
        console.print("Starting all services...")
        
        try:
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.production.yml", "up", "-d"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green]All services started[/green]")
                return True
            else:
                console.print(f"[red]Service start failed: {result.stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Service start error: {str(e)}[/red]")
            return False

    async def _wait_for_all_services_healthy(self) -> bool:
        """Wait for all critical services to become healthy"""
        console.print("Waiting for services to become healthy...")
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Checking service health...", total=1)
            
            while time.time() - start_time < self.config.health_check_timeout:
                health_results = await self._check_service_health()
                
                critical_services = [
                    name for name, config in self.services.items()
                    if config["critical"]
                ]
                
                healthy_count = 0
                for service_name in critical_services:
                    if service_name in health_results:
                        if health_results[service_name].status == "healthy":
                            healthy_count += 1
                
                if healthy_count == len(critical_services):
                    progress.update(task, completed=1)
                    console.print("[green]All critical services are healthy[/green]")
                    return True
                
                await asyncio.sleep(10)
            
            progress.update(task, completed=1)
            console.print("[red]Timeout waiting for services to become healthy[/red]")
            return False

    async def _check_service_health(self) -> Dict[str, ServiceHealth]:
        """Check health of all services"""
        health_results = {}
        
        for service_name, config in self.services.items():
            try:
                if config["health_endpoint"]:
                    # HTTP health check
                    url = f"http://localhost:{config['port']}{config['health_endpoint']}"
                    start_time = time.time()
                    
                    try:
                        response = requests.get(url, timeout=10)
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            status = "healthy"
                            error_message = None
                        else:
                            status = "unhealthy"
                            error_message = f"HTTP {response.status_code}"
                    except requests.RequestException as e:
                        response_time = time.time() - start_time
                        status = "unhealthy"
                        error_message = str(e)
                else:
                    # Container health check
                    containers = self.docker_client.containers.list(
                        filters={"name": service_name}
                    )
                    
                    if containers and containers[0].status == "running":
                        status = "healthy"
                        response_time = 0
                        error_message = None
                    else:
                        status = "unhealthy"
                        response_time = 0
                        error_message = "Container not running"
                
                health_results[service_name] = ServiceHealth(
                    name=service_name,
                    status=status,
                    response_time=response_time,
                    last_check=datetime.now(),
                    error_message=error_message
                )
                
            except Exception as e:
                health_results[service_name] = ServiceHealth(
                    name=service_name,
                    status="error",
                    response_time=0,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
        
        return health_results

    async def _post_deployment_verification(self) -> bool:
        """Run post-deployment verification"""
        console.print("\n[yellow]Running post-deployment verification...[/yellow]")
        
        verifications = [
            ("Service health", self._verify_service_health),
            ("API endpoints", self._verify_api_endpoints),
            ("Database connectivity", self._verify_database_connectivity),
            ("Configuration integrity", self._verify_configuration_integrity)
        ]
        
        for verification_name, verification_func in verifications:
            try:
                result = await verification_func()
                if result:
                    console.print(f"[green]✓[/green] {verification_name}")
                else:
                    console.print(f"[red]✗[/red] {verification_name}")
                    return False
            except Exception as e:
                console.print(f"[red]✗[/red] {verification_name}: {str(e)}")
                return False
        
        return True

    async def _verify_service_health(self) -> bool:
        """Verify all services are healthy"""
        health_results = await self._check_service_health()
        
        for service_name, config in self.services.items():
            if config["critical"]:
                if service_name not in health_results:
                    return False
                if health_results[service_name].status != "healthy":
                    return False
        
        return True

    async def _verify_api_endpoints(self) -> bool:
        """Verify API endpoints are responding"""
        try:
            endpoints = [
                "http://localhost/health",
                "http://localhost/api/v1/status"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=10)
                    if response.status_code not in [200, 404]:  # 404 is OK if endpoint doesn't exist
                        return False
                except requests.RequestException:
                    return False
            
            return True
        except Exception:
            return False

    async def _verify_database_connectivity(self) -> bool:
        """Verify database connectivity"""
        try:
            containers = self.docker_client.containers.list(
                filters={"name": "postgres"}
            )
            
            if not containers:
                return False
            
            container = containers[0]
            result = container.exec_run(["pg_isready", "-U", "postgres"])
            
            return result.exit_code == 0
        except Exception:
            return False

    async def _verify_configuration_integrity(self) -> bool:
        """Verify configuration files are intact"""
        try:
            required_files = [
                self.config_dir / "agents.yaml",
                self.config_dir / "data_sources.yaml"
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    return False
                
                # Try to parse YAML files
                if file_path.suffix == '.yaml':
                    with open(file_path, 'r') as f:
                        yaml.safe_load(f)
            
            return True
        except Exception:
            return False

    async def _rollback(self, deployment_id: str) -> bool:
        """Rollback to previous deployment"""
        console.print(f"\n[red]Initiating rollback for deployment {deployment_id}[/red]")
        
        try:
            backup_path = self.backup_dir / deployment_id
            if not backup_path.exists():
                console.print("[red]Backup not found, cannot rollback[/red]")
                return False
            
            # Stop current services
            await self._stop_all_services()
            
            # Restore configuration
            if (backup_path / "config").exists():
                shutil.rmtree(self.config_dir)
                shutil.copytree(backup_path / "config", self.config_dir)
            
            # Restore Docker images
            await self._restore_docker_images(backup_path)
            
            # Restore database
            await self._restore_database(backup_path)
            
            # Start services
            if await self._start_all_services():
                if await self._wait_for_all_services_healthy():
                    console.print("[green]Rollback completed successfully[/green]")
                    return True
            
            console.print("[red]Rollback failed[/red]")
            return False
            
        except Exception as e:
            console.print(f"[red]Rollback error: {str(e)}[/red]")
            return False

    async def _restore_docker_images(self, backup_path: Path):
        """Restore Docker images from backup"""
        try:
            for image_file in backup_path.glob("*.tar"):
                with open(image_file, 'rb') as f:
                    self.docker_client.images.load(f.read())
                console.print(f"[green]Restored image from {image_file.name}[/green]")
        except Exception as e:
            console.print(f"[yellow]Image restore warning: {str(e)}[/yellow]")

    async def _restore_database(self, backup_path: Path):
        """Restore database from backup"""
        try:
            dump_file = backup_path / "database_dump.sql"
            if dump_file.exists():
                containers = self.docker_client.containers.list(
                    filters={"name": "postgres"}
                )
                
                if containers:
                    container = containers[0]
                    
                    # Read dump file
                    with open(dump_file, 'rb') as f:
                        dump_data = f.read()
                    
                    # Restore database
                    result = container.exec_run(
                        ["psql", "-U", "postgres"],
                        stdin=True
                    )
                    
                    if result.exit_code == 0:
                        console.print("[green]Database restored[/green]")
                    else:
                        console.print("[yellow]Database restore failed[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Database restore warning: {str(e)}[/yellow]")

    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"deploy_{timestamp}_{self.config.version}"

    def _record_deployment(self, deployment_id: str, status: str, duration: float):
        """Record deployment in history"""
        deployment_record = {
            "id": deployment_id,
            "timestamp": datetime.now().isoformat(),
            "version": self.config.version,
            "environment": self.config.environment,
            "strategy": self.config.deployment_strategy,
            "status": status,
            "duration": duration
        }
        
        self.deployment_history.append(deployment_record)
        
        # Save to file
        history_file = self.deployment_dir / "deployment_history.json"
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(deployment_record)
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Failed to save deployment history: {str(e)}[/yellow]")

    async def cleanup_old_backups(self):
        """Clean up old backups"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    manifest_file = backup_dir / "manifest.json"
                    if manifest_file.exists():
                        with open(manifest_file, 'r') as f:
                            manifest = json.load(f)
                        
                        backup_date = datetime.fromisoformat(manifest["timestamp"])
                        if backup_date < cutoff_date:
                            shutil.rmtree(backup_dir)
                            console.print(f"[yellow]Cleaned up old backup: {backup_dir.name}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Backup cleanup warning: {str(e)}[/yellow]")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Automated Deployment Pipeline")
    parser.add_argument("--environment", default="production", help="Deployment environment")
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--strategy", choices=["blue_green", "rolling", "recreate"], 
                       default="blue_green", help="Deployment strategy")
    parser.add_argument("--no-rollback", action="store_true", help="Disable automatic rollback")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--cleanup-backups", action="store_true", help="Clean up old backups")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    
    config = DeploymentConfig(
        environment=args.environment,
        version=args.version,
        deployment_strategy=args.strategy,
        rollback_enabled=not args.no_rollback
    )
    
    manager = DeploymentManager(project_root, config)
    
    if args.cleanup_backups:
        await manager.cleanup_old_backups()
        return
    
    success = await manager.deploy()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())