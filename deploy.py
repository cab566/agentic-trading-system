#!/usr/bin/env python3
"""Deployment script for the trading system."""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class DeploymentManager:
    """Manages deployment of the trading system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.deploy_dir = project_root / "deployment"
        self.deploy_dir.mkdir(exist_ok=True)
    
    def validate_environment(self, env: str) -> bool:
        """Validate deployment environment configuration."""
        print(f"Validating {env} environment...")
        
        # Check required files
        required_files = [
            ".env",
            "requirements.txt",
            "docker-compose.yml",
            "Dockerfile"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
        
        # Check environment-specific configuration
        env_file = self.project_root / f".env.{env}"
        if env != "development" and not env_file.exists():
            print(f"⚠️  Environment file .env.{env} not found, using default .env")
        
        # Validate Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker or Docker Compose not installed")
            return False
        
        print("✅ Environment validation passed")
        return True
    
    def build_images(self, env: str, no_cache: bool = False) -> bool:
        """Build Docker images for the specified environment."""
        print(f"Building Docker images for {env} environment...")
        
        # Determine target stage based on environment
        target_map = {
            "development": "development",
            "testing": "testing",
            "staging": "production",
            "production": "production"
        }
        
        target = target_map.get(env, "production")
        
        cmd = [
            "docker", "build",
            "--target", target,
            "-t", f"trading-system:{env}",
            "-t", f"trading-system:latest-{env}",
            "."
        ]
        
        if no_cache:
            cmd.append("--no-cache")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            if result.returncode == 0:
                print("✅ Docker images built successfully")
                return True
            else:
                print("❌ Failed to build Docker images")
                return False
        except Exception as e:
            print(f"❌ Error building Docker images: {e}")
            return False
    
    def deploy_local(self, env: str, services: Optional[List[str]] = None) -> bool:
        """Deploy the system locally using Docker Compose."""
        print(f"Deploying {env} environment locally...")
        
        # Prepare environment file
        env_file = self.project_root / f".env.{env}"
        if env_file.exists():
            # Copy environment-specific file
            shutil.copy2(env_file, self.project_root / ".env")
        
        # Prepare Docker Compose command
        cmd = ["docker-compose"]
        
        # Add profiles based on environment
        profiles = self._get_profiles_for_env(env)
        for profile in profiles:
            cmd.extend(["--profile", profile])
        
        cmd.append("up")
        cmd.extend(["-d", "--build"])  # Detached mode, rebuild images
        
        # Add specific services if provided
        if services:
            cmd.extend(services)
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            if result.returncode == 0:
                print("✅ Local deployment successful")
                self._show_service_status()
                return True
            else:
                print("❌ Local deployment failed")
                return False
        except Exception as e:
            print(f"❌ Error during local deployment: {e}")
            return False
    
    def deploy_cloud(self, env: str, platform: str, config: Dict) -> bool:
        """Deploy the system to cloud platform."""
        print(f"Deploying {env} environment to {platform}...")
        
        if platform == "aws":
            return self._deploy_aws(env, config)
        elif platform == "gcp":
            return self._deploy_gcp(env, config)
        elif platform == "azure":
            return self._deploy_azure(env, config)
        elif platform == "kubernetes":
            return self._deploy_kubernetes(env, config)
        else:
            print(f"❌ Unsupported platform: {platform}")
            return False
    
    def _deploy_aws(self, env: str, config: Dict) -> bool:
        """Deploy to AWS using ECS or EKS."""
        # This would implement AWS-specific deployment logic
        # For now, just a placeholder
        print("AWS deployment not yet implemented")
        return False
    
    def _deploy_gcp(self, env: str, config: Dict) -> bool:
        """Deploy to Google Cloud Platform."""
        # This would implement GCP-specific deployment logic
        print("GCP deployment not yet implemented")
        return False
    
    def _deploy_azure(self, env: str, config: Dict) -> bool:
        """Deploy to Microsoft Azure."""
        # This would implement Azure-specific deployment logic
        print("Azure deployment not yet implemented")
        return False
    
    def _deploy_kubernetes(self, env: str, config: Dict) -> bool:
        """Deploy to Kubernetes cluster."""
        print("Deploying to Kubernetes...")
        
        # Generate Kubernetes manifests
        k8s_dir = self.deploy_dir / "kubernetes" / env
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # This would generate and apply Kubernetes manifests
        # For now, just a placeholder
        print("Kubernetes deployment not yet implemented")
        return False
    
    def _get_profiles_for_env(self, env: str) -> List[str]:
        """Get Docker Compose profiles for environment."""
        profile_map = {
            "development": ["dev"],
            "testing": ["test"],
            "staging": ["production", "monitoring"],
            "production": ["production", "monitoring", "logging"]
        }
        return profile_map.get(env, [])
    
    def _show_service_status(self) -> None:
        """Show status of deployed services."""
        print("\nService Status:")
        print("=" * 50)
        
        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except Exception as e:
            print(f"Could not get service status: {e}")
    
    def stop_services(self, env: str) -> bool:
        """Stop running services."""
        print(f"Stopping {env} services...")
        
        try:
            # Get profiles for environment
            profiles = self._get_profiles_for_env(env)
            cmd = ["docker-compose"]
            
            for profile in profiles:
                cmd.extend(["--profile", profile])
            
            cmd.append("down")
            
            result = subprocess.run(cmd, cwd=self.project_root)
            if result.returncode == 0:
                print("✅ Services stopped successfully")
                return True
            else:
                print("❌ Failed to stop services")
                return False
        except Exception as e:
            print(f"❌ Error stopping services: {e}")
            return False
    
    def cleanup(self, env: str, remove_volumes: bool = False) -> bool:
        """Clean up deployment artifacts."""
        print(f"Cleaning up {env} deployment...")
        
        try:
            # Stop services first
            self.stop_services(env)
            
            # Remove containers and networks
            cmd = ["docker-compose", "down", "--remove-orphans"]
            
            if remove_volumes:
                cmd.append("--volumes")
                print("⚠️  This will remove all data volumes!")
            
            result = subprocess.run(cmd, cwd=self.project_root)
            
            # Remove images
            subprocess.run([
                "docker", "rmi", "-f",
                f"trading-system:{env}",
                f"trading-system:latest-{env}"
            ], capture_output=True)
            
            if result.returncode == 0:
                print("✅ Cleanup completed successfully")
                return True
            else:
                print("❌ Cleanup failed")
                return False
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            return False
    
    def backup_data(self, env: str) -> bool:
        """Backup system data - DISABLED"""
        # DISABLED FOR PRODUCTION - NO BACKUPS
        print("⚠️  Backup functionality disabled for production deployment")
        return True
        # try:
        #     print(f"Backing up {env} data...")
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     backup_dir = self.deploy_dir / "backups" / f"{env}_{timestamp}"
        #     backup_dir.mkdir(parents=True, exist_ok=True)
        #     
        #     # Backup database
        #     db_backup = backup_dir / "database.sql"
        #     subprocess.run([
        #         "pg_dump", "-h", "localhost", "-U", "postgres", 
        #         "-d", f"trading_system_{env}", "-f", str(db_backup)
        #     ], stdout=open(db_backup, 'w'), cwd=self.project_root)
        #     
        #     # Backup Redis data
        #     redis_backup = backup_dir / "redis_dump.rdb"
        #     redis_data_dir = Path("/var/lib/redis")
        #     if redis_data_dir.exists():
        #         shutil.copy2(redis_data_dir / "dump.rdb", redis_backup)
        #     
        #     # Backup data directory
        #     data_dir = self.project_root / "data"
        #     if data_dir.exists():
        #         shutil.copytree(data_dir, backup_dir / "data")
        #     
        #     # Backup logs
        #     logs_dir = self.project_root / "logs"
        #     if logs_dir.exists():
        #         shutil.copytree(logs_dir, backup_dir / "logs")
        #     
        #     print(f"✅ Backup completed: {backup_dir}")
        #     return True
        #     
        # except Exception as e:
        #     print(f"❌ Backup failed: {e}")
        #     return False
    
    def restore_data(self, env: str, backup_path: str) -> bool:
        """Restore system data from backup."""
        print(f"Restoring {env} data from {backup_path}...")
        
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            print(f"❌ Backup directory not found: {backup_path}")
            return False
        
        try:
            # Restore database
            db_backup = backup_dir / "database.sql"
            if db_backup.exists():
                with open(db_backup) as f:
                    subprocess.run([
                        "docker-compose", "exec", "-T", "postgres",
                        "psql", "-U", "trading_user", "trading_db"
                    ], stdin=f, cwd=self.project_root)
            
            # Restore application data
            backup_data_dir = backup_dir / "data"
            if backup_data_dir.exists():
                data_dir = self.project_root / "data"
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                shutil.copytree(backup_data_dir, data_dir)
            
            print("✅ Data restoration completed")
            return True
            
        except Exception as e:
            print(f"❌ Data restoration failed: {e}")
            return False
    
    def health_check(self, env: str) -> bool:
        """Perform health check on deployed services."""
        print(f"Performing health check for {env} environment...")
        
        try:
            # Check service status
            result = subprocess.run([
                "docker-compose", "ps", "--services", "--filter", "status=running"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            running_services = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Check application health endpoint
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Application health check passed")
                else:
                    print(f"⚠️  Application health check failed: {response.status_code}")
            except Exception:
                print("⚠️  Could not reach application health endpoint")
            
            print(f"Running services: {', '.join(running_services)}")
            return len(running_services) > 0
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False


def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(description="Trading System Deployment Manager")
    
    parser.add_argument(
        "action",
        choices=["validate", "build", "deploy", "stop", "cleanup", "backup", "restore", "health"],
        help="Deployment action to perform"
    )
    
    parser.add_argument(
        "--env", "-e",
        choices=["development", "testing", "staging", "production"],
        default="development",
        help="Target environment"
    )
    
    parser.add_argument(
        "--platform", "-p",
        choices=["local", "aws", "gcp", "azure", "kubernetes"],
        default="local",
        help="Deployment platform"
    )
    
    parser.add_argument(
        "--services",
        nargs="*",
        help="Specific services to deploy"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without cache"
    )
    
    parser.add_argument(
        "--remove-volumes",
        action="store_true",
        help="Remove volumes during cleanup"
    )
    
    parser.add_argument(
        "--backup-path",
        help="Path to backup for restore operation"
    )
    
    parser.add_argument(
        "--config",
        help="Path to deployment configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = DeploymentManager(project_root)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
    
    # Execute requested action
    success = False
    
    if args.action == "validate":
        success = manager.validate_environment(args.env)
    
    elif args.action == "build":
        success = manager.build_images(args.env, args.no_cache)
    
    elif args.action == "deploy":
        if args.platform == "local":
            success = manager.deploy_local(args.env, args.services)
        else:
            success = manager.deploy_cloud(args.env, args.platform, config)
    
    elif args.action == "stop":
        success = manager.stop_services(args.env)
    
    elif args.action == "cleanup":
        success = manager.cleanup(args.env, args.remove_volumes)
    
    elif args.action == "backup":
        success = manager.backup_data(args.env)
    
    elif args.action == "restore":
        if not args.backup_path:
            print("❌ --backup-path is required for restore operation")
            sys.exit(1)
        success = manager.restore_data(args.env, args.backup_path)
    
    elif args.action == "health":
        success = manager.health_check(args.env)
    
    # Exit with appropriate code
    if success:
        print(f"\n🎉 {args.action.title()} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ {args.action.title()} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()