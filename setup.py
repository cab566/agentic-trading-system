#!/usr/bin/env python3
"""
Trading System v2 Setup Script

This script helps set up the trading system environment, including:
- Virtual environment creation
- Dependency installation
- Database initialization
- Configuration validation
- Initial system checks

Usage:
    python setup.py [options]

Options:
    --dev           Install development dependencies
    --no-venv       Skip virtual environment creation
    --force         Force reinstall dependencies
    --check-only    Only run system checks
    --init-db       Initialize database only
    --validate      Validate configuration only
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import json

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message: str, color: str = Colors.OKGREEN):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(message: str):
    """Print header message."""
    print_colored(f"\n{'='*60}", Colors.HEADER)
    print_colored(f"{message.center(60)}", Colors.HEADER)
    print_colored(f"{'='*60}", Colors.HEADER)

def print_step(step: str, message: str):
    """Print step message."""
    print_colored(f"[{step}] {message}", Colors.OKBLUE)

def print_success(message: str):
    """Print success message."""
    print_colored(f"✓ {message}", Colors.OKGREEN)

def print_warning(message: str):
    """Print warning message."""
    print_colored(f"⚠ {message}", Colors.WARNING)

def print_error(message: str):
    """Print error message."""
    print_colored(f"✗ {message}", Colors.FAIL)

def run_command(command: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command and return result."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(command)}")
        print_error(f"Error: {e.stderr}")
        raise

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_step("1", "Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} ✓")
    return True

def check_system_dependencies() -> bool:
    """Check for required system dependencies."""
    print_step("2", "Checking system dependencies...")
    
    required_commands = {
        'git': 'Git version control',
        'curl': 'HTTP client',
        'pip': 'Python package manager'
    }
    
    missing = []
    for cmd, desc in required_commands.items():
        if not shutil.which(cmd):
            missing.append(f"{cmd} ({desc})")
        else:
            print_success(f"{desc} found")
    
    if missing:
        print_error("Missing required dependencies:")
        for dep in missing:
            print_error(f"  - {dep}")
        return False
    
    return True

def create_virtual_environment(project_root: Path) -> bool:
    """Create Python virtual environment."""
    print_step("3", "Creating virtual environment...")
    
    venv_path = project_root / "venv"
    
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        return True
    
    try:
        run_command([sys.executable, "-m", "venv", str(venv_path)])
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to create virtual environment")
        return False

def get_pip_command(project_root: Path) -> List[str]:
    """Get pip command for the virtual environment."""
    if os.name == 'nt':  # Windows
        return [str(project_root / "venv" / "Scripts" / "pip.exe")]
    else:  # Unix-like
        return [str(project_root / "venv" / "bin" / "pip")]

def install_dependencies(project_root: Path, dev: bool = False, force: bool = False) -> bool:
    """Install Python dependencies."""
    print_step("4", "Installing dependencies...")
    
    pip_cmd = get_pip_command(project_root)
    
    # Upgrade pip first
    try:
        run_command(pip_cmd + ["install", "--upgrade", "pip"])
        print_success("Pip upgraded")
    except subprocess.CalledProcessError:
        print_warning("Failed to upgrade pip")
    
    # Install main dependencies
    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False
    
    install_cmd = pip_cmd + ["install", "-r", str(requirements_file)]
    if force:
        install_cmd.append("--force-reinstall")
    
    try:
        run_command(install_cmd)
        print_success("Main dependencies installed")
    except subprocess.CalledProcessError:
        print_error("Failed to install main dependencies")
        return False
    
    # Install development dependencies if requested
    if dev:
        dev_requirements = project_root / "requirements-dev.txt"
        if dev_requirements.exists():
            try:
                dev_cmd = pip_cmd + ["install", "-r", str(dev_requirements)]
                if force:
                    dev_cmd.append("--force-reinstall")
                run_command(dev_cmd)
                print_success("Development dependencies installed")
            except subprocess.CalledProcessError:
                print_warning("Failed to install development dependencies")
    
    return True

def create_directories(project_root: Path) -> bool:
    """Create necessary directories."""
    print_step("5", "Creating directories...")
    
    directories = [
        "data",
        "logs",
        "models",
        "backups",
        "temp",
        "exports",
        "reports"
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print_success(f"Directory created: {dir_name}/")
    
    # Create .gitkeep files for empty directories
    for dir_name in directories:
        gitkeep = project_root / dir_name / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    return True

def setup_environment_file(project_root: Path) -> bool:
    """Set up environment configuration file."""
    print_step("6", "Setting up environment configuration...")
    
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"
    
    if not env_example.exists():
        print_error(".env.example file not found")
        return False
    
    if env_file.exists():
        print_warning(".env file already exists")
        response = input("Overwrite existing .env file? (y/N): ")
        if response.lower() != 'y':
            print_success("Keeping existing .env file")
            return True
    
    try:
        shutil.copy2(env_example, env_file)
        print_success(".env file created from template")
        print_warning("Please edit .env file with your actual API keys and settings")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def validate_configuration(project_root: Path) -> bool:
    """Validate system configuration."""
    print_step("7", "Validating configuration...")
    
    env_file = project_root / ".env"
    if not env_file.exists():
        print_error(".env file not found")
        return False
    
    # Check for required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "TRADING_MODE",
        "DATABASE_URL",
        "LOG_LEVEL"
    ]
    
    missing_vars = []
    with open(env_file, 'r') as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=your-" in content or f"{var}=sk-your-" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print_warning("Configuration incomplete. Missing or placeholder values for:")
        for var in missing_vars:
            print_warning(f"  - {var}")
        print_warning("Please update .env file with actual values")
    else:
        print_success("Configuration appears complete")
    
    return True

def initialize_database(project_root: Path) -> bool:
    """Initialize the database."""
    print_step("8", "Initializing database...")
    
    try:
        # Import here to avoid issues if dependencies aren't installed yet
        sys.path.insert(0, str(project_root))
        
        # This would typically run database migrations
        # For now, just create the database file if using SQLite
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        db_file = data_dir / "trading_system.db"
        if not db_file.exists():
            db_file.touch()
            print_success("Database file created")
        else:
            print_success("Database file already exists")
        
        return True
    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        return False

def run_system_checks(project_root: Path) -> bool:
    """Run comprehensive system checks."""
    print_step("9", "Running system checks...")
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Python imports
    try:
        import crewai
        import pandas
        import numpy
        print_success("Core Python packages importable")
        checks_passed += 1
    except ImportError as e:
        print_error(f"Import error: {e}")
    
    # Check 2: Configuration files
    config_files = ["config/agents.yaml", "config/crews.yaml"]
    config_ok = True
    for config_file in config_files:
        if not (project_root / config_file).exists():
            print_error(f"Missing config file: {config_file}")
            config_ok = False
    
    if config_ok:
        print_success("Configuration files present")
        checks_passed += 1
    
    # Check 3: Directory structure
    required_dirs = ["tools", "agents", "crews", "data", "logs"]
    dirs_ok = True
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            print_error(f"Missing directory: {dir_name}")
            dirs_ok = False
    
    if dirs_ok:
        print_success("Directory structure correct")
        checks_passed += 1
    
    # Check 4: Environment file
    if (project_root / ".env").exists():
        print_success("Environment file present")
        checks_passed += 1
    else:
        print_error("Environment file missing")
    
    # Check 5: Write permissions
    try:
        test_file = project_root / "temp" / "write_test.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print_success("Write permissions OK")
        checks_passed += 1
    except Exception:
        print_error("Write permission issues")
    
    print_colored(f"\nSystem checks: {checks_passed}/{total_checks} passed", Colors.OKBLUE)
    return checks_passed == total_checks

def print_next_steps():
    """Print next steps for the user."""
    print_header("SETUP COMPLETE")
    
    print_colored("Next steps:", Colors.OKBLUE)
    print("1. Edit .env file with your API keys and settings")
    print("2. Activate virtual environment:")
    if os.name == 'nt':
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Run system validation:")
    print("   python main.py validate")
    print("4. Start the trading system:")
    print("   python main.py run --mode paper")
    
    print_colored("\nImportant:", Colors.WARNING)
    print("- Always use paper trading mode for testing")
    print("- Keep your API keys secure")
    print("- Monitor system logs regularly")
    print("- Read the README.md for detailed documentation")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Trading System v2 Setup")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--force", action="store_true", help="Force reinstall dependencies")
    parser.add_argument("--check-only", action="store_true", help="Only run system checks")
    parser.add_argument("--init-db", action="store_true", help="Initialize database only")
    parser.add_argument("--validate", action="store_true", help="Validate configuration only")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.absolute()
    
    print_header("TRADING SYSTEM V2 SETUP")
    print_colored(f"Project root: {project_root}", Colors.OKCYAN)
    
    success = True
    
    try:
        if args.check_only:
            success = run_system_checks(project_root)
        elif args.init_db:
            success = initialize_database(project_root)
        elif args.validate:
            success = validate_configuration(project_root)
        else:
            # Full setup process
            steps = [
                (check_python_version, []),
                (check_system_dependencies, []),
            ]
            
            if not args.no_venv:
                steps.append((create_virtual_environment, [project_root]))
                steps.append((install_dependencies, [project_root, args.dev, args.force]))
            
            steps.extend([
                (create_directories, [project_root]),
                (setup_environment_file, [project_root]),
                (validate_configuration, [project_root]),
                (initialize_database, [project_root]),
                (run_system_checks, [project_root])
            ])
            
            for step_func, step_args in steps:
                if not step_func(*step_args):
                    success = False
                    break
        
        if success:
            if not any([args.check_only, args.init_db, args.validate]):
                print_next_steps()
            print_colored("\n🎉 Setup completed successfully!", Colors.OKGREEN)
        else:
            print_colored("\n❌ Setup failed. Please check the errors above.", Colors.FAIL)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print_colored("\n\n⚠ Setup interrupted by user", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()