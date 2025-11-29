"""
Production readiness tests for the trading system.
"""
import pytest
import os
import json
from pathlib import Path


class TestProductionReadiness:
    """Test that the system is ready for production deployment."""
    
    def test_no_backup_directories(self):
        """Ensure no backup directories exist."""
        project_root = Path(__file__).parent.parent
        
        # Check for backup directories
        backup_dirs = list(project_root.glob("**/backup*"))
        assert len(backup_dirs) == 0, f"Found backup directories: {backup_dirs}"
        
        # Check for archive directories (excluding venv and system packages)
        archive_dirs = list(project_root.glob("**/archive*"))
        # Filter out venv and system package archives
        archive_dirs = [d for d in archive_dirs if "venv" not in str(d) and "site-packages" not in str(d)]
        assert len(archive_dirs) == 0, f"Found archive directories: {archive_dirs}"
    
    def test_backup_functionality_disabled(self):
        """Ensure backup functionality is disabled in configuration."""
        config_path = Path(__file__).parent.parent / "config" / "orchestrator_config.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Check that backup interval is disabled
            assert config.get("data_backup_interval", 0) == 0, "Backup interval should be 0 (disabled)"
    
    def test_environment_variables(self):
        """Check that production environment variables are set correctly."""
        # These should be set for production
        env_vars = {
            "TRADING_MODE": "paper_trading",  # Should be paper_trading for safety
            "LOG_LEVEL": "WARNING"  # Accept WARNING as valid production log level
        }
        
        for var, expected in env_vars.items():
            actual = os.getenv(var)
            if actual is not None:
                assert actual == expected, f"{var} should be {expected}, got {actual}"
    
    def test_no_test_data_files(self):
        """Ensure no test data files are present in production."""
        project_root = Path(__file__).parent.parent
        
        # Check for test data patterns
        test_patterns = ["*test*.csv", "*test*.db", "*mock*", "*sample*"]
        
        for pattern in test_patterns:
            test_files = list(project_root.glob(f"**/{pattern}"))
            # Filter out actual test files in tests directory
            test_files = [f for f in test_files if "tests/" not in str(f)]
            assert len(test_files) == 0, f"Found test data files: {test_files}"
    
    def test_production_mode_configuration(self):
        """Verify system is configured for production mode."""
        config_path = Path(__file__).parent.parent / "config" / "orchestrator_config.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Verify production settings
            assert config.get("execution_mode") in ["paper_trading", "live_trading"], \
                "Execution mode should be paper_trading or live_trading"
            
            assert config.get("log_level") in ["INFO", "WARNING", "ERROR"], \
                "Log level should be INFO, WARNING, or ERROR for production"
            
            assert config.get("auto_restart_failed_components", False) is True, \
                "Auto restart should be enabled for production"