#!/usr/bin/env python3
"""
Configuration Manager for CrewAI Trading System

Handles loading, validation, and hot-reloading of all system configurations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import yaml
from pydantic import BaseModel, ValidationError

# Optional watchdog import for file monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    WATCHDOG_AVAILABLE = False


# Optional file change handler - only available if watchdog is installed
if WATCHDOG_AVAILABLE:
    class ConfigChangeHandler(FileSystemEventHandler):
        """Handler for configuration file changes."""
        
        def __init__(self, config_manager):
            self.config_manager = config_manager
            self.logger = logging.getLogger(__name__)
        
        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith('.yaml'):
                self.logger.info(f"Configuration file changed: {event.src_path}")
                self.config_manager.reload_config(event.src_path)
else:
    ConfigChangeHandler = None


class ConfigManager:
    """
    Manages all system configurations with hot-reloading capabilities.
    
    Handles loading, validation, and monitoring of configuration files
    for agents, data sources, strategies, and system settings.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration directory
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Configuration storage
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.config_timestamps: Dict[str, datetime] = {}
        self.trading_mode: str = "paper"  # Default trading mode
        
        # File watcher for hot-reloading
        self.observer: Optional[Observer] = None
        self.watch_enabled = True
        
        # Load all configurations
        self._load_all_configs()
        
        # Start file watching
        if self.watch_enabled:
            self._start_file_watcher()
        
        self.logger.info("Configuration Manager initialized")
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        config_files = {
            'agents': 'agents.yaml',
            'data_sources': 'data_sources.yaml',
            'strategies': 'strategies.yaml'
        }
        
        for config_name, filename in config_files.items():
            config_file = self.config_path / filename
            if config_file.exists():
                self._load_config_file(config_name, config_file)
            else:
                self.logger.warning(f"Configuration file not found: {config_file}")
                self.configs[config_name] = {}
    
    def _load_config_file(self, config_name: str, config_file: Path) -> None:
        """Load a specific configuration file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate configuration
            self._validate_config(config_name, config_data)
            
            # Store configuration
            self.configs[config_name] = config_data
            self.config_timestamps[config_name] = datetime.now()
            
            self.logger.info(f"Loaded configuration: {config_name} from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration {config_name}: {e}")
            # Keep existing configuration if reload fails
            if config_name not in self.configs:
                self.configs[config_name] = {}
    
    def _validate_config(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        if config_name == 'agents':
            self._validate_agents_config(config_data)
        elif config_name == 'data_sources':
            self._validate_data_sources_config(config_data)
        elif config_name == 'strategies':
            self._validate_strategies_config(config_data)
    
    def _validate_agents_config(self, config: Dict[str, Any]) -> None:
        """Validate agents configuration."""
        required_fields = ['agents', 'crew_settings', 'llm_config']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in agents config: {field}")
        
        # Validate each agent
        for agent_name, agent_config in config['agents'].items():
            required_agent_fields = ['role', 'goal', 'backstory']
            for field in required_agent_fields:
                if field not in agent_config:
                    raise ValueError(f"Missing required field '{field}' in agent '{agent_name}'")
    
    def _validate_data_sources_config(self, config: Dict[str, Any]) -> None:
        """Validate data sources configuration."""
        if 'data_sources' not in config:
            raise ValueError("Missing 'data_sources' section in data sources config")
        
        # Validate each data source
        for source_name, source_config in config['data_sources'].items():
            if 'enabled' not in source_config:
                raise ValueError(f"Missing 'enabled' field in data source '{source_name}'")
    
    def _validate_strategies_config(self, config: Dict[str, Any]) -> None:
        """Validate strategies configuration."""
        if 'strategies' not in config:
            raise ValueError("Missing 'strategies' section in strategies config")
        
        # Validate each strategy
        for strategy_name, strategy_config in config['strategies'].items():
            required_fields = ['enabled', 'description', 'parameters']
            for field in required_fields:
                if field not in strategy_config:
                    raise ValueError(f"Missing required field '{field}' in strategy '{strategy_name}'")
    
    def _start_file_watcher(self) -> None:
        """Start watching configuration files for changes."""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Watchdog not available - file watching disabled")
            self.watch_enabled = False
            return
            
        try:
            # Check if observer already exists and is running
            if self.observer is not None:
                if self.observer.is_alive():
                    self.logger.info("File watcher already running")
                    return
                else:
                    # Clean up old observer
                    try:
                        self.observer.stop()
                        self.observer.join()
                    except:
                        pass
            
            self.observer = Observer()
            event_handler = ConfigChangeHandler(self)
            self.observer.schedule(event_handler, str(self.config_path), recursive=False)
            self.observer.start()
            self.logger.info("Configuration file watcher started")
        except Exception as e:
            self.logger.error(f"Failed to start file watcher: {e}")
            self.watch_enabled = False
    
    def reload_config(self, file_path: str) -> None:
        """Reload a specific configuration file."""
        file_path = Path(file_path)
        config_name = None
        
        # Determine which configuration to reload
        if file_path.name == 'agents.yaml':
            config_name = 'agents'
        elif file_path.name == 'data_sources.yaml':
            config_name = 'data_sources'
        elif file_path.name == 'strategies.yaml':
            config_name = 'strategies'
        
        if config_name:
            self.logger.info(f"Reloading configuration: {config_name}")
            self._load_config_file(config_name, file_path)
        else:
            self.logger.warning(f"Unknown configuration file: {file_path}")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a complete configuration."""
        return self.configs.get(config_name, {})
    
    def get_agent_configs(self) -> Dict[str, Any]:
        """Get agent configurations."""
        return self.configs.get('agents', {}).get('agents', {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        agents = self.get_agent_configs()
        return agents.get(agent_name, {})
    
    def get_crew_config(self) -> Dict[str, Any]:
        """Get CrewAI crew configuration."""
        return self.configs.get('agents', {}).get('crew_settings', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.configs.get('agents', {}).get('llm_config', {})
    
    def get_data_source_configs(self) -> Dict[str, Any]:
        """Get all data source configurations."""
        return self.configs.get('data_sources', {}).get('data_sources', {})
    
    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """Get configuration for a specific data source."""
        sources = self.get_data_source_configs()
        return sources.get(source_name, {})
    
    def get_enabled_data_sources(self) -> List[str]:
        """Get list of enabled data sources."""
        sources = self.get_data_source_configs()
        return [name for name, config in sources.items() if config.get('enabled', False)]
    
    def get_strategy_configs(self) -> Dict[str, Any]:
        """Get all strategy configurations."""
        return self.configs.get('strategies', {}).get('strategies', {})
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        strategies = self.get_strategy_configs()
        return strategies.get(strategy_name, {})
    
    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategies."""
        strategies = self.get_strategy_configs()
        return [name for name, config in strategies.items() if config.get('enabled', False)]
    
    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio management configuration."""
        return self.configs.get('strategies', {}).get('portfolio_management', {})
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration."""
        return self.configs.get('strategies', {}).get('execution', {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.configs.get('agents', {}).get('environment', {})
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return self.configs.get('agents', {}).get('tool_settings', {})
    
    def get_data_management_config(self) -> Dict[str, Any]:
        """Get data management configuration."""
        return self.configs.get('data_sources', {}).get('data_management', {})
    
    def get_rate_limiting_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self.configs.get('data_sources', {}).get('rate_limiting', {})
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get trading scheduler configuration with default intervals."""
        # Default trading intervals in seconds
        default_config = {
            'crypto_interval': 60,      # 1 minute for crypto
            'forex_interval': 300,      # 5 minutes for forex
            'stocks_interval': 60,      # 1 minute for stocks
            'options_interval': 300,    # 5 minutes for options
        }
        
        # Try to get from strategies config, fall back to defaults
        strategies_config = self.configs.get('strategies', {})
        scheduler_config = strategies_config.get('scheduler', {})
        
        # Merge with defaults
        default_config.update(scheduler_config)
        return default_config
    
    def update_agent_config(self, agent_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration for a specific agent."""
        if 'agents' not in self.configs:
            self.configs['agents'] = {'agents': {}}
        
        if 'agents' not in self.configs['agents']:
            self.configs['agents']['agents'] = {}
        
        if agent_name not in self.configs['agents']['agents']:
            self.configs['agents']['agents'][agent_name] = {}
        
        # Update configuration
        self.configs['agents']['agents'][agent_name].update(config_updates)
        
        # Update timestamp
        self.config_timestamps['agents'] = datetime.now()
        
        self.logger.info(f"Updated configuration for agent: {agent_name}")
    
    def update_strategy_config(self, strategy_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration for a specific strategy."""
        if 'strategies' not in self.configs:
            self.configs['strategies'] = {'strategies': {}}
        
        if 'strategies' not in self.configs['strategies']:
            self.configs['strategies']['strategies'] = {}
        
        if strategy_name not in self.configs['strategies']['strategies']:
            self.configs['strategies']['strategies'][strategy_name] = {}
        
        # Update configuration
        self.configs['strategies']['strategies'][strategy_name].update(config_updates)
        
        # Update timestamp
        self.config_timestamps['strategies'] = datetime.now()
        
        self.logger.info(f"Updated configuration for strategy: {strategy_name}")
    
    def save_config(self, config_name: str) -> None:
        """Save configuration to file."""
        config_files = {
            'agents': 'agents.yaml',
            'data_sources': 'data_sources.yaml',
            'strategies': 'strategies.yaml'
        }
        
        if config_name not in config_files:
            raise ValueError(f"Unknown configuration name: {config_name}")
        
        config_file = self.config_path / config_files[config_name]
        
        try:
            # Create backup
            backup_file = config_file.with_suffix(f'.yaml.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            if config_file.exists():
                config_file.rename(backup_file)
            
            # Save new configuration
            with open(config_file, 'w') as f:
                yaml.dump(self.configs[config_name], f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Saved configuration: {config_name} to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration {config_name}: {e}")
            # Restore backup if save failed
            if backup_file.exists():
                backup_file.rename(config_file)
            raise
    
    def set_trading_mode(self, mode: str) -> None:
        """Set the trading mode (live or paper)."""
        if mode not in ['live', 'paper']:
            raise ValueError(f"Invalid trading mode: {mode}. Must be 'live' or 'paper'")
        self.trading_mode = mode
        self.logger.info(f"Trading mode set to: {mode}")
    
    def get_trading_mode(self) -> str:
        """Get the current trading mode."""
        return self.trading_mode
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get status of all configurations."""
        return {
            'loaded_configs': list(self.configs.keys()),
            'config_timestamps': {k: v.isoformat() for k, v in self.config_timestamps.items()},
            'watch_enabled': self.watch_enabled,
            'config_path': str(self.config_path),
            'trading_mode': self.trading_mode
        }
    
    def validate_all_configs(self) -> Dict[str, Any]:
        """Validate all loaded configurations."""
        validation_results = {}
        
        for config_name, config_data in self.configs.items():
            try:
                self._validate_config(config_name, config_data)
                validation_results[config_name] = {'valid': True, 'errors': []}
            except Exception as e:
                validation_results[config_name] = {'valid': False, 'errors': [str(e)]}
        
        return validation_results
    
    def stop(self) -> None:
        """Stop the configuration manager."""
        if WATCHDOG_AVAILABLE and hasattr(self, 'observer') and self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        
        # Safe logging - check if logger exists and is available
        if hasattr(self, 'logger') and self.logger:
            try:
                self.logger.info("Configuration Manager stopped")
            except (AttributeError, RuntimeError):
                # Logger might not be available during shutdown
                pass
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            # Safe cleanup - only call stop if the object is properly initialized
            if hasattr(self, 'observer'):
                self.stop()
        except (AttributeError, RuntimeError):
            # Ignore errors during destruction - object might be partially destroyed
            pass


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager(Path("../config"))
    
    # Get agent configurations
    agents = config_manager.get_agent_configs()
    print(f"Loaded {len(agents)} agents")
    
    # Get enabled strategies
    strategies = config_manager.get_enabled_strategies()
    print(f"Enabled strategies: {strategies}")
    
    # Get system status
    status = config_manager.get_config_status()
    print(f"Config status: {status}")