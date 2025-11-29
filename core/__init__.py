"""
Core trading system modules.

This package contains the core components of the agentic trading system:
- Configuration management
- Data management and aggregation
- Trading orchestration and execution
- Risk management
- Health monitoring
"""

# Import key classes for easier access
from .config_manager import ConfigManager
from .data_manager import UnifiedDataManager

# Import other core modules as needed
try:
    from .orchestrator import TradingOrchestrator
except ImportError:
    TradingOrchestrator = None

try:
    from .execution_engine import ExecutionEngine
except ImportError:
    ExecutionEngine = None

try:
    from .risk_manager_24_7 import RiskManager24_7
except ImportError:
    RiskManager24_7 = None

try:
    from .health_monitor import HealthMonitor
except ImportError:
    HealthMonitor = None

__all__ = [
    'ConfigManager',
    'UnifiedDataManager'
]