#!/usr/bin/env python3
"""
Simple integration tests for the trading system core components.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_imports():
    """Test that core system components can be imported."""
    try:
        from core.config_manager import ConfigManager
        from core.data_manager import UnifiedDataManager
        from core.data_types import DataRequest, DataResponse, DataSourceAdapter
        print("✓ Core components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Core import failed: {e}")
        return False

def test_config_manager():
    """Test ConfigManager functionality."""
    try:
        from core.config_manager import ConfigManager
        
        config = ConfigManager()
        
        # Test basic config access
        assert hasattr(config, 'get')
        assert hasattr(config, 'set')
        
        # Test setting and getting values
        config.set('test_key', 'test_value')
        assert config.get('test_key') == 'test_value'
        
        print("✓ ConfigManager works correctly")
        return True
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        return False

def test_data_types():
    """Test data type classes."""
    try:
        from core.data_types import DataRequest, DataResponse
        
        # Test DataRequest creation
        request = DataRequest(
            symbol="AAPL",
            data_type="price",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        assert request.symbol == "AAPL"
        assert request.data_type == "price"
        
        # Test DataResponse creation
        sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Close': np.random.uniform(100, 200, 10)
        })
        
        response = DataResponse(
            data=sample_data,
            metadata={'source': 'test', 'symbol': 'AAPL'},
            success=True
        )
        
        assert response.success is True
        assert response.metadata['symbol'] == 'AAPL'
        
        print("✓ Data types work correctly")
        return True
    except Exception as e:
        print(f"✗ Data types test failed: {e}")
        return False

async def test_data_manager_creation():
    """Test UnifiedDataManager creation and basic functionality."""
    try:
        from core.config_manager import ConfigManager
        from core.data_manager import UnifiedDataManager
        
        # Create config manager
        config = ConfigManager()
        
        # Mock external dependencies
        with patch('core.data_manager.CryptoDataManager') as mock_crypto, \
             patch('core.data_manager.ForexDataManager') as mock_forex:
            
            # Create mock instances
            mock_crypto.return_value = Mock()
            mock_forex.return_value = Mock()
            
            # Create data manager
            data_manager = UnifiedDataManager(config)
            
            assert hasattr(data_manager, 'config')
            assert hasattr(data_manager, 'get_historical_data')
            assert hasattr(data_manager, 'get_real_time_data')
            
            print("✓ UnifiedDataManager created successfully")
            return True
    except Exception as e:
        print(f"✗ UnifiedDataManager test failed: {e}")
        return False

def test_system_initialization():
    """Test basic system initialization without external dependencies."""
    try:
        from core.config_manager import ConfigManager
        
        # Test with minimal config
        config = ConfigManager()
        
        # Set required config values
        config.set('trading.mode', 'paper')
        config.set('data.sources.stock.primary', 'yahoo')
        config.set('logging.level', 'INFO')
        
        # Test config validation
        mode = config.get('trading.mode')
        assert mode == 'paper'
        
        print("✓ System initialization successful")
        return True
    except Exception as e:
        print(f"✗ System initialization failed: {e}")
        return False

def test_portfolio_calculations():
    """Test basic portfolio calculation functions."""
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample portfolio data
        portfolio = {
            'AAPL': {'quantity': 100, 'avg_price': 150.0, 'current_price': 155.0},
            'GOOGL': {'quantity': 50, 'avg_price': 2500.0, 'current_price': 2550.0},
            'MSFT': {'quantity': 75, 'avg_price': 300.0, 'current_price': 310.0}
        }
        
        # Calculate total value
        total_value = sum(
            pos['quantity'] * pos['current_price'] 
            for pos in portfolio.values()
        )
        
        # Calculate total cost
        total_cost = sum(
            pos['quantity'] * pos['avg_price'] 
            for pos in portfolio.values()
        )
        
        # Calculate P&L
        pnl = total_value - total_cost
        pnl_percent = (pnl / total_cost) * 100
        
        assert total_value > 0
        assert total_cost > 0
        assert isinstance(pnl_percent, float)
        
        print(f"✓ Portfolio calculations successful (P&L: {pnl_percent:.2f}%)")
        return True
    except Exception as e:
        print(f"✗ Portfolio calculations failed: {e}")
        return False

def test_technical_analysis():
    """Test basic technical analysis calculations."""
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = pd.Series(
            np.random.uniform(100, 200, 50),
            index=dates
        )
        
        # Calculate simple moving average
        sma_20 = prices.rolling(window=20).mean()
        
        # Calculate RSI (simplified)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        assert not sma_20.isna().all()
        assert not rsi.isna().all()
        assert len(sma_20) == len(prices)
        
        print("✓ Technical analysis calculations successful")
        return True
    except Exception as e:
        print(f"✗ Technical analysis failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("Running integration tests...\n")
    
    sync_tests = [
        ("Core Imports", test_core_imports),
        ("Config Manager", test_config_manager),
        ("Data Types", test_data_types),
        ("System Initialization", test_system_initialization),
        ("Portfolio Calculations", test_portfolio_calculations),
        ("Technical Analysis", test_technical_analysis),
    ]
    
    async_tests = [
        ("Data Manager Creation", test_data_manager_creation),
    ]
    
    passed = 0
    total = len(sync_tests) + len(async_tests)
    
    # Run sync tests
    for test_name, test_func in sync_tests:
        print(f"=== {test_name} ===")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
        print()
    
    # Run async tests
    for test_name, test_func in async_tests:
        print(f"=== {test_name} ===")
        try:
            if await test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
        print()
    
    print(f"=== SUMMARY ===")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some integration tests failed")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))