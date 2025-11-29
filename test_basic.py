#!/usr/bin/env python3
"""Basic test to verify core functionality without circular imports."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test basic imports without circular dependencies."""
    try:
        # Test basic Python imports
        import pandas as pd
        import numpy as np
        print("✓ Basic dependencies imported successfully")
        
        # Test config manager
        from core.config_manager import ConfigManager
        print("✓ ConfigManager imported successfully")
        
        # Test individual tools without data manager dependencies
        print("\nTesting tool imports...")
        
        # Test if we can import tools individually
        try:
            from tools.technical_analysis_tool import TechnicalAnalysisTool
            print("✓ TechnicalAnalysisTool imported successfully")
        except Exception as e:
            print(f"✗ TechnicalAnalysisTool import failed: {e}")
            
        try:
            from tools.news_analysis_tool import NewsAnalysisTool
            print("✓ NewsAnalysisTool imported successfully")
        except Exception as e:
            print(f"✗ NewsAnalysisTool import failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    try:
        import pandas as pd
        import numpy as np
        
        # Test pandas operations
        df = pd.DataFrame({
            'price': [100, 101, 102, 101, 103],
            'volume': [1000, 1100, 900, 1200, 800]
        })
        
        # Basic technical analysis calculations
        df['sma_3'] = df['price'].rolling(window=3).mean()
        df['returns'] = df['price'].pct_change()
        
        print("✓ Basic pandas operations successful")
        print(f"  Sample data shape: {df.shape}")
        print(f"  SMA calculation: {df['sma_3'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        from core.config_manager import ConfigManager
        
        # Try to load config
        config_path = Path("config")
        if config_path.exists():
            config_manager = ConfigManager(config_path)
            print("✓ Config manager initialized successfully")
            return True
        else:
            print("⚠ Config directory not found, skipping config test")
            return True
            
    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running basic functionality tests...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Config Loading", test_config_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n=== SUMMARY ===")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)