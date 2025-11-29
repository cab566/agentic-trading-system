#!/usr/bin/env python3

import subprocess
import sys
import time

def check_running_system():
    """Check the current running system for data adapter status."""
    try:
        # Get the process ID of the running trading system
        result = subprocess.run(['pgrep', '-f', 'main.py run --mode paper'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pid = result.stdout.strip()
            print(f"Found running trading system with PID: {pid}")
            
            # Check if we can find log files or output
            print("\nChecking for recent log messages...")
            
            # Try to find any log files in the current directory
            log_result = subprocess.run(['find', '.', '-name', '*.log', '-mtime', '-1'], 
                                      capture_output=True, text=True)
            
            if log_result.stdout:
                print(f"Found log files: {log_result.stdout}")
            else:
                print("No recent log files found")
                
            return True
        else:
            print("No running trading system found")
            return False
            
    except Exception as e:
        print(f"Error checking system: {e}")
        return False

def test_data_adapters():
    """Test data adapter initialization directly."""
    print("\nTesting data adapter initialization...")
    
    try:
        # Import the necessary modules
        sys.path.append('.')
        from core.config_manager import ConfigManager
        from core.data_manager import UnifiedDataManager
        
        # Initialize config manager
        config_manager = ConfigManager('config/')
        
        # Get data source configs
        data_sources = config_manager.get_data_source_configs()
        enabled_sources = config_manager.get_enabled_data_sources()
        
        print(f"\nEnabled data sources: {list(enabled_sources) if isinstance(enabled_sources, list) else list(enabled_sources.keys())}")
        
        # Try to initialize data manager
        data_manager = UnifiedDataManager(config_manager)
        
        print(f"\nData manager adapters: {list(data_manager.adapters.keys())}")
        
        # Test each adapter
        if hasattr(data_manager, 'adapters') and data_manager.adapters:
            for name, adapter in data_manager.adapters.items():
                print(f"  - {name}: {type(adapter).__name__} (enabled: {adapter.enabled})")
        else:
            print("  No adapters found or adapters not initialized")
            
    except Exception as e:
        print(f"Error testing adapters: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Trading System Data Adapter Check ===")
    
    # Check running system
    is_running = check_running_system()
    
    # Test adapters directly
    test_data_adapters()
    
    print("\n=== Check Complete ===")