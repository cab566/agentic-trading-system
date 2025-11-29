#!/usr/bin/env python3
"""
Debug script to identify configuration loading issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.config_manager import ConfigManager

def debug_config():
    """Debug configuration loading"""
    print("🔍 Debugging Configuration Loading...")
    
    try:
        config_manager = ConfigManager(Path("config"))
        print("✅ ConfigManager initialized successfully")
        
        # Check data source configs
        print("\n📊 Data Source Configuration Debug:")
        data_sources = config_manager.get_data_source_configs()
        print(f"Type of data_sources: {type(data_sources)}")
        print(f"Value of data_sources: {data_sources}")
        
        if isinstance(data_sources, dict):
            print(f"Number of data sources: {len(data_sources)}")
            for name, config in data_sources.items():
                print(f"  - {name}: {type(config)} - enabled: {config.get('enabled', 'N/A')}")
                if not isinstance(config, dict):
                    print(f"    ❌ ERROR: Config for {name} is not a dict: {config}")
        else:
            print(f"❌ ERROR: Expected dict, got {type(data_sources)}")
            
        # Test enabled data sources
        print("\n✅ Enabled Data Sources:")
        enabled_sources = config_manager.get_enabled_data_sources()
        print(f"Enabled sources: {enabled_sources}")
            
        # Check raw config
        print("\n🔧 Raw Configuration Debug:")
        raw_config = config_manager.get_config('data_sources')
        print(f"Raw data_sources config type: {type(raw_config)}")
        print(f"Raw data_sources config: {raw_config}")
        
        # Check all configs
        print("\n📋 All Configurations:")
        for config_name in ['agents', 'data_sources', 'strategies']:
            config = config_manager.get_config(config_name)
            print(f"{config_name}: {type(config)} - Keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
            
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config()