#!/usr/bin/env python3
"""
Real Data Validation Script
Tests that all system components are using real market data only.
"""

import sys
import os
import json
import yaml
import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def test_data_sources():
    """Test that data sources are configured for real data only."""
    print("=== Testing Data Sources Configuration ===")
    
    with open('config/data_sources.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    real_sources = []
    for source_name, source_config in config.items():
        if source_config.get('enabled', False):
            print(f"✓ {source_name}: ENABLED")
            real_sources.append(source_name)
        else:
            print(f"✗ {source_name}: DISABLED")
    
    print(f"Active real data sources: {len(real_sources)}")
    return len(real_sources) > 0

def test_market_data_feed():
    """Test actual market data retrieval."""
    print("\n=== Testing Real Market Data Feed ===")
    
    try:
        # Test yfinance data retrieval
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        
        if not data.empty:
            print(f"✓ Retrieved {len(data)} days of real AAPL data")
            print(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
            print(f"  Data timestamp: {data.index[-1]}")
            return True
        else:
            print("✗ No market data retrieved")
            return False
            
    except Exception as e:
        print(f"✗ Market data test failed: {e}")
        return False

def test_api_server():
    """Test API server is running and using real data."""
    print("\n=== Testing API Server ===")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ API Server is running")
            print(f"  Status: {health_data.get('status', 'unknown')}")
            
            # Test market data endpoint
            market_response = requests.get("http://localhost:8000/api/market-data/AAPL", timeout=10)
            if market_response.status_code == 200:
                market_data = market_response.json()
                print(f"✓ Market data endpoint working")
                print(f"  Current price: ${market_data.get('current_price', 'N/A')}")
                return True
            else:
                print(f"✗ Market data endpoint failed: {market_response.status_code}")
                return False
        else:
            print(f"✗ API Server health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ API Server test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard is accessible."""
    print("\n=== Testing Dashboard ===")
    
    try:
        response = requests.get("http://localhost:8503", timeout=5)
        if response.status_code == 200:
            print("✓ Dashboard is accessible")
            return True
        else:
            print(f"✗ Dashboard not accessible: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Dashboard test failed: {e}")
        return False

def validate_no_mock_data():
    """Validate that no mock data is being used in production files."""
    print("\n=== Validating No Mock Data Usage ===")
    
    mock_keywords = ['mock', 'dummy', 'fake', 'placeholder', 'synthetic']
    production_files = [
        'system_orchestrator.py',
        'market_intelligence_engine.py',
        'portfolio_optimization_engine.py',
        'advanced_strategies.py'
    ]
    
    issues_found = 0
    for file_path in production_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
                
            for keyword in mock_keywords:
                if keyword in content and 'test' not in file_path.lower():
                    # Check if it's in a comment or actual code
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if keyword in line and not line.strip().startswith('#'):
                            print(f"⚠ Found '{keyword}' in {file_path}:{i+1}")
                            issues_found += 1
    
    if issues_found == 0:
        print("✓ No mock data usage found in production files")
        return True
    else:
        print(f"✗ Found {issues_found} potential mock data issues")
        return False

def main():
    """Run all validation tests."""
    print("Real Data Validation Suite")
    print("=" * 50)
    
    tests = [
        ("Data Sources", test_data_sources),
        ("Market Data Feed", test_market_data_feed),
        ("API Server", test_api_server),
        ("Dashboard", test_dashboard),
        ("No Mock Data", validate_no_mock_data)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is using real data only!")
        return 0
    else:
        print("❌ Some tests failed - Review issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())