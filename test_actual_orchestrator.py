#!/usr/bin/env python3
"""
Test the actual SystemOrchestrator method with proper mocking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system_orchestrator import SystemOrchestrator
from datetime import datetime
import pytz
from unittest.mock import patch, MagicMock

def test_actual_orchestrator():
    """Test the actual SystemOrchestrator._get_active_market_symbols method"""
    print("🔍 Testing Actual SystemOrchestrator Method with Proper Mocking")
    print("=" * 70)
    
    # Create orchestrator instance
    orchestrator = SystemOrchestrator()
    
    # Test scenarios
    test_scenarios = [
        ("Current Time", None),  # No mocking, use current time
        ("Market Hours (14 UTC, Tuesday)", datetime(2025, 9, 16, 14, 0, 0, tzinfo=pytz.UTC)),
        ("After Hours (18 UTC, Tuesday)", datetime(2025, 9, 16, 18, 0, 0, tzinfo=pytz.UTC)),
        ("Before Hours (7 UTC, Tuesday)", datetime(2025, 9, 16, 7, 0, 0, tzinfo=pytz.UTC)),
        ("Weekend (14 UTC, Saturday)", datetime(2025, 9, 13, 14, 0, 0, tzinfo=pytz.UTC))
    ]
    
    for scenario_name, mock_time in test_scenarios:
        print(f"\n📊 {scenario_name}")
        
        if mock_time is None:
            # No mocking - use current time
            current_time = datetime.now(pytz.UTC)
            print(f"Time: {current_time} (REAL TIME)")
            active_symbols = orchestrator._get_active_market_symbols()
        else:
            # Mock datetime at the point where it's imported in the method
            print(f"Time: {mock_time} (MOCKED)")
            
            # Create a mock datetime class that behaves like the real one
            mock_datetime_class = MagicMock()
            mock_datetime_class.now.return_value = mock_time
            
            # Patch the import inside the method
            with patch.dict('sys.modules', {'datetime': mock_datetime_class}):
                # Also patch the local import
                with patch('builtins.__import__') as mock_import:
                    def side_effect(name, *args, **kwargs):
                        if name == 'datetime':
                            return mock_datetime_class
                        return __import__(name, *args, **kwargs)
                    
                    mock_import.side_effect = side_effect
                    active_symbols = orchestrator._get_active_market_symbols()
        
        print(f"Active symbols: {active_symbols}")
        
        # Check what types of symbols are included
        stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        forex_symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD']
        
        contains_stocks = any(stock in active_symbols for stock in stock_symbols)
        contains_crypto = any(crypto in active_symbols for crypto in crypto_symbols)
        contains_forex = any(forex in active_symbols for forex in forex_symbols)
        
        print(f"📈 Contains stocks: {contains_stocks}")
        print(f"🪙 Contains crypto: {contains_crypto}")
        print(f"💱 Contains forex: {contains_forex}")
        
        # Determine if market should be open
        test_time = mock_time if mock_time else datetime.now(pytz.UTC)
        market_should_be_open = test_time.weekday() < 5 and 9 <= test_time.hour <= 16
        
        print(f"\n🎯 Expected: Market {'OPEN' if market_should_be_open else 'CLOSED'}")
        print(f"🎯 Actual: Stocks {'INCLUDED' if contains_stocks else 'EXCLUDED'}")
        
        if market_should_be_open == contains_stocks:
            print(f"✅ CORRECT: Market hours logic working")
        else:
            print(f"❌ ISSUE: Market hours logic not working")
            print(f"   Time: {test_time}")
            print(f"   Hour: {test_time.hour}, Weekday: {test_time.weekday()}")
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY:")
    print("This test calls the actual SystemOrchestrator method with proper mocking")
    print("to verify the market hours logic is working correctly.")

def test_simple_method_call():
    """Simple test to see what the method returns right now"""
    print("\n\n🔍 Simple Method Call Test")
    print("=" * 40)
    
    orchestrator = SystemOrchestrator()
    current_time = datetime.now(pytz.UTC)
    
    print(f"Current time: {current_time}")
    print(f"Hour: {current_time.hour}, Weekday: {current_time.weekday()}")
    
    active_symbols = orchestrator._get_active_market_symbols()
    print(f"Active symbols: {active_symbols}")
    
    # Check if stocks are included
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    contains_stocks = any(stock in active_symbols for stock in stock_symbols)
    
    market_should_be_open = current_time.weekday() < 5 and 9 <= current_time.hour <= 16
    
    print(f"Market should be open: {market_should_be_open}")
    print(f"Contains stocks: {contains_stocks}")
    
    if market_should_be_open == contains_stocks:
        print("✅ Current logic is working correctly")
    else:
        print("❌ Current logic has issues")

if __name__ == "__main__":
    test_actual_orchestrator()
    test_simple_method_call()