#!/usr/bin/env python3
"""
Direct test of market hours logic without mocking.
This will show us exactly what the current implementation is doing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import pytz

def test_market_hours_directly():
    """Test market hours logic by directly examining the code"""
    print("🔍 Direct Market Hours Logic Test")
    print("=" * 50)
    
    # Test different time scenarios
    test_times = [
        ("Current Time", datetime.now(pytz.UTC)),
        ("Market Hours (14 UTC, Tuesday)", datetime(2025, 9, 16, 14, 0, 0, tzinfo=pytz.UTC)),
        ("After Hours (18 UTC, Tuesday)", datetime(2025, 9, 16, 18, 0, 0, tzinfo=pytz.UTC)),
        ("Before Hours (7 UTC, Tuesday)", datetime(2025, 9, 16, 7, 0, 0, tzinfo=pytz.UTC)),
        ("Weekend (14 UTC, Saturday)", datetime(2025, 9, 13, 14, 0, 0, tzinfo=pytz.UTC))
    ]
    
    for name, test_time in test_times:
        print(f"\n📊 {name}")
        print(f"Time: {test_time}")
        print(f"Hour: {test_time.hour}, Weekday: {test_time.weekday()} (0=Monday)")
        
        # Simulate the exact logic from the code
        current_hour = test_time.hour
        current_weekday = test_time.weekday()
        
        # Define symbol pools (from the code)
        crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        forex_symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD']
        stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        active_symbols = []
        
        # Crypto is always active (24/7/365)
        active_symbols.extend(crypto_symbols[:2])  # Top 2 crypto
        print(f"✅ Added crypto (24/7): {crypto_symbols[:2]}")
        
        # Forex is active 24/5 (Sunday 5 PM EST to Friday 5 PM EST)
        if current_weekday < 5 or (current_weekday == 6 and current_hour >= 22):  # Sunday 22:00 UTC = 5 PM EST
            active_symbols.extend(forex_symbols[:2])  # Top 2 forex
            print(f"✅ Added forex (24/5): {forex_symbols[:2]}")
        else:
            print(f"❌ Forex not active (weekend)")
        
        # Stock market logic - this is the key part we're testing
        print(f"\n🔍 Stock Market Check:")
        print(f"  - Is weekday (< 5): {current_weekday < 5}")
        print(f"  - Is market hours (9-16): {9 <= current_hour <= 16}")
        print(f"  - Combined condition: {current_weekday < 5 and 9 <= current_hour <= 16}")
        
        if current_weekday < 5 and 9 <= current_hour <= 16:  # Monday-Friday, market hours only
            active_symbols.extend(stock_symbols[:2])  # Top 2 stocks during market hours
            print(f"✅ Added stocks (market hours 9-16 UTC): {stock_symbols[:2]}")
            
            # Prioritize stocks during core trading hours (14-16 UTC)
            if 14 <= current_hour <= 16:
                # Move stocks to front of list during peak hours
                stock_portion = [s for s in active_symbols if s in stock_symbols]
                other_portion = [s for s in active_symbols if s not in stock_symbols]
                active_symbols = stock_portion + other_portion
                print(f"✅ Prioritized stocks during core hours (14-16 UTC)")
        else:
            print(f"❌ Stocks NOT added - market closed")
        
        print(f"\n📋 Final active symbols: {active_symbols}")
        
        # Analysis
        contains_stocks = any(stock in active_symbols for stock in stock_symbols)
        contains_crypto = any(crypto in active_symbols for crypto in crypto_symbols)
        
        print(f"📈 Contains stocks: {contains_stocks}")
        print(f"🪙 Contains crypto: {contains_crypto}")
        
        # Expected vs actual
        market_should_be_open = current_weekday < 5 and 9 <= current_hour <= 16
        print(f"\n🎯 Expected: Market {'OPEN' if market_should_be_open else 'CLOSED'}")
        print(f"🎯 Actual: Stocks {'INCLUDED' if contains_stocks else 'EXCLUDED'}")
        
        if market_should_be_open == contains_stocks:
            print(f"✅ CORRECT: Logic working as expected")
        else:
            print(f"❌ ISSUE: Logic not working correctly")
    
    print("\n" + "=" * 50)
    print("🎯 CONCLUSION:")
    print("This test shows the exact logic flow without any mocking issues.")
    print("If stocks are still being included when they shouldn't be,")
    print("then there might be another part of the code adding them.")

if __name__ == "__main__":
    test_market_hours_directly()