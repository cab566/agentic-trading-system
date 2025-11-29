#!/usr/bin/env python3
"""
Test script to verify real data connections are working properly
"""
from pathlib import Path
from core.config_manager import ConfigManager
from core.data_manager import UnifiedDataManager
from datetime import datetime, timedelta
import asyncio
import sys

async def test_real_data():
    print('🔍 Testing Real Data Connections...')
    
    try:
        # Initialize managers with correct config path
        config_path = Path('config')  # Directory path, not file path
        config_manager = ConfigManager(config_path)
        data_manager = UnifiedDataManager(config_manager)
        
        # Test YFinance data
        print('📈 Testing YFinance market data...')
        try:
            market_data = await data_manager.get_price_data('AAPL', '1d', 
                                                          datetime.now() - timedelta(days=7), 
                                                          datetime.now())
            print(f'✅ YFinance: Got response with success: {market_data.success}')
            if market_data.data is not None and len(market_data.data) > 0:
                if isinstance(market_data.data, dict):
                    print(f'   Data points: {len(market_data.data)}')
                    print(f'   Latest entry: {list(market_data.data.keys())[-1] if market_data.data else "None"}')
                elif hasattr(market_data.data, '__len__'):
                    print(f'   Data length: {len(market_data.data)}')
                print(f'   Source: {market_data.source}')
                print(f'   Cached: {market_data.cached}')
            else:
                print('   No data received')
                if market_data.error:
                    print(f'   Error: {market_data.error}')
        except Exception as e:
            print(f'❌ YFinance error: {e}')
        
        # Test News API
        print('📰 Testing News API...')
        try:
            news_data = await data_manager.get_news_data('AAPL', limit=5)
            print(f'✅ News API: Got response with success: {news_data.success}')
            if news_data.data is not None and len(news_data.data) > 0:
                if isinstance(news_data.data, list):
                    print(f'   Articles: {len(news_data.data)}')
                    if news_data.data[0] and isinstance(news_data.data[0], dict):
                        print(f'   Latest: {news_data.data[0].get("title", "No title")[:50]}...')
                print(f'   Source: {news_data.source}')
            else:
                print('   No news data received')
                if news_data.error:
                    print(f'   Error: {news_data.error}')
        except Exception as e:
            print(f'❌ News API error: {e}')
        
        # Test historical data
        print('📊 Testing historical data...')
        try:
            hist_data = await data_manager.get_historical_data('AAPL', '1d', 
                                                             datetime.now() - timedelta(days=30), 
                                                             datetime.now())
            if hist_data is not None:
                print(f'✅ Historical data: Got {len(hist_data)} rows')
                print(f'   Columns: {list(hist_data.columns)}')
                print(f'   Date range: {hist_data.index[0]} to {hist_data.index[-1]}')
            else:
                print('   No historical data received')
        except Exception as e:
            print(f'❌ Historical data error: {e}')
        
        print('✅ Real data connection test completed!')
        
    except Exception as e:
        print(f'❌ Critical error during testing: {e}')
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_real_data())