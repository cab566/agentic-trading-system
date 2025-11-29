#!/usr/bin/env python3
"""
Test script to validate external data sources are fetching real data.
This ensures alternative data sources like news, economic indicators, etc. are live.
"""

import asyncio
import sys
import requests
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from pathlib import Path
import os

# Load environment variables
load_dotenv()

from core.config_manager import ConfigManager
from core.data_manager import UnifiedDataManager
from core.data_types import DataRequest

async def test_external_data_sources():
    """Test external data sources to ensure they're fetching real data."""
    print("Testing External Data Sources")
    print("=" * 60)
    
    try:
        # Initialize data manager
        config_manager = ConfigManager(Path("config"))
        data_manager = UnifiedDataManager(config_manager)
        
        # Test 1: News API validation
        print("\n1. Testing News API")
        print("-" * 40)
        
        news_api_key = os.getenv('NEWS_API_KEY')
        if news_api_key:
            # Direct API test
            try:
                response = requests.get(
                    f"https://newsapi.org/v2/everything?q=stocks&apiKey={news_api_key}&pageSize=5",
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('articles') and len(data['articles']) > 0:
                        latest_article = data['articles'][0]
                        published_at = latest_article.get('publishedAt', '')
                        print(f"✅ News API: Live connection (latest: {published_at[:10]})")
                    else:
                        print("⚠️  News API: No articles returned")
                else:
                    print(f"❌ News API: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ News API: {e}")
        else:
            print("⚠️  News API: No API key configured")
        
        # Test via data manager
        request = DataRequest(
            symbol='AAPL',
            data_type='news',
            timeframe='',
            parameters={'limit': 3}
        )
        
        response = await data_manager.get_data(request)
        if response.error:
            print(f"⚠️  Data Manager News: {response.error}")
        elif isinstance(response.data, list) and len(response.data) > 0:
            print(f"✅ Data Manager News: {len(response.data)} articles retrieved")
        else:
            print("⚠️  Data Manager News: No data returned")
        
        # Test 2: FRED Economic Data
        print("\n2. Testing FRED Economic Data")
        print("-" * 40)
        
        fred_api_key = os.getenv('FRED_API_KEY')
        if fred_api_key:
            # Direct API test
            try:
                response = requests.get(
                    f"https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={fred_api_key}&file_type=json&limit=1&sort_order=desc",
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('observations') and len(data['observations']) > 0:
                        latest_obs = data['observations'][0]
                        obs_date = latest_obs.get('date', '')
                        print(f"✅ FRED API: Live connection (latest GDP: {obs_date})")
                    else:
                        print("⚠️  FRED API: No observations returned")
                else:
                    print(f"❌ FRED API: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ FRED API: {e}")
        else:
            print("⚠️  FRED API: No API key configured")
        
        # Test via data manager
        request = DataRequest(
            symbol='GDP',
            data_type='economic',
            timeframe='quarterly',
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )
        
        response = await data_manager.get_data(request)
        if response.error:
            print(f"⚠️  Data Manager FRED: {response.error}")
        elif hasattr(response.data, '__len__') and len(response.data) > 0:
            print(f"✅ Data Manager FRED: Economic data retrieved")
        else:
            print("⚠️  Data Manager FRED: No data returned")
        
        # Test 3: Polygon API validation
        print("\n3. Testing Polygon API")
        print("-" * 40)
        
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if polygon_api_key:
            # Direct API test
            try:
                response = requests.get(
                    f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apikey={polygon_api_key}",
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results') and len(data['results']) > 0:
                        result = data['results'][0]
                        timestamp = result.get('t', 0)
                        date_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
                        print(f"✅ Polygon API: Live connection (latest: {date_str})")
                    else:
                        print("⚠️  Polygon API: No results returned")
                else:
                    print(f"❌ Polygon API: HTTP {response.status_code} - {response.text[:100]}")
            except Exception as e:
                print(f"❌ Polygon API: {e}")
        else:
            print("⚠️  Polygon API: No API key configured")
        
        # Test 4: Forex Factory (web scraping)
        print("\n4. Testing Forex Factory")
        print("-" * 40)
        
        try:
            response = requests.get(
                "https://www.forexfactory.com/calendar",
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                timeout=10
            )
            if response.status_code == 200:
                if "calendar" in response.text.lower():
                    print("✅ Forex Factory: Live connection established")
                else:
                    print("⚠️  Forex Factory: Unexpected response format")
            else:
                print(f"❌ Forex Factory: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Forex Factory: {e}")
        
        # Test 5: Cryptocurrency data sources
        print("\n5. Testing Cryptocurrency Sources")
        print("-" * 40)
        
        # Test Binance
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if 'symbol' in data and data['symbol'] == 'BTCUSDT':
                    print("✅ Binance API: Live connection")
                else:
                    print("⚠️  Binance API: Unexpected response")
            else:
                print(f"❌ Binance API: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Binance API: {e}")
        
        # Test Coinbase
        try:
            response = requests.get(
                "https://api.coinbase.com/v2/exchange-rates?currency=BTC",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rates' in data['data']:
                    print("✅ Coinbase API: Live connection")
                else:
                    print("⚠️  Coinbase API: Unexpected response")
            else:
                print(f"❌ Coinbase API: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Coinbase API: {e}")
        
        # Test 6: Data freshness validation
        print("\n6. Testing Data Freshness")
        print("-" * 40)
        
        current_time = datetime.now(pytz.UTC)
        
        # Test multiple symbols for freshness
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        fresh_data_count = 0
        
        for symbol in test_symbols:
            request = DataRequest(
                symbol=symbol,
                data_type='price',
                timeframe='1d',
                start_date=current_time - timedelta(days=7),
                end_date=current_time
            )
            
            response = await data_manager.get_data(request)
            
            if not response.error and hasattr(response.data, 'index') and len(response.data) > 0:
                latest_timestamp = response.data.index[-1]
                
                # Handle timezone-aware datetime comparison
                if hasattr(latest_timestamp, 'to_pydatetime'):
                    latest_dt = latest_timestamp.to_pydatetime()
                else:
                    latest_dt = latest_timestamp
                
                # Make both datetimes timezone-aware for comparison
                if latest_dt.tzinfo is None:
                    latest_dt = pytz.UTC.localize(latest_dt)
                elif latest_dt.tzinfo != pytz.UTC:
                    latest_dt = latest_dt.astimezone(pytz.UTC)
                
                time_diff = current_time - latest_dt
                
                if time_diff.days <= 3:  # Allow for weekends
                    fresh_data_count += 1
                    print(f"✅ {symbol}: Fresh data ({time_diff.days} days old)")
                else:
                    print(f"⚠️  {symbol}: Stale data ({time_diff.days} days old)")
            else:
                print(f"❌ {symbol}: No data available")
        
        # Summary
        print("\n" + "=" * 60)
        print("EXTERNAL DATA SOURCES VALIDATION SUMMARY")
        print("=" * 60)
        
        # Check API keys configuration
        api_keys_configured = 0
        total_api_keys = 4
        
        if os.getenv('NEWS_API_KEY'):
            api_keys_configured += 1
            print("✅ News API: Configured")
        else:
            print("⚠️  News API: Not configured")
        
        if os.getenv('FRED_API_KEY'):
            api_keys_configured += 1
            print("✅ FRED API: Configured")
        else:
            print("⚠️  FRED API: Not configured")
        
        if os.getenv('POLYGON_API_KEY'):
            api_keys_configured += 1
            print("✅ Polygon API: Configured")
        else:
            print("⚠️  Polygon API: Not configured")
        
        if os.getenv('ALPACA_API_KEY'):
            api_keys_configured += 1
            print("✅ Alpaca API: Configured")
        else:
            print("⚠️  Alpaca API: Not configured")
        
        print(f"\nAPI Configuration: {api_keys_configured}/{total_api_keys} configured")
        print(f"Fresh Data Sources: {fresh_data_count}/{len(test_symbols)} symbols")
        print("External Connectivity: Multiple sources tested")
        
        if api_keys_configured >= 2 and fresh_data_count >= 2:
            print("\n🎉 EXTERNAL DATA VALIDATION PASSED")
            return True
        else:
            print("\n⚠️  EXTERNAL DATA VALIDATION: Some issues detected")
            return False
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        return False
    
    finally:
        if 'data_manager' in locals():
            data_manager.stop()

if __name__ == "__main__":
    success = asyncio.run(test_external_data_sources())
    sys.exit(0 if success else 1)