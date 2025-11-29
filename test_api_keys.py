#!/usr/bin/env python3
"""
API Key Validation and Data Source Connectivity Test
Tests all configured API keys and data sources for live data access.
"""

import os
import sys
from dotenv import load_dotenv
import requests
import yfinance as yf
from datetime import datetime

def test_api_keys():
    """Test all API key configurations."""
    print('=== API Key Validation ===')
    
    # Load environment variables
    load_dotenv()
    
    issues = []
    
    # Test OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    print(f'OpenAI API Key: {openai_key[:20]}...' if openai_key and len(openai_key) > 20 else f'OpenAI API Key: {openai_key}')
    if openai_key == 'demo-key-replace-with-real-key':
        print('❌ OpenAI API key is placeholder - needs real key')
        issues.append('OpenAI API key is placeholder')
    elif not openai_key:
        print('❌ OpenAI API key is not set')
        issues.append('OpenAI API key is not set')
    else:
        print('✅ OpenAI API key appears to be set')

    # Test Alpha Vantage API key
    av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    print(f'Alpha Vantage API Key: {av_key}')
    if av_key == 'IXQZQHQJQZQZQZQZ':
        print('❌ Alpha Vantage API key is placeholder')
        issues.append('Alpha Vantage API key is placeholder')
    elif not av_key:
        print('❌ Alpha Vantage API key is not set')
        issues.append('Alpha Vantage API key is not set')
    else:
        print('✅ Alpha Vantage API key appears to be set')

    # Test News API key
    news_key = os.getenv('NEWS_API_KEY')
    print(f'News API Key: {news_key[:20]}...' if news_key and len(news_key) > 20 else f'News API Key: {news_key}')
    if not news_key:
        print('❌ News API key is not set')
        issues.append('News API key is not set')
    else:
        print('✅ News API key appears to be set')

    # Test Polygon API key
    polygon_key = os.getenv('POLYGON_API_KEY')
    print(f'Polygon API Key: {polygon_key[:20]}...' if polygon_key and len(polygon_key) > 20 else f'Polygon API Key: {polygon_key}')
    if not polygon_key:
        print('❌ Polygon API key is not set')
        issues.append('Polygon API key is not set')
    else:
        print('✅ Polygon API key appears to be set')

    # Test FRED API key
    fred_key = os.getenv('FRED_API_KEY')
    print(f'FRED API Key: {fred_key}')
    if not fred_key:
        print('❌ FRED API key is not set')
        issues.append('FRED API key is not set')
    else:
        print('✅ FRED API key appears to be set')

    # Test Alpaca API keys
    alpaca_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
    print(f'Alpaca API Key: {alpaca_key}')
    print(f'Alpaca Secret Key: {alpaca_secret[:20]}...' if alpaca_secret and len(alpaca_secret) > 20 else f'Alpaca Secret Key: {alpaca_secret}')
    
    if not alpaca_key:
        print('❌ Alpaca API key is not set')
        issues.append('Alpaca API key is not set')
    if not alpaca_secret:
        print('❌ Alpaca secret key is not set')
        issues.append('Alpaca secret key is not set')
    
    if alpaca_key and alpaca_secret:
        print('✅ Alpaca API keys appear to be set')
    
    return issues

def test_data_sources():
    """Test connectivity to all data sources."""
    print('\n=== Testing Data Source Connectivity ===')
    
    connectivity_issues = []
    
    # Test YFinance (no API key required)
    try:
        print('Testing YFinance...')
        ticker = yf.Ticker('AAPL')
        data = ticker.history(period='1d')
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            print(f'✅ YFinance: Successfully fetched AAPL data')
            print(f'   Latest price: ${latest_price:.2f}')
            print(f'   Data timestamp: {data.index[-1]}')
        else:
            print('❌ YFinance: No data returned')
            connectivity_issues.append('YFinance returned no data')
    except Exception as e:
        print(f'❌ YFinance: Error - {e}')
        connectivity_issues.append(f'YFinance error: {e}')

    # Test News API
    try:
        print('\nTesting News API...')
        news_key = os.getenv('NEWS_API_KEY')
        if news_key and news_key != 'your-news-api-key':
            response = requests.get(
                f'https://newsapi.org/v2/everything?q=apple&apiKey={news_key}&pageSize=1',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('totalResults', 0) > 0:
                    print('✅ News API: Successfully fetched news data')
                    print(f'   Total results available: {data.get("totalResults")}')
                else:
                    print('❌ News API: No articles returned')
                    connectivity_issues.append('News API returned no articles')
            else:
                print(f'❌ News API: HTTP {response.status_code} - {response.text[:100]}')
                connectivity_issues.append(f'News API HTTP {response.status_code}')
        else:
            print('⚠️  News API: Key not set or is placeholder')
            connectivity_issues.append('News API key not configured')
    except Exception as e:
        print(f'❌ News API: Error - {e}')
        connectivity_issues.append(f'News API error: {e}')

    # Test FRED API
    try:
        print('\nTesting FRED API...')
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key and fred_key != 'your-fred-api-key':
            response = requests.get(
                f'https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={fred_key}&file_type=json&limit=1',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data and data['observations']:
                    print('✅ FRED API: Successfully fetched economic data')
                    obs = data['observations'][0]
                    print(f'   Latest GDP data: {obs.get("value")} ({obs.get("date")})')
                else:
                    print('❌ FRED API: No observations returned')
                    connectivity_issues.append('FRED API returned no observations')
            else:
                print(f'❌ FRED API: HTTP {response.status_code}')
                connectivity_issues.append(f'FRED API HTTP {response.status_code}')
        else:
            print('⚠️  FRED API: Key not set or is placeholder')
            connectivity_issues.append('FRED API key not configured')
    except Exception as e:
        print(f'❌ FRED API: Error - {e}')
        connectivity_issues.append(f'FRED API error: {e}')

    # Test Alpaca API
    try:
        print('\nTesting Alpaca API...')
        alpaca_key = os.getenv('ALPACA_API_KEY')
        alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if alpaca_key and alpaca_secret:
            headers = {
                'APCA-API-KEY-ID': alpaca_key,
                'APCA-API-SECRET-KEY': alpaca_secret
            }
            
            # Test account endpoint
            response = requests.get(f'{base_url}/v2/account', headers=headers, timeout=10)
            if response.status_code == 200:
                account_data = response.json()
                print('✅ Alpaca API: Successfully connected to account')
                print(f'   Account status: {account_data.get("status")}')
                print(f'   Buying power: ${float(account_data.get("buying_power", 0)):,.2f}')
                print(f'   Portfolio value: ${float(account_data.get("portfolio_value", 0)):,.2f}')
            else:
                print(f'❌ Alpaca API: HTTP {response.status_code} - {response.text[:100]}')
                connectivity_issues.append(f'Alpaca API HTTP {response.status_code}')
        else:
            print('⚠️  Alpaca API: Keys not configured')
            connectivity_issues.append('Alpaca API keys not configured')
    except Exception as e:
        print(f'❌ Alpaca API: Error - {e}')
        connectivity_issues.append(f'Alpaca API error: {e}')
    
    return connectivity_issues

def main():
    """Main test function."""
    print(f'API Key and Data Source Validation - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)
    
    # Test API keys
    api_issues = test_api_keys()
    
    # Test data source connectivity
    connectivity_issues = test_data_sources()
    
    # Summary
    print('\n=== Summary ===')
    total_issues = len(api_issues) + len(connectivity_issues)
    
    if total_issues == 0:
        print('✅ All API keys and data sources are properly configured and accessible!')
    else:
        print(f'❌ Found {total_issues} issues:')
        
        if api_issues:
            print('\nAPI Key Issues:')
            for issue in api_issues:
                print(f'  - {issue}')
        
        if connectivity_issues:
            print('\nConnectivity Issues:')
            for issue in connectivity_issues:
                print(f'  - {issue}')
    
    print('\n=== Configuration Status ===')
    trading_mode = os.getenv('TRADING_MODE', 'paper')
    demo_mode = os.getenv('DEMO_MODE', 'false')
    print(f'Trading Mode: {trading_mode}')
    print(f'Demo Mode: {demo_mode}')
    
    if demo_mode.lower() == 'true':
        print('⚠️  Demo mode is enabled - system may use mock data')
    else:
        print('✅ Demo mode is disabled - system will use real data')
    
    return total_issues == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)