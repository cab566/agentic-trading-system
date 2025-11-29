#!/usr/bin/env python3
"""
Test script to verify the newly configured API keys work correctly.
Tests News API, Polygon API, and FRED API connectivity.
"""

import os
import sys
import asyncio
import requests
from datetime import datetime, timedelta
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_manager import UnifiedDataManager
from core.data_types import DataRequest

def test_news_api_direct():
    """Test News API directly"""
    print("\n=== Testing News API (Direct) ===")
    
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        print("❌ NEWS_API_KEY not found in environment")
        return False
    
    print(f"✓ API Key found: {api_key[:8]}...")
    
    # Test with a simple query
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'Apple stock',
        'apiKey': api_key,
        'pageSize': 5,
        'sortBy': 'publishedAt'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            print(f"✓ Successfully fetched {len(articles)} articles")
            if articles:
                print(f"  Latest article: {articles[0]['title'][:60]}...")
            return True
        else:
            print(f"❌ API returned error: {data.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ News API test failed: {e}")
        return False

def test_polygon_api_direct():
    """Test Polygon API directly"""
    print("\n=== Testing Polygon API (Direct) ===")
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not found in environment")
        return False
    
    print(f"✓ API Key found: {api_key[:8]}...")
    
    # Test with a simple stock quote
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
    params = {
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK':
            results = data.get('results', [])
            if results:
                result = results[0]
                print(f"✓ Successfully fetched AAPL data")
                print(f"  Close: ${result.get('c', 'N/A')}")
                print(f"  Volume: {result.get('v', 'N/A'):,}")
                return True
            else:
                print("❌ No results returned")
                return False
        else:
            print(f"❌ API returned error: {data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Polygon API test failed: {e}")
        return False

def test_fred_api_direct():
    """Test FRED API directly"""
    print("\n=== Testing FRED API (Direct) ===")
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("❌ FRED_API_KEY not found in environment")
        return False
    
    print(f"✓ API Key found: {api_key[:8]}...")
    
    # Test with GDP data
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'GDP',
        'api_key': api_key,
        'file_type': 'json',
        'limit': 5,
        'sort_order': 'desc'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        observations = data.get('observations', [])
        if observations:
            latest = observations[0]
            print(f"✓ Successfully fetched GDP data")
            print(f"  Latest GDP ({latest['date']}): ${latest['value']} billion")
            return True
        else:
            print("❌ No observations returned")
            return False
            
    except Exception as e:
        print(f"❌ FRED API test failed: {e}")
        return False

async def test_unified_data_manager():
    """Test the APIs through the UnifiedDataManager"""
    print("\n=== Testing APIs through UnifiedDataManager ===")
    
    try:
        # Initialize the data manager
        data_manager = UnifiedDataManager()
        
        # Test News API through data manager
        print("\n--- Testing News via DataManager ---")
        news_request = DataRequest(
            data_type="news",
            symbol="AAPL",
            parameters={"query": "Apple stock", "limit": 3}
        )
        
        news_response = await data_manager.fetch_data(news_request)
        if news_response.success:
            print(f"✓ News data fetched successfully")
            if news_response.data:
                print(f"  Articles count: {len(news_response.data)}")
        else:
            print(f"❌ News fetch failed: {news_response.error}")
        
        # Test Polygon API through data manager
        print("\n--- Testing Polygon via DataManager ---")
        polygon_request = DataRequest(
            data_type="price",
            symbol="AAPL",
            parameters={"timeframe": "1D", "limit": 1}
        )
        
        polygon_response = await data_manager.fetch_data(polygon_request)
        if polygon_response.success:
            print(f"✓ Polygon price data fetched successfully")
            if polygon_response.data:
                print(f"  Data points: {len(polygon_response.data)}")
        else:
            print(f"❌ Polygon fetch failed: {polygon_response.error}")
        
        # Test FRED API through data manager
        print("\n--- Testing FRED via DataManager ---")
        fred_request = DataRequest(
            data_type="economic",
            symbol="GDP",
            parameters={"series_id": "GDP", "limit": 1}
        )
        
        fred_response = await data_manager.fetch_data(fred_request)
        if fred_response.success:
            print(f"✓ FRED economic data fetched successfully")
            if fred_response.data:
                print(f"  Data points: {len(fred_response.data)}")
        else:
            print(f"❌ FRED fetch failed: {fred_response.error}")
            
    except Exception as e:
        print(f"❌ UnifiedDataManager test failed: {e}")

def main():
    """Main test function"""
    print("🔧 Testing newly configured API keys...")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Track test results
    results = []
    
    # Test each API directly
    results.append(("News API", test_news_api_direct()))
    results.append(("Polygon API", test_polygon_api_direct()))
    results.append(("FRED API", test_fred_api_direct()))
    
    # Test through UnifiedDataManager
    print("\n" + "=" * 60)
    asyncio.run(test_unified_data_manager())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 API Test Summary:")
    print("=" * 60)
    
    for api_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{api_name:15} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} APIs working correctly")
    
    if passed == total:
        print("🎉 All API keys are configured and working!")
        return True
    else:
        print("⚠️  Some API keys need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)