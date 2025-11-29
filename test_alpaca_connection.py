#!/usr/bin/env python3
"""
Test Alpaca API connection and credentials
"""

import asyncio
from dotenv import load_dotenv
from core.data_manager import AlpacaAdapter
from core.data_types import DataRequest

def test_alpaca_connection():
    print("Testing Alpaca API Connection")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Create config for AlpacaAdapter
        config = {
            'api_key_env': 'ALPACA_API_KEY',
            'secret_key_env': 'ALPACA_SECRET_KEY',
            'base_url': 'https://paper-api.alpaca.markets'
        }
        adapter = AlpacaAdapter(config)
        
        # Check credentials
        print(f"API Key loaded: {'Yes' if adapter.api_key else 'No'}")
        print(f"Secret Key loaded: {'Yes' if adapter.secret_key else 'No'}")
        print(f"Base URL: {adapter.base_url}")
        print()
        
        if not adapter.api_key or not adapter.secret_key:
            print("❌ FAILED: Alpaca API credentials not found")
            return False
        
        # Test account info
        print("Testing account info retrieval...")
        account_request = DataRequest(
            symbol="",
            data_type="account_info",
            timeframe="",
            start_date=None,
            end_date=None
        )
        
        async def test_account():
            response = await adapter.fetch_data(account_request)
            if response.error:
                print(f"❌ FAILED: {response.error}")
                return False
            else:
                print("✅ SUCCESS: Account info retrieved")
                print(f"Account ID: {response.data.get('id', 'N/A')}")
                print(f"Account Status: {response.data.get('status', 'N/A')}")
                print(f"Buying Power: ${response.data.get('buying_power', 'N/A')}")
                return True
        
        # Test positions
        print("\nTesting positions retrieval...")
        positions_request = DataRequest(
            symbol="",
            data_type="positions",
            timeframe="",
            start_date=None,
            end_date=None
        )
        
        async def test_positions():
            response = await adapter.fetch_data(positions_request)
            if response.error:
                print(f"❌ FAILED: {response.error}")
                return False
            else:
                print("✅ SUCCESS: Positions retrieved")
                print(f"Number of positions: {len(response.data) if isinstance(response.data, list) else 'N/A'}")
                return True
        
        # Run async tests
        async def run_tests():
            account_success = await test_account()
            positions_success = await test_positions()
            return account_success and positions_success
        
        success = asyncio.run(run_tests())
        
        if success:
            print("\n🎉 All Alpaca API tests passed!")
            print("✅ Real Alpaca API connection verified")
        else:
            print("\n❌ Some Alpaca API tests failed")
        
        return success
        
    except Exception as e:
        print(f"❌ FAILED: Exception occurred - {e}")
        return False

if __name__ == "__main__":
    test_alpaca_connection()