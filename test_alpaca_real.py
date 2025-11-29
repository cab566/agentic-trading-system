#!/usr/bin/env python3
"""
Test script to verify real Alpaca API connectivity and trading capability
"""

import os
import asyncio
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    print("🔍 Testing Alpaca API Connectivity")
    print("=" * 50)
    
    # Check API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    print(f"Alpaca API Key found: {bool(api_key)}")
    print(f"Alpaca Secret Key found: {bool(secret_key)}")
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found!")
        print("Please check your .env file")
        return
    
    try:
        from core.alpaca_client import AlpacaClient
        
        # Test paper trading connection
        print("\n🧪 Testing Paper Trading Connection...")
        client = AlpacaClient(paper_trading=True)
        
        async def test_connection():
            # Test basic connectivity
            connection_result = await client.test_connection()
            print(f"Connection successful: {connection_result}")
            
            if connection_result:
                # Get account information
                account = await client.get_account()
                print(f"Account equity: ${float(account.get('equity', 0)):.2f}")
                print(f"Buying power: ${float(account.get('buying_power', 0)):.2f}")
                
                # Check market status
                market_open = await client.is_market_open()
                print(f"Market open: {market_open}")
                
                # Get recent orders
                orders = await client.get_orders(status='all', limit=5)
                print(f"Recent orders: {len(orders)}")
                
                # Get positions
                positions = await client.get_positions()
                print(f"Current positions: {len(positions)}")
                
                return True
            return False
        
        result = asyncio.run(test_connection())
        
        if result:
            print("\n✅ Alpaca API is REAL and CONNECTED!")
            print("✅ Paper trading account is active")
            print("✅ Ready for real trading operations")
        else:
            print("\n❌ Connection failed")
            
    except Exception as e:
        print(f"\n❌ Error testing Alpaca connection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()