#!/usr/bin/env python3
"""
Test Order Execution - Verify Alpaca Order Placement
"""
import sys
import os
import asyncio
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from core.alpaca_client import AlpacaClient, AlpacaOrder, AlpacaOrderSide, AlpacaOrderType, AlpacaTimeInForce
from dotenv import load_dotenv

async def test_order_execution():
    """Test placing a small order in paper trading mode"""
    
    # Load environment variables
    load_dotenv()
    
    print("=== Testing Order Execution ===")
    
    try:
        # Initialize Alpaca client in paper trading mode
        alpaca_client = AlpacaClient(paper_trading=True)
        
        # Test connection first
        print("Testing connection...")
        await alpaca_client.test_connection()
        print("✅ Connection successful")
        
        # Get account info
        account = await alpaca_client.get_account()
        buying_power = float(account.get('buying_power', 0))
        print(f"Buying Power: ${buying_power:,.2f}")
        
        if buying_power < 100:
            print("❌ Insufficient buying power for test order")
            return False
        
        # Create a small test order (1 share of AAPL)
        test_order = AlpacaOrder(
            symbol='AAPL',
            qty=1,
            side=AlpacaOrderSide.BUY,
            type=AlpacaOrderType.MARKET,
            time_in_force=AlpacaTimeInForce.DAY,
            client_order_id=f"test_order_{int(asyncio.get_event_loop().time())}"
        )
        
        print(f"\nPlacing test order:")
        print(f"  Symbol: {test_order.symbol}")
        print(f"  Side: {test_order.side.value}")
        print(f"  Quantity: {test_order.qty}")
        print(f"  Type: {test_order.type.value}")
        
        # Submit the order
        order_result = await alpaca_client.submit_order(test_order)
        
        print(f"\n✅ Order placed successfully!")
        print(f"  Order ID: {order_result.get('id')}")
        print(f"  Status: {order_result.get('status')}")
        print(f"  Symbol: {order_result.get('symbol')}")
        print(f"  Side: {order_result.get('side')}")
        print(f"  Quantity: {order_result.get('qty')}")
        
        # Wait a moment and check order status
        await asyncio.sleep(2)
        
        # Get updated orders to see if it filled
        orders = await alpaca_client.get_orders(status='all', limit=5)
        
        print(f"\n📋 Recent Orders:")
        for i, order in enumerate(orders[:3], 1):
            print(f"  {i}. {order.get('symbol')} {order.get('side')} {order.get('qty')} - {order.get('status')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing order execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_order_execution())
    
    if success:
        print(f"\n🎉 Order execution test completed successfully!")
        print("The trading system can place real orders through Alpaca.")
    else:
        print(f"\n💥 Order execution test failed!")
        print("There may be an issue with order placement.")