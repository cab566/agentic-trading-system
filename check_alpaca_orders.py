#!/usr/bin/env python3
"""
Script to examine recent Alpaca orders and positions in detail
"""

import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import json

def main():
    # Load environment variables
    load_dotenv()
    
    print("📊 Analyzing Alpaca Trading Activity")
    print("=" * 50)
    
    try:
        from core.alpaca_client import AlpacaClient
        
        client = AlpacaClient(paper_trading=True)
        
        async def analyze_trading():
            # Get recent orders
            print("\n📋 Recent Orders:")
            print("-" * 30)
            orders = await client.get_orders(status='all', limit=10)
            
            for i, order in enumerate(orders, 1):
                print(f"\nOrder {i}:")
                print(f"  Symbol: {order.get('symbol', 'N/A')}")
                print(f"  Side: {order.get('side', 'N/A')}")
                print(f"  Qty: {order.get('qty', 'N/A')}")
                print(f"  Status: {order.get('status', 'N/A')}")
                print(f"  Order Type: {order.get('order_type', 'N/A')}")
                print(f"  Time in Force: {order.get('time_in_force', 'N/A')}")
                
                # Parse timestamps
                created_at = order.get('created_at', '')
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        print(f"  Created: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    except:
                        print(f"  Created: {created_at}")
                
                filled_qty = order.get('filled_qty', '0')
                if float(filled_qty) > 0:
                    print(f"  Filled Qty: {filled_qty}")
                    filled_avg_price = order.get('filled_avg_price')
                    if filled_avg_price:
                        print(f"  Avg Fill Price: ${float(filled_avg_price):.2f}")
            
            # Get current positions
            print("\n\n💼 Current Positions:")
            print("-" * 30)
            positions = await client.get_positions()
            
            total_market_value = 0
            for i, position in enumerate(positions, 1):
                qty = float(position.get('qty', 0))
                market_value = float(position.get('market_value', 0))
                unrealized_pl = float(position.get('unrealized_pl', 0))
                
                print(f"\nPosition {i}:")
                print(f"  Symbol: {position.get('symbol', 'N/A')}")
                print(f"  Quantity: {qty}")
                print(f"  Market Value: ${market_value:.2f}")
                print(f"  Unrealized P&L: ${unrealized_pl:.2f}")
                print(f"  Current Price: ${float(position.get('current_price', 0)):.2f}")
                
                total_market_value += market_value
            
            print(f"\nTotal Portfolio Market Value: ${total_market_value:.2f}")
            
            # Get account info
            print("\n\n💰 Account Summary:")
            print("-" * 30)
            account = await client.get_account()
            print(f"Equity: ${float(account.get('equity', 0)):.2f}")
            print(f"Cash: ${float(account.get('cash', 0)):.2f}")
            print(f"Buying Power: ${float(account.get('buying_power', 0)):.2f}")
            print(f"Day Trade Count: {account.get('daytrade_count', 0)}")
            
            # Check if orders are from our system
            print("\n\n🔍 Order Analysis:")
            print("-" * 30)
            
            recent_symbols = set()
            for order in orders:
                if order.get('status') == 'filled':
                    recent_symbols.add(order.get('symbol'))
            
            print(f"Recently traded symbols: {', '.join(recent_symbols) if recent_symbols else 'None'}")
            
            # Check if these match our simple_trading_bot symbols
            bot_symbols = ['TSLA', 'NVDA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'AAPL', 'NFLX']
            matching_symbols = recent_symbols.intersection(bot_symbols)
            
            if matching_symbols:
                print(f"✅ Symbols matching our bot: {', '.join(matching_symbols)}")
                print("✅ This suggests real trading activity from our system!")
            else:
                print("⚠️  No symbols match our trading bot")
        
        asyncio.run(analyze_trading())
        
    except Exception as e:
        print(f"\n❌ Error analyzing trading activity: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()