#!/usr/bin/env python3
"""
Test script to verify Alpaca integration for dashboard
"""
import sys
import os
import asyncio
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from core.alpaca_client import AlpacaClient
from dotenv import load_dotenv

async def test_alpaca_dashboard_integration():
    """Test Alpaca connection and data retrieval for dashboard"""
    
    # Load environment variables
    load_dotenv()
    
    print("=== Testing Alpaca Connection for Dashboard ===")
    
    try:
        # Initialize Alpaca client
        alpaca_client = AlpacaClient(paper_trading=True)
        
        # Test connection and get real data
        print("Fetching account information...")
        account = await alpaca_client.get_account()
        
        print("Fetching positions...")
        positions = await alpaca_client.get_positions()
        
        # Display account information
        print(f"\nAccount ID: {account.get('id')}")
        print(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"Day Trade Buying Power: ${float(account.get('daytrading_buying_power', 0)):,.2f}")
        print(f"Number of Positions: {len(positions)}")
        
        # Display positions
        if positions:
            print("\n=== Current Positions ===")
            for pos in positions:
                symbol = pos.get('symbol')
                qty = float(pos.get('qty', 0))
                market_value = float(pos.get('market_value', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                print(f"{symbol}: {qty} shares, ${market_value:,.2f} value, ${unrealized_pl:,.2f} P&L")
        
        # Test portfolio allocation calculation (same as dashboard)
        print("\n=== Portfolio Allocation Calculation ===")
        portfolio_value = float(account.get('portfolio_value', 0))
        cash = float(account.get('cash', 0))
        
        if portfolio_value > 0:
            allocation = {}
            
            # Add positions
            for position in positions:
                symbol = position.get('symbol')
                market_value = abs(float(position.get('market_value', 0)))
                if market_value > 0:
                    allocation[symbol] = (market_value / portfolio_value) * 100
            
            # Add cash allocation
            if cash > 0:
                allocation["CASH"] = (cash / portfolio_value) * 100
            
            print("Portfolio Allocation:")
            for symbol, percentage in allocation.items():
                print(f"  {symbol}: {percentage:.2f}%")
                
            return True, allocation
        else:
            print("No portfolio value found")
            return False, {}
            
    except Exception as e:
        print(f"Error testing Alpaca connection: {e}")
        return False, {}

if __name__ == "__main__":
    success, allocation = asyncio.run(test_alpaca_dashboard_integration())
    
    if success:
        print(f"\n✅ Alpaca integration test successful!")
        print(f"Dashboard should now display real portfolio data with {len(allocation)} allocations")
    else:
        print(f"\n❌ Alpaca integration test failed!")
        print("Dashboard may fall back to database or demo data")