#!/usr/bin/env python3
"""
Earnings Calendar Monitor
Monitors upcoming earnings announcements and identifies trading opportunities
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np

# Handle imports with fallback for direct execution
try:
    from ..core.config_manager import ConfigManager
    from ..utils.cache_manager import CacheManager
    from ..utils.notifications import NotificationManager
    from ..utils.yfinance_optimizer import BatchDataDownloader, BatchRequest
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from core.config_manager import ConfigManager
        from utils.cache_manager import CacheManager
        from utils.notifications import NotificationManager
        from utils.yfinance_optimizer import BatchDataDownloader, BatchRequest
    except ImportError:
        # Create mock classes for testing
        class ConfigManager:
            def __init__(self):
                self.config = {}
            def get(self, key, default=None):
                return self.config.get(key, default)
        
        class CacheManager:
            def __init__(self):
                pass
            async def get(self, key):
                return None
            async def set(self, key, value, ttl=3600):
                pass
        
        class NotificationManager:
            def __init__(self):
                pass
            async def send_notification(self, message, priority="info"):
                print(f"NOTIFICATION [{priority}]: {message}")
        
        class BatchDataDownloader:
            def __init__(self):
                pass
            async def download_batch(self, requests):
                return []
            def get_metrics(self):
                return {'total_requests': 0, 'successful_requests': 0, 'failed_requests': 0, 'success_rate': 0, 'average_batch_time': 0, 'total_api_calls_saved': 0}
        
        class BatchRequest:
            def __init__(self, symbols, data_type, period='1d', **kwargs):
                self.symbols = symbols
                self.data_type = data_type
                self.period = period

logger = logging.getLogger(__name__)

class EarningsCalendarMonitor:
    """Monitor earnings calendar and identify trading opportunities"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.cache_manager = CacheManager()
        self.notification_manager = NotificationManager()
        
        # Initialize batch data downloader for optimized API calls
        self.batch_downloader = BatchDataDownloader(
            cache_manager=self.cache_manager,
            max_workers=4
        )
        
        # Configuration
        config_data = self.config_manager.get_config('earnings_calendar_monitor')
        self.lookback_days = config_data.get('earnings_lookback_days', 30)
        self.lookahead_days = config_data.get('earnings_lookahead_days', 14)
        self.min_market_cap = config_data.get('earnings_min_market_cap', 1e9)  # $1B
        self.volatility_threshold = config_data.get('earnings_volatility_threshold', 0.05)  # 5%
        
        # Sample earnings calendar data (in production, this would come from a real API)
        self.sample_earnings = [
            {'symbol': 'AAPL', 'date': '2024-01-25', 'time': 'AMC', 'estimate': 2.10},
            {'symbol': 'MSFT', 'date': '2024-01-24', 'time': 'AMC', 'estimate': 2.78},
            {'symbol': 'GOOGL', 'date': '2024-01-30', 'time': 'AMC', 'estimate': 1.35},
            {'symbol': 'TSLA', 'date': '2024-01-24', 'time': 'AMC', 'estimate': 0.73},
            {'symbol': 'NVDA', 'date': '2024-02-21', 'time': 'AMC', 'estimate': 4.56},
            {'symbol': 'META', 'date': '2024-02-01', 'time': 'AMC', 'estimate': 4.96},
            {'symbol': 'AMZN', 'date': '2024-02-01', 'time': 'AMC', 'estimate': 0.80},
            {'symbol': 'NFLX', 'date': '2024-01-23', 'time': 'AMC', 'estimate': 2.22}
        ]
        
        logger.info("EarningsCalendarMonitor initialized")
    
    async def monitor_earnings_calendar(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Monitor earnings calendar for trading opportunities"""
        try:
            logger.info("Starting earnings calendar monitoring...")
            
            # Get upcoming earnings
            upcoming_earnings = await self.get_upcoming_earnings(symbols)
            
            # Analyze each earnings event
            opportunities = []
            for earnings in upcoming_earnings:
                analysis = await self.analyze_earnings_impact(earnings['symbol'], earnings)
                if analysis['opportunity_score'] > 0.6:  # High opportunity threshold
                    opportunities.append({
                        'symbol': earnings['symbol'],
                        'earnings_date': earnings['date'],
                        'earnings_time': earnings['time'],
                        'estimate': earnings['estimate'],
                        'analysis': analysis,
                        'opportunity_type': 'earnings_play'
                    })
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x['analysis']['opportunity_score'], reverse=True)
            
            result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'total_upcoming_earnings': len(upcoming_earnings),
                'high_opportunity_count': len(opportunities),
                'opportunities': opportunities[:10],  # Top 10
                'monitoring_period': f"{self.lookahead_days} days ahead",
                'criteria': {
                    'min_market_cap': self.min_market_cap,
                    'volatility_threshold': self.volatility_threshold
                }
            }
            
            # Send notifications for high-priority opportunities
            if opportunities:
                await self.notification_manager.send_notification(
                    f"Found {len(opportunities)} high-opportunity earnings plays",
                    priority="high"
                )
            
            logger.info(f"Earnings calendar monitoring completed. Found {len(opportunities)} opportunities")
            return result
            
        except Exception as e:
            logger.error(f"Error in earnings calendar monitoring: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_upcoming_earnings(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get upcoming earnings announcements"""
        try:
            # In production, this would fetch from a real earnings calendar API
            # For now, we'll use sample data and filter by date
            
            current_date = datetime.now().date()
            end_date = current_date + timedelta(days=self.lookahead_days)
            
            upcoming = []
            for earnings in self.sample_earnings:
                earnings_date = datetime.strptime(earnings['date'], '%Y-%m-%d').date()
                
                # Filter by date range
                if current_date <= earnings_date <= end_date:
                    # Filter by symbols if provided
                    if symbols is None or earnings['symbol'] in symbols:
                        upcoming.append(earnings)
            
            logger.info(f"Found {len(upcoming)} upcoming earnings announcements")
            return upcoming
            
        except Exception as e:
            logger.error(f"Error getting upcoming earnings: {e}")
            return []
    
    async def analyze_earnings_impact(self, symbol: str, earnings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential impact of earnings announcement"""
        try:
            # Try to use batch processing for data fetching
            hist_data = None
            info_data = None
            
            try:
                # Create batch requests for historical and info data
                batch_requests = [
                    BatchRequest(
                        symbols=[symbol],
                        data_type='history',
                        period='3mo'
                    ),
                    BatchRequest(
                        symbols=[symbol],
                        data_type='info',
                        period='1d'
                    )
                ]
                
                # Execute batch download
                batch_results = await self.batch_downloader.download_batch(batch_requests)
                
                if batch_results and len(batch_results) >= 2:
                    hist_data = batch_results[0]
                    info_data = batch_results[1]
                    
            except Exception as e:
                logger.debug(f"Batch processing failed for {symbol}, using fallback: {e}")
            
            # Fallback to individual yfinance calls if batch processing failed
            if hist_data is None or (hasattr(hist_data, 'empty') and hist_data.empty):
                stock = yf.Ticker(symbol)
                hist_data = stock.history(period="3mo")
                
            if info_data is None:
                stock = yf.Ticker(symbol)
                info_data = stock.info
            
            if hist_data is None or (hasattr(hist_data, 'empty') and hist_data.empty):
                return {'opportunity_score': 0, 'reason': 'No price data available'}
            
            # Calculate metrics
            current_price = hist_data['Close'].iloc[-1]
            volatility = hist_data['Close'].pct_change().std() * np.sqrt(252)  # Annualized
            avg_volume = hist_data['Volume'].mean()
            recent_volume = hist_data['Volume'].iloc[-5:].mean()  # Last 5 days
            
            # Price momentum
            price_change_1w = (current_price - hist_data['Close'].iloc[-5]) / hist_data['Close'].iloc[-5]
            price_change_1m = (current_price - hist_data['Close'].iloc[-20]) / hist_data['Close'].iloc[-20]
            
            # Market cap
            market_cap = info_data.get('marketCap', 0) if isinstance(info_data, dict) else 0
            
            # Calculate opportunity score
            opportunity_score = 0.0
            factors = []
            
            # High volatility increases opportunity
            if volatility > self.volatility_threshold:
                opportunity_score += 0.3
                factors.append(f"High volatility: {volatility:.1%}")
            
            # Volume surge indicates interest
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 1.2:
                opportunity_score += 0.2
                factors.append(f"Volume surge: {volume_ratio:.1f}x")
            
            # Market cap filter
            if market_cap >= self.min_market_cap:
                opportunity_score += 0.2
                factors.append(f"Large cap: ${market_cap/1e9:.1f}B")
            
            # Price momentum
            if abs(price_change_1w) > 0.05:  # 5% move in past week
                opportunity_score += 0.2
                factors.append(f"Recent momentum: {price_change_1w:.1%}")
            
            # Earnings timing (after market close is often more impactful)
            if earnings_data.get('time') == 'AMC':  # After Market Close
                opportunity_score += 0.1
                factors.append("After-hours announcement")
            
            return {
                'opportunity_score': min(opportunity_score, 1.0),
                'current_price': float(current_price),
                'volatility': float(volatility),
                'volume_ratio': float(volume_ratio),
                'price_change_1w': float(price_change_1w),
                'price_change_1m': float(price_change_1m),
                'market_cap': market_cap,
                'factors': factors,
                'recommendation': self._get_recommendation(opportunity_score, factors)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing earnings impact for {symbol}: {e}")
            return {
                'opportunity_score': 0,
                'error': str(e)
            }
    
    def _get_recommendation(self, score: float, factors: List[str]) -> str:
        """Get trading recommendation based on opportunity score"""
        if score >= 0.8:
            return "STRONG BUY - High-probability earnings play"
        elif score >= 0.6:
            return "BUY - Good earnings opportunity"
        elif score >= 0.4:
            return "WATCH - Monitor for entry signals"
        else:
            return "PASS - Low opportunity score"
    
    async def get_earnings_history(self, symbol: str, periods: int = 4) -> Dict[str, Any]:
        """Get historical earnings performance for a symbol"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get earnings data (this is simplified - real implementation would use earnings API)
            hist = stock.history(period="1y")
            
            # Simulate earnings dates (quarterly)
            earnings_dates = []
            current_date = datetime.now()
            for i in range(periods):
                earnings_date = current_date - timedelta(days=90 * (i + 1))
                earnings_dates.append(earnings_date.strftime('%Y-%m-%d'))
            
            # Calculate average post-earnings moves
            avg_move = hist['Close'].pct_change().abs().mean()
            
            return {
                'symbol': symbol,
                'historical_earnings_dates': earnings_dates,
                'average_post_earnings_move': float(avg_move),
                'volatility_pattern': 'Moderate' if avg_move < 0.05 else 'High'
            }
            
        except Exception as e:
            logger.error(f"Error getting earnings history for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_monitor_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the earnings calendar monitor"""
        try:
            # Get batch processing metrics
            batch_metrics = {}
            if hasattr(self.batch_downloader, 'get_metrics'):
                batch_metrics = self.batch_downloader.get_performance_metrics()
            
            # Get cache metrics
            cache_metrics = {}
            if hasattr(self.cache_manager, 'get_metrics'):
                cache_metrics = self.cache_manager.get_metrics()
            
            return {
                'monitor_type': 'earnings_calendar',
                'configuration': {
                    'lookback_days': self.lookback_days,
                    'lookahead_days': self.lookahead_days,
                    'min_market_cap': self.min_market_cap,
                    'volatility_threshold': self.volatility_threshold
                },
                'batch_processing': batch_metrics,
                'caching': cache_metrics,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting monitor metrics: {e}")
            return {
                'monitor_type': 'earnings_calendar',
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }

# Example usage and testing
async def main():
    """Test the earnings calendar monitor"""
    monitor = EarningsCalendarMonitor()
    
    print("Testing Earnings Calendar Monitor...")
    
    # Test monitoring
    result = await monitor.monitor_earnings_calendar(['AAPL', 'MSFT', 'GOOGL'])
    print(f"Monitoring result: {result['status']}")
    print(f"Found {result.get('high_opportunity_count', 0)} opportunities")
    
    # Test specific analysis
    if result.get('opportunities'):
        symbol = result['opportunities'][0]['symbol']
        analysis = await monitor.analyze_earnings_impact(symbol, {'date': '2024-01-25', 'time': 'AMC'})
        print(f"Analysis for {symbol}: Score {analysis.get('opportunity_score', 0):.2f}")

if __name__ == "__main__":
    asyncio.run(main())