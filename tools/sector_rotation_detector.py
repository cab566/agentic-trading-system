#!/usr/bin/env python3
"""
Sector Rotation Detector
Detects sector rotation patterns and identifies emerging sector opportunities
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
            def __init__(self, *args, **kwargs):
                pass
            async def fetch_batch(self, *args, **kwargs):
                return {}
            def get_metrics(self):
                return {}
        
        class BatchRequest:
            def __init__(self, *args, **kwargs):
                pass

logger = logging.getLogger(__name__)

class SectorRotationDetector:
    """Detect sector rotation patterns and opportunities"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.cache_manager = CacheManager()
        self.notification_manager = NotificationManager()
        
        # Initialize batch data downloader
        self.batch_downloader = BatchDataDownloader(
            cache_manager=self.cache_manager,
            max_workers=4
        )
        logger.info("Initialized SectorRotationDetector with batch data downloader")
        
        # Configuration
        config_data = self.config_manager.get_config('sector_rotation_detector')
        self.lookback_period = config_data.get('sector_lookback_period', 30)  # days
        self.rotation_threshold = config_data.get('sector_rotation_threshold', 0.02)  # 2%
        self.min_relative_strength = config_data.get('sector_min_relative_strength', 1.1)
        
        # Sector ETFs for analysis
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        # Market benchmark
        self.benchmark = 'SPY'
        
        logger.info("SectorRotationDetector initialized")
    
    async def detect_sector_rotation(self, custom_sectors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Detect sector rotation patterns and opportunities"""
        try:
            logger.info("Starting sector rotation detection...")
            
            sectors = custom_sectors or self.sector_etfs
            
            # Get sector performance data
            sector_performance = await self.analyze_sector_performance(sectors)
            
            # Calculate relative strength
            relative_strength = await self.calculate_relative_strength(sector_performance)
            
            # Identify rotation patterns
            rotation_signals = self._identify_rotation_signals(relative_strength)
            
            # Find opportunities
            opportunities = self._find_sector_opportunities(rotation_signals, sector_performance)
            
            result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'analysis_period': f"{self.lookback_period} days",
                'sectors_analyzed': len(sectors),
                'rotation_signals': rotation_signals,
                'sector_performance': sector_performance,
                'relative_strength': relative_strength,
                'opportunities': opportunities,
                'top_sectors': self._get_top_sectors(relative_strength, 3),
                'bottom_sectors': self._get_bottom_sectors(relative_strength, 3)
            }
            
            # Send notifications for significant rotations
            if opportunities:
                await self.notification_manager.send_notification(
                    f"Detected {len(opportunities)} sector rotation opportunities",
                    priority="medium"
                )
            
            logger.info(f"Sector rotation detection completed. Found {len(opportunities)} opportunities")
            return result
            
        except Exception as e:
            logger.error(f"Error in sector rotation detection: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def analyze_sector_performance(self, sectors: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance of each sector using batch processing"""
        try:
            performance_data = {}
            
            # Prepare symbols for batch processing
            all_symbols = list(sectors.values()) + [self.benchmark]
            
            # Create batch request for historical data
            hist_request = BatchRequest(
                symbols=all_symbols,
                data_type='history',
                period=f"{self.lookback_period + 10}d"
            )
            
            # Fetch batch data
            batch_results = await self.batch_downloader.fetch_batch([hist_request])
            hist_data = batch_results.get('history', {})
            
            # Get benchmark data
            benchmark_hist = hist_data.get(self.benchmark)
            if benchmark_hist is None or benchmark_hist.empty:
                # Fallback to individual fetch
                benchmark = yf.Ticker(self.benchmark)
                benchmark_hist = benchmark.history(period=f"{self.lookback_period + 10}d")
                
            if benchmark_hist.empty:
                raise ValueError(f"No data available for benchmark {self.benchmark}")
            
            benchmark_return = (benchmark_hist['Close'].iloc[-1] / benchmark_hist['Close'].iloc[-self.lookback_period] - 1)
            
            for sector_name, etf_symbol in sectors.items():
                try:
                    # Get sector ETF data from batch results
                    hist = hist_data.get(etf_symbol)
                    
                    if hist is None or hist.empty:
                        # Fallback to individual fetch
                        logger.warning(f"Batch data not available for {etf_symbol}, using individual fetch")
                        etf = yf.Ticker(etf_symbol)
                        hist = etf.history(period=f"{self.lookback_period + 10}d")
                    
                    if hist.empty:
                        logger.warning(f"No data available for {etf_symbol}")
                        continue
                    
                    # Calculate performance metrics
                    current_price = hist['Close'].iloc[-1]
                    period_start_price = hist['Close'].iloc[-self.lookback_period]
                    
                    total_return = (current_price / period_start_price - 1)
                    daily_returns = hist['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252)  # Annualized
                    
                    # Calculate momentum indicators
                    sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else sma_20
                    
                    # Volume analysis
                    avg_volume = hist['Volume'].mean()
                    recent_volume = hist['Volume'].iloc[-5:].mean()
                    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                    
                    performance_data[sector_name] = {
                        'symbol': etf_symbol,
                        'current_price': float(current_price),
                        'total_return': float(total_return),
                        'benchmark_return': float(benchmark_return),
                        'excess_return': float(total_return - benchmark_return),
                        'volatility': float(volatility),
                        'sma_20': float(sma_20),
                        'sma_50': float(sma_50),
                        'price_vs_sma20': float((current_price / sma_20 - 1)),
                        'price_vs_sma50': float((current_price / sma_50 - 1)),
                        'volume_ratio': float(volume_ratio),
                        'momentum_score': self._calculate_momentum_score(hist)
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing sector {sector_name} ({etf_symbol}): {e}")
                    continue
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error in sector performance analysis: {e}")
            return {}
    
    async def calculate_relative_strength(self, sector_performance: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate relative strength of each sector vs benchmark"""
        try:
            relative_strength = {}
            
            for sector_name, performance in sector_performance.items():
                # Relative strength = sector return / benchmark return
                benchmark_return = performance['benchmark_return']
                sector_return = performance['total_return']
                
                if benchmark_return != 0:
                    rs = (1 + sector_return) / (1 + benchmark_return)
                else:
                    rs = 1 + sector_return
                
                relative_strength[sector_name] = float(rs)
            
            return relative_strength
            
        except Exception as e:
            logger.error(f"Error calculating relative strength: {e}")
            return {}
    
    def _calculate_momentum_score(self, hist: pd.DataFrame) -> float:
        """Calculate momentum score for a sector"""
        try:
            if len(hist) < 20:
                return 0.0
            
            # Price momentum (20-day)
            price_momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1)
            
            # Volume momentum
            volume_momentum = (hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-20:-5].mean() - 1)
            
            # Moving average alignment
            sma_5 = hist['Close'].rolling(5).mean().iloc[-1]
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma_alignment = (sma_5 / sma_20 - 1)
            
            # Combine factors
            momentum_score = (price_momentum * 0.5 + volume_momentum * 0.2 + ma_alignment * 0.3)
            
            return float(momentum_score)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _identify_rotation_signals(self, relative_strength: Dict[str, float]) -> Dict[str, str]:
        """Identify sector rotation signals"""
        signals = {}
        
        # Sort sectors by relative strength
        sorted_sectors = sorted(relative_strength.items(), key=lambda x: x[1], reverse=True)
        
        for sector, rs in sorted_sectors:
            if rs >= self.min_relative_strength:
                if rs >= 1.15:  # 15% outperformance
                    signals[sector] = 'STRONG_OUTPERFORM'
                elif rs >= 1.05:  # 5% outperformance
                    signals[sector] = 'OUTPERFORM'
                else:
                    signals[sector] = 'NEUTRAL'
            elif rs <= 0.95:  # 5% underperformance
                if rs <= 0.85:  # 15% underperformance
                    signals[sector] = 'STRONG_UNDERPERFORM'
                else:
                    signals[sector] = 'UNDERPERFORM'
            else:
                signals[sector] = 'NEUTRAL'
        
        return signals
    
    def _find_sector_opportunities(self, rotation_signals: Dict[str, str], 
                                 sector_performance: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find sector rotation opportunities"""
        opportunities = []
        
        for sector, signal in rotation_signals.items():
            if sector not in sector_performance:
                continue
                
            performance = sector_performance[sector]
            
            # Look for strong outperformers with momentum
            if signal in ['STRONG_OUTPERFORM', 'OUTPERFORM'] and performance['momentum_score'] > 0.02:
                opportunities.append({
                    'sector': sector,
                    'symbol': performance['symbol'],
                    'signal': signal,
                    'opportunity_type': 'MOMENTUM_CONTINUATION',
                    'relative_strength': performance['excess_return'],
                    'momentum_score': performance['momentum_score'],
                    'volume_ratio': performance['volume_ratio'],
                    'recommendation': 'BUY',
                    'confidence': self._calculate_confidence(performance, signal)
                })
            
            # Look for oversold sectors showing signs of reversal
            elif signal == 'STRONG_UNDERPERFORM' and performance['momentum_score'] > -0.01:
                opportunities.append({
                    'sector': sector,
                    'symbol': performance['symbol'],
                    'signal': signal,
                    'opportunity_type': 'REVERSAL_PLAY',
                    'relative_strength': performance['excess_return'],
                    'momentum_score': performance['momentum_score'],
                    'volume_ratio': performance['volume_ratio'],
                    'recommendation': 'WATCH_FOR_REVERSAL',
                    'confidence': self._calculate_confidence(performance, signal) * 0.7  # Lower confidence for reversals
                })
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities
    
    def _calculate_confidence(self, performance: Dict[str, Any], signal: str) -> float:
        """Calculate confidence score for an opportunity"""
        confidence = 0.5  # Base confidence
        
        # Volume confirmation
        if performance['volume_ratio'] > 1.2:
            confidence += 0.2
        
        # Momentum alignment
        if performance['momentum_score'] > 0.02:
            confidence += 0.2
        elif performance['momentum_score'] < -0.02:
            confidence -= 0.1
        
        # Price vs moving averages
        if performance['price_vs_sma20'] > 0 and performance['price_vs_sma50'] > 0:
            confidence += 0.1
        
        # Signal strength
        if signal in ['STRONG_OUTPERFORM', 'STRONG_UNDERPERFORM']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_top_sectors(self, relative_strength: Dict[str, float], count: int) -> List[Dict[str, Any]]:
        """Get top performing sectors"""
        sorted_sectors = sorted(relative_strength.items(), key=lambda x: x[1], reverse=True)
        return [{'sector': sector, 'relative_strength': rs} for sector, rs in sorted_sectors[:count]]
    
    def _get_bottom_sectors(self, relative_strength: Dict[str, float], count: int) -> List[Dict[str, Any]]:
        """Get bottom performing sectors"""
        sorted_sectors = sorted(relative_strength.items(), key=lambda x: x[1])
        return [{'sector': sector, 'relative_strength': rs} for sector, rs in sorted_sectors[:count]]
    
    def get_detector_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the sector rotation detector"""
        metrics = {
            'detector_type': 'sector_rotation',
            'lookback_period': self.lookback_period,
            'rotation_threshold': self.rotation_threshold,
            'min_relative_strength': self.min_relative_strength,
            'sectors_tracked': len(self.sector_etfs),
            'benchmark': self.benchmark
        }
        
        # Add batch processing metrics
        if hasattr(self, 'batch_downloader'):
            batch_metrics = self.batch_downloader.get_performance_metrics()
            metrics.update({
                'batch_processing': batch_metrics,
                'batch_enabled': True
            })
        else:
            metrics['batch_enabled'] = False
        
        # Add cache metrics
        if hasattr(self, 'cache_manager') and hasattr(self.cache_manager, 'get_metrics'):
            cache_metrics = self.cache_manager.get_metrics()
            metrics.update({
                'cache_metrics': cache_metrics,
                'cache_ttl': self.config_manager.get('cache_ttl', 3600)
            })
        
        return metrics

# Example usage and testing
async def main():
    """Test the sector rotation detector"""
    detector = SectorRotationDetector()
    
    print("Testing Sector Rotation Detector...")
    
    # Test detection
    result = await detector.detect_sector_rotation()
    print(f"Detection result: {result['status']}")
    print(f"Found {len(result.get('opportunities', []))} opportunities")
    
    # Show top and bottom sectors
    if result.get('top_sectors'):
        print(f"Top sectors: {[s['sector'] for s in result['top_sectors']]}")
    if result.get('bottom_sectors'):
        print(f"Bottom sectors: {[s['sector'] for s in result['bottom_sectors']]}")

if __name__ == "__main__":
    asyncio.run(main())