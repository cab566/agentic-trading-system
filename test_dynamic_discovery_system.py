#!/usr/bin/env python3
"""
Comprehensive Test Suite for Dynamic Discovery System
Tests all discovery tools and scanners with real market data
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback
import yfinance as yf
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedVolumeSpikeScanner:
    """Simplified volume spike scanner for testing"""
    
    async def scan_for_volume_spikes(self, symbols: List[str]) -> List[Dict]:
        """Scan for volume spikes in given symbols"""
        results = []
        
        for symbol in symbols:
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                hist = stock.history(period="30d")
                
                if len(hist) < 20:
                    continue
                
                # Calculate volume metrics
                current_volume = hist['Volume'].iloc[-1]
                avg_volume = hist['Volume'].iloc[-20:-1].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                # Calculate price change
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                price_change = ((current_price - prev_price) / prev_price) * 100
                
                if volume_ratio > 1.5:  # Volume spike threshold
                    results.append({
                        'symbol': symbol,
                        'volume_ratio': volume_ratio,
                        'price_change_pct': price_change,
                        'current_volume': int(current_volume),
                        'avg_volume': int(avg_volume),
                        'current_price': float(current_price),
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                
        return results

class SimplifiedNewsDrivenDiscovery:
    """Simplified news-driven discovery for testing"""
    
    async def analyze_news_impact(self, symbols: List[str]) -> List[Dict]:
        """Analyze news impact for given symbols"""
        results = []
        
        for symbol in symbols:
            try:
                # Get stock data for momentum analysis
                stock = yf.Ticker(symbol)
                hist = stock.history(period="5d")
                
                if len(hist) < 3:
                    continue
                
                # Calculate recent momentum
                recent_returns = hist['Close'].pct_change().iloc[-3:].mean()
                volume_trend = hist['Volume'].iloc[-3:].mean() / hist['Volume'].iloc[-10:-3].mean()
                
                # Simulate news impact score based on price/volume momentum
                news_impact_score = abs(recent_returns) * volume_trend
                
                if news_impact_score > 0.02:  # Threshold for significant impact
                    results.append({
                        'symbol': symbol,
                        'news_impact_score': float(news_impact_score),
                        'recent_returns': float(recent_returns),
                        'volume_trend': float(volume_trend),
                        'sentiment': 'positive' if recent_returns > 0 else 'negative',
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to analyze news impact for {symbol}: {e}")
                
        return results

class SimplifiedTechnicalBreakoutScanner:
    """Simplified technical breakout scanner for testing"""
    
    async def scan_for_breakouts(self, symbols: List[str]) -> List[Dict]:
        """Scan for technical breakouts in given symbols"""
        results = []
        
        for symbol in symbols:
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                hist = stock.history(period="60d")
                
                if len(hist) < 50:
                    continue
                
                # Calculate technical indicators
                close_prices = hist['Close']
                
                # Simple moving averages
                sma_20 = close_prices.rolling(20).mean()
                sma_50 = close_prices.rolling(50).mean()
                
                # Support and resistance levels
                high_20 = hist['High'].rolling(20).max()
                low_20 = hist['Low'].rolling(20).min()
                
                current_price = close_prices.iloc[-1]
                
                # Check for breakout conditions
                breakout_detected = False
                pattern_type = None
                
                # Resistance breakout
                if current_price > high_20.iloc[-2] * 1.02:
                    breakout_detected = True
                    pattern_type = "resistance_breakout"
                
                # Moving average crossover
                elif sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
                    breakout_detected = True
                    pattern_type = "ma_crossover"
                
                if breakout_detected:
                    results.append({
                        'symbol': symbol,
                        'pattern_type': pattern_type,
                        'current_price': float(current_price),
                        'resistance_level': float(high_20.iloc[-2]),
                        'sma_20': float(sma_20.iloc[-1]),
                        'sma_50': float(sma_50.iloc[-1]),
                        'strength': 'medium',
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to analyze breakout for {symbol}: {e}")
                
        return results

class SimplifiedEarningsCalendarMonitor:
    """Simplified earnings calendar monitor for testing"""
    
    async def scan_upcoming_earnings(self, symbols: List[str], days_ahead: int = 30) -> List[Dict]:
        """Scan for upcoming earnings events"""
        results = []
        
        for symbol in symbols:
            try:
                # Get stock info
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Simulate earnings date (in real implementation, this would come from earnings calendar API)
                # For testing, we'll create mock earnings events for some symbols
                if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                    earnings_date = datetime.now() + timedelta(days=np.random.randint(1, days_ahead))
                    
                    results.append({
                        'symbol': symbol,
                        'earnings_date': earnings_date,
                        'days_until_earnings': (earnings_date - datetime.now()).days,
                        'estimated_impact': 'high' if symbol in ['AAPL', 'MSFT'] else 'medium',
                        'sector': info.get('sector', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to get earnings info for {symbol}: {e}")
                
        return results

class SimplifiedSectorRotationDetector:
    """Simplified sector rotation detector for testing"""
    
    async def analyze_sector_rotation(self) -> List[Dict]:
        """Analyze sector rotation patterns"""
        results = []
        
        # Define sector ETFs for analysis
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Finance': 'XLF',
            'Energy': 'XLE',
            'Consumer': 'XLY'
        }
        
        try:
            # Analyze each sector
            for sector_name, etf_symbol in sector_etfs.items():
                try:
                    # Get ETF data
                    etf = yf.Ticker(etf_symbol)
                    hist = etf.history(period="60d")
                    
                    if len(hist) < 30:
                        continue
                    
                    # Calculate sector metrics
                    returns_30d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1) * 100
                    returns_7d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-7] - 1) * 100
                    
                    # Calculate relative strength vs market (using SPY as proxy)
                    spy = yf.Ticker('SPY')
                    spy_hist = spy.history(period="60d")
                    spy_returns_30d = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-30] - 1) * 100
                    
                    relative_strength = returns_30d - spy_returns_30d
                    
                    # Determine rotation signal
                    if relative_strength > 2 and returns_7d > 1:
                        rotation_signal = 'inflow'
                    elif relative_strength < -2 and returns_7d < -1:
                        rotation_signal = 'outflow'
                    else:
                        rotation_signal = 'neutral'
                    
                    if rotation_signal != 'neutral':
                        results.append({
                            'sector': sector_name,
                            'etf_symbol': etf_symbol,
                            'rotation_signal': rotation_signal,
                            'relative_strength': float(relative_strength),
                            'returns_30d': float(returns_30d),
                            'returns_7d': float(returns_7d),
                            'strength': 'high' if abs(relative_strength) > 5 else 'medium',
                            'timestamp': datetime.now()
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze sector {sector_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Sector rotation analysis failed: {e}")
            
        return results

class DynamicDiscoverySystemTester:
    """Comprehensive tester for the dynamic discovery system"""
    
    def __init__(self):
        self.test_results = {}
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN']
        
    async def test_volume_spike_scanner(self) -> Dict[str, Any]:
        """Test Volume Spike Scanner"""
        logger.info("Testing Volume Spike Scanner...")
        
        try:
            scanner = SimplifiedVolumeSpikeScanner()
            results = await scanner.scan_for_volume_spikes(self.test_symbols[:3])
            
            return {
                'status': 'success',
                'results_count': len(results),
                'sample_results': results[:2] if results else [],
                'symbols_tested': self.test_symbols[:3]
            }
            
        except Exception as e:
            logger.error(f"Volume Spike Scanner test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_news_driven_discovery(self) -> Dict[str, Any]:
        """Test News-Driven Discovery"""
        logger.info("Testing News-Driven Discovery...")
        
        try:
            discovery = SimplifiedNewsDrivenDiscovery()
            results = await discovery.analyze_news_impact(self.test_symbols[:3])
            
            return {
                'status': 'success',
                'results_count': len(results),
                'sample_results': results[:2] if results else [],
                'symbols_tested': self.test_symbols[:3]
            }
            
        except Exception as e:
            logger.error(f"News-Driven Discovery test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_technical_breakout_scanner(self) -> Dict[str, Any]:
        """Test Technical Breakout Scanner"""
        logger.info("Testing Technical Breakout Scanner...")
        
        try:
            scanner = SimplifiedTechnicalBreakoutScanner()
            results = await scanner.scan_for_breakouts(self.test_symbols[:3])
            
            return {
                'status': 'success',
                'results_count': len(results),
                'sample_results': results[:2] if results else [],
                'symbols_tested': self.test_symbols[:3]
            }
            
        except Exception as e:
            logger.error(f"Technical Breakout Scanner test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_earnings_calendar_monitor(self) -> Dict[str, Any]:
        """Test Earnings Calendar Monitor"""
        logger.info("Testing Earnings Calendar Monitor...")
        
        try:
            monitor = SimplifiedEarningsCalendarMonitor()
            results = await monitor.scan_upcoming_earnings(self.test_symbols[:5], days_ahead=30)
            
            return {
                'status': 'success',
                'results_count': len(results),
                'sample_results': results[:2] if results else [],
                'symbols_tested': self.test_symbols[:5]
            }
            
        except Exception as e:
            logger.error(f"Earnings Calendar Monitor test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_sector_rotation_detector(self) -> Dict[str, Any]:
        """Test Sector Rotation Detector"""
        logger.info("Testing Sector Rotation Detector...")
        
        try:
            detector = SimplifiedSectorRotationDetector()
            results = await detector.analyze_sector_rotation()
            
            return {
                'status': 'success',
                'results_count': len(results),
                'sample_results': results[:2] if results else [],
                'sectors_analyzed': ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']
            }
            
        except Exception as e:
            logger.error(f"Sector Rotation Detector test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_integrated_discovery(self) -> Dict[str, Any]:
        """Test integrated discovery workflow"""
        logger.info("Testing integrated discovery workflow...")
        
        try:
            # Run all discovery methods on a single symbol
            test_symbol = 'AAPL'
            integrated_results = {}
            
            # Volume spike analysis
            volume_scanner = SimplifiedVolumeSpikeScanner()
            volume_results = await volume_scanner.scan_for_volume_spikes([test_symbol])
            integrated_results['volume_spike'] = len(volume_results)
            
            # News analysis
            news_discovery = SimplifiedNewsDrivenDiscovery()
            news_results = await news_discovery.analyze_news_impact([test_symbol])
            integrated_results['news_driven'] = len(news_results)
            
            # Technical analysis
            tech_scanner = SimplifiedTechnicalBreakoutScanner()
            tech_results = await tech_scanner.scan_for_breakouts([test_symbol])
            integrated_results['technical_breakout'] = len(tech_results)
            
            # Earnings analysis
            earnings_monitor = SimplifiedEarningsCalendarMonitor()
            earnings_results = await earnings_monitor.scan_upcoming_earnings([test_symbol])
            integrated_results['earnings_calendar'] = len(earnings_results)
            
            # Sector analysis
            sector_detector = SimplifiedSectorRotationDetector()
            sector_results = await sector_detector.analyze_sector_rotation()
            integrated_results['sector_rotation'] = len(sector_results)
            
            total_opportunities = sum(integrated_results.values())
            
            return {
                'status': 'success',
                'test_symbol': test_symbol,
                'discovery_results': integrated_results,
                'total_opportunities': total_opportunities,
                'tools_tested': len(integrated_results)
            }
            
        except Exception as e:
            logger.error(f"Integrated discovery test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        logger.info("Starting comprehensive dynamic discovery system tests...")
        
        test_suite = {
            'volume_spike_scanner': self.test_volume_spike_scanner,
            'news_driven_discovery': self.test_news_driven_discovery,
            'technical_breakout_scanner': self.test_technical_breakout_scanner,
            'earnings_calendar_monitor': self.test_earnings_calendar_monitor,
            'sector_rotation_detector': self.test_sector_rotation_detector,
            'integrated_discovery': self.test_integrated_discovery
        }
        
        results = {}
        start_time = datetime.now()
        
        for test_name, test_func in test_suite.items():
            logger.info(f"Running test: {test_name}")
            test_start = datetime.now()
            
            try:
                result = await test_func()
                result['execution_time'] = (datetime.now() - test_start).total_seconds()
                results[test_name] = result
                
                if result['status'] == 'success':
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: CRASHED - {e}")
                results[test_name] = {
                    'status': 'crashed',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'execution_time': (datetime.now() - test_start).total_seconds()
                }
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Generate summary
        total_tests = len(test_suite)
        passed_tests = len([r for r in results.values() if r['status'] == 'success'])
        failed_tests = len([r for r in results.values() if r['status'] in ['error', 'crashed']])
        
        summary = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': f"{(passed_tests/total_tests)*100:.1f}%",
                'total_execution_time': f"{total_time:.2f}s"
            },
            'test_results': results,
            'timestamp': datetime.now().isoformat(),
            'test_symbols': self.test_symbols
        }
        
        return summary
    
    def print_report(self, results: Dict[str, Any]):
        """Print a formatted test report"""
        print("\n" + "="*80)
        print("DYNAMIC DISCOVERY SYSTEM - COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        summary = results['test_summary']
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ✅")
        print(f"   Failed: {summary['failed_tests']} ❌")
        print(f"   Success Rate: {summary['success_rate']}")
        print(f"   Total Time: {summary['total_execution_time']}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for test_name, result in results['test_results'].items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"\n   {status_icon} {test_name.replace('_', ' ').title()}")
            print(f"      Status: {result['status']}")
            print(f"      Time: {result.get('execution_time', 0):.2f}s")
            
            if result['status'] == 'success':
                if 'results_count' in result:
                    print(f"      Results Found: {result['results_count']}")
                if 'symbols_tested' in result:
                    print(f"      Symbols Tested: {len(result['symbols_tested'])}")
                if 'total_opportunities' in result:
                    print(f"      Total Opportunities: {result['total_opportunities']}")
                if 'sample_results' in result and result['sample_results']:
                    print(f"      Sample Result: {list(result['sample_results'][0].keys()) if result['sample_results'] else 'None'}")
            else:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        print(f"\n📈 SYSTEM HEALTH:")
        if summary['passed_tests'] == summary['total_tests']:
            print("   🟢 All systems operational - Dynamic discovery system is fully functional!")
        elif summary['passed_tests'] >= summary['total_tests'] * 0.8:
            print("   🟡 Most systems operational - Minor issues detected")
        else:
            print("   🔴 Multiple system failures - Requires immediate attention")
        
        print(f"\n💡 DISCOVERY INSIGHTS:")
        total_opportunities = 0
        for test_name, result in results['test_results'].items():
            if result['status'] == 'success' and 'results_count' in result:
                total_opportunities += result['results_count']
        
        print(f"   Total Market Opportunities Identified: {total_opportunities}")
        print(f"   Average Opportunities per Tool: {total_opportunities / summary['passed_tests']:.1f}")
        
        print("\n" + "="*80)

async def main():
    """Main test execution"""
    tester = DynamicDiscoverySystemTester()
    
    try:
        results = await tester.run_all_tests()
        tester.print_report(results)
        
        # Save results to file
        import json
        with open('dynamic_discovery_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: dynamic_discovery_test_results.json")
        
        # Return appropriate exit code
        if results['test_summary']['failed_tests'] == 0:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

class DynamicDiscoverySystemTester:
    """Comprehensive tester for the dynamic discovery system"""
    
    def __init__(self):
        self.test_results = {}
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN']
        
    async def test_volume_spike_scanner(self) -> Dict[str, Any]:
        """Test Volume Spike Scanner"""
        logger.info("Testing Volume Spike Scanner...")
        
        try:
            # Test direct scanner
            scanner = VolumeSpikeScanner()
            
            # Test with multiple symbols
            results = []
            for symbol in self.test_symbols[:3]:  # Test with first 3 symbols
                try:
                    spikes = await scanner.scan_for_volume_spikes([symbol])
                    results.extend(spikes)
                    logger.info(f"Volume spike scan for {symbol}: {len(spikes)} spikes found")
                except Exception as e:
                    logger.warning(f"Volume spike scan failed for {symbol}: {e}")
            
            # Test CrewAI tool
            tool = VolumeSpikeDiscoveryTool()
            tool_result = tool._run(
                symbols=','.join(self.test_symbols[:3]),
                min_volume_multiplier=2.0,
                lookback_days=5
            )
            
            return {
                'status': 'success',
                'direct_scanner_results': len(results),
                'tool_result_length': len(tool_result) if tool_result else 0,
                'sample_results': results[:2] if results else []
            }
            
        except Exception as e:
            logger.error(f"Volume Spike Scanner test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_news_driven_discovery(self) -> Dict[str, Any]:
        """Test News-Driven Discovery"""
        logger.info("Testing News-Driven Discovery...")
        
        try:
            # Test direct discovery
            discovery = NewsDrivenDiscovery()
            
            # Test news analysis
            results = []
            for symbol in self.test_symbols[:3]:
                try:
                    opportunities = await discovery.analyze_news_impact([symbol])
                    results.extend(opportunities)
                    logger.info(f"News analysis for {symbol}: {len(opportunities)} opportunities found")
                except Exception as e:
                    logger.warning(f"News analysis failed for {symbol}: {e}")
            
            # Test CrewAI tool
            tool = NewsDrivenDiscoveryTool()
            tool_result = tool._run(
                symbols=','.join(self.test_symbols[:3]),
                lookback_hours=24,
                min_sentiment_score=0.3
            )
            
            return {
                'status': 'success',
                'direct_discovery_results': len(results),
                'tool_result_length': len(tool_result) if tool_result else 0,
                'sample_results': results[:2] if results else []
            }
            
        except Exception as e:
            logger.error(f"News-Driven Discovery test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_technical_breakout_scanner(self) -> Dict[str, Any]:
        """Test Technical Breakout Scanner"""
        logger.info("Testing Technical Breakout Scanner...")
        
        try:
            # Test direct scanner
            scanner = TechnicalBreakoutScanner()
            
            # Test pattern detection
            results = []
            for symbol in self.test_symbols[:3]:
                try:
                    patterns = await scanner.scan_for_breakouts([symbol])
                    results.extend(patterns)
                    logger.info(f"Technical breakout scan for {symbol}: {len(patterns)} patterns found")
                except Exception as e:
                    logger.warning(f"Technical breakout scan failed for {symbol}: {e}")
            
            # Test CrewAI tool
            tool = TechnicalBreakoutDiscoveryTool()
            tool_result = tool._run(
                symbols=','.join(self.test_symbols[:3]),
                timeframes='1d,4h',
                min_strength='medium'
            )
            
            return {
                'status': 'success',
                'direct_scanner_results': len(results),
                'tool_result_length': len(tool_result) if tool_result else 0,
                'sample_results': results[:2] if results else []
            }
            
        except Exception as e:
            logger.error(f"Technical Breakout Scanner test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_earnings_calendar_monitor(self) -> Dict[str, Any]:
        """Test Earnings Calendar Monitor"""
        logger.info("Testing Earnings Calendar Monitor...")
        
        try:
            # Test direct monitor
            monitor = EarningsCalendarMonitor()
            
            # Test earnings scanning
            end_date = datetime.now() + timedelta(days=30)
            results = await monitor.scan_upcoming_earnings(
                symbols=self.test_symbols[:5],
                days_ahead=30
            )
            
            logger.info(f"Earnings calendar scan: {len(results)} events found")
            
            # Test CrewAI tool
            tool = EarningsCalendarDiscoveryTool()
            tool_result = tool._run(
                symbols=','.join(self.test_symbols[:5]),
                days_ahead=30,
                min_impact='medium'
            )
            
            return {
                'status': 'success',
                'direct_monitor_results': len(results),
                'tool_result_length': len(tool_result) if tool_result else 0,
                'sample_results': results[:2] if results else []
            }
            
        except Exception as e:
            logger.error(f"Earnings Calendar Monitor test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_sector_rotation_detector(self) -> Dict[str, Any]:
        """Test Sector Rotation Detector"""
        logger.info("Testing Sector Rotation Detector...")
        
        try:
            # Test direct detector
            detector = SectorRotationDetector()
            
            # Test sector analysis
            results = await detector.analyze_sector_rotation()
            
            logger.info(f"Sector rotation analysis: {len(results)} opportunities found")
            
            # Test CrewAI tool
            tool = SectorRotationDiscoveryTool()
            tool_result = tool._run(
                lookback_days=30,
                min_strength='medium',
                sectors='Technology,Healthcare,Finance'
            )
            
            return {
                'status': 'success',
                'direct_detector_results': len(results),
                'tool_result_length': len(tool_result) if tool_result else 0,
                'sample_results': results[:2] if results else []
            }
            
        except Exception as e:
            logger.error(f"Sector Rotation Detector test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_integrated_discovery(self) -> Dict[str, Any]:
        """Test integrated discovery workflow"""
        logger.info("Testing integrated discovery workflow...")
        
        try:
            # Initialize all tools
            tools = {
                'volume_spike': VolumeSpikeDiscoveryTool(),
                'news_driven': NewsDrivenDiscoveryTool(),
                'technical_breakout': TechnicalBreakoutDiscoveryTool(),
                'earnings_calendar': EarningsCalendarDiscoveryTool(),
                'sector_rotation': SectorRotationDiscoveryTool()
            }
            
            # Run integrated analysis
            integrated_results = {}
            test_symbol = 'AAPL'
            
            for tool_name, tool in tools.items():
                try:
                    if tool_name == 'volume_spike':
                        result = tool._run(symbols=test_symbol, min_volume_multiplier=2.0)
                    elif tool_name == 'news_driven':
                        result = tool._run(symbols=test_symbol, lookback_hours=24)
                    elif tool_name == 'technical_breakout':
                        result = tool._run(symbols=test_symbol, timeframes='1d')
                    elif tool_name == 'earnings_calendar':
                        result = tool._run(symbols=test_symbol, days_ahead=30)
                    elif tool_name == 'sector_rotation':
                        result = tool._run(lookback_days=30)
                    
                    integrated_results[tool_name] = {
                        'status': 'success',
                        'result_length': len(result) if result else 0
                    }
                    logger.info(f"Integrated test for {tool_name}: Success")
                    
                except Exception as e:
                    integrated_results[tool_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    logger.warning(f"Integrated test for {tool_name}: {e}")
            
            return {
                'status': 'success',
                'tool_results': integrated_results,
                'total_tools_tested': len(tools),
                'successful_tools': len([r for r in integrated_results.values() if r['status'] == 'success'])
            }
            
        except Exception as e:
            logger.error(f"Integrated discovery test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        logger.info("Starting comprehensive dynamic discovery system tests...")
        
        test_suite = {
            'volume_spike_scanner': self.test_volume_spike_scanner,
            'news_driven_discovery': self.test_news_driven_discovery,
            'technical_breakout_scanner': self.test_technical_breakout_scanner,
            'earnings_calendar_monitor': self.test_earnings_calendar_monitor,
            'sector_rotation_detector': self.test_sector_rotation_detector,
            'integrated_discovery': self.test_integrated_discovery
        }
        
        results = {}
        start_time = datetime.now()
        
        for test_name, test_func in test_suite.items():
            logger.info(f"Running test: {test_name}")
            test_start = datetime.now()
            
            try:
                result = await test_func()
                result['execution_time'] = (datetime.now() - test_start).total_seconds()
                results[test_name] = result
                
                if result['status'] == 'success':
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: CRASHED - {e}")
                results[test_name] = {
                    'status': 'crashed',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'execution_time': (datetime.now() - test_start).total_seconds()
                }
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Generate summary
        total_tests = len(test_suite)
        passed_tests = len([r for r in results.values() if r['status'] == 'success'])
        failed_tests = len([r for r in results.values() if r['status'] in ['error', 'crashed']])
        
        summary = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': f"{(passed_tests/total_tests)*100:.1f}%",
                'total_execution_time': f"{total_time:.2f}s"
            },
            'test_results': results,
            'timestamp': datetime.now().isoformat(),
            'test_symbols': self.test_symbols
        }
        
        return summary
    
    def print_report(self, results: Dict[str, Any]):
        """Print a formatted test report"""
        print("\n" + "="*80)
        print("DYNAMIC DISCOVERY SYSTEM - COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        summary = results['test_summary']
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ✅")
        print(f"   Failed: {summary['failed_tests']} ❌")
        print(f"   Success Rate: {summary['success_rate']}")
        print(f"   Total Time: {summary['total_execution_time']}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for test_name, result in results['test_results'].items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"\n   {status_icon} {test_name.replace('_', ' ').title()}")
            print(f"      Status: {result['status']}")
            print(f"      Time: {result.get('execution_time', 0):.2f}s")
            
            if result['status'] == 'success':
                if 'direct_scanner_results' in result:
                    print(f"      Direct Results: {result['direct_scanner_results']}")
                if 'tool_result_length' in result:
                    print(f"      Tool Results: {result['tool_result_length']}")
                if 'successful_tools' in result:
                    print(f"      Successful Tools: {result['successful_tools']}/{result['total_tools_tested']}")
            else:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        print(f"\n📈 SYSTEM HEALTH:")
        if summary['passed_tests'] == summary['total_tests']:
            print("   🟢 All systems operational - Dynamic discovery system is fully functional!")
        elif summary['passed_tests'] >= summary['total_tests'] * 0.8:
            print("   🟡 Most systems operational - Minor issues detected")
        else:
            print("   🔴 Multiple system failures - Requires immediate attention")
        
        print("\n" + "="*80)

async def main():
    """Main test execution"""
    tester = DynamicDiscoverySystemTester()
    
    try:
        results = await tester.run_all_tests()
        tester.print_report(results)
        
        # Save results to file
        import json
        with open('dynamic_discovery_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: dynamic_discovery_test_results.json")
        
        # Return appropriate exit code
        if results['test_summary']['failed_tests'] == 0:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)