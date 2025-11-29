#!/usr/bin/env python3
"""
Test script for batch processing optimizations in discovery tools
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import optimized discovery tools
from tools.volume_spike_scanner import VolumeSpikeScanner
from tools.sector_rotation_detector import SectorRotationDetector
from tools.technical_breakout_scanner import TechnicalBreakoutScanner
from tools.earnings_calendar_monitor import EarningsCalendarMonitor
from core.config_manager import ConfigManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchOptimizationTester:
    """Test suite for batch processing optimizations"""
    
    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
    def test_volume_spike_scanner(self):
        """Test Volume Spike Scanner batch processing."""
        try:
            config_manager = ConfigManager(Path('config'))
            scanner = VolumeSpikeScanner(config_manager)
            
            # Test scanning functionality (async method)
            spikes = asyncio.run(scanner.scan_for_volume_spikes())
            
            # Get metrics
            metrics = scanner.get_scanner_metrics()
            
            self.logger.info(f"Volume Spike Scanner - Spikes found: {len(spikes)}")
            self.logger.info(f"Batch metrics: {metrics.get('batch_processing', {})}")
            
            return True, f"Found {len(spikes)} volume spikes"
            
        except Exception as e:
            return False, str(e)
    
    def test_sector_rotation_detector(self):
        """Test Sector Rotation Detector batch processing."""
        try:
            config_manager = ConfigManager(Path('config'))
            detector = SectorRotationDetector(config_manager)
            
            # Test rotation analysis (async method)
            rotation_data = asyncio.run(detector.detect_sector_rotation())
            
            # Get metrics
            metrics = detector.get_detector_metrics()
            
            sectors_analyzed = len(rotation_data.get('sector_performance', {}))
            
            self.logger.info(f"Sector Rotation Detector - Sectors analyzed: {sectors_analyzed}")
            self.logger.info(f"Batch metrics: {metrics.get('batch_processing', {})}")
            
            return True, f"Analyzed {sectors_analyzed} sectors"
            
        except Exception as e:
            return False, str(e)
    
    def test_technical_breakout_scanner(self):
        """Test Technical Breakout Scanner batch processing."""
        try:
            config_manager = ConfigManager(Path('config'))
            scanner = TechnicalBreakoutScanner(config_manager)
            
            # Test pattern scanning (async method)
            patterns = asyncio.run(scanner.scan_for_patterns())
            
            # Get metrics
            metrics = scanner.get_scanner_metrics()
            
            self.logger.info(f"Technical Breakout Scanner - Patterns found: {len(patterns)}")
            self.logger.info(f"Batch metrics: {metrics.get('batch_processing', {})}")
            
            return True, f"Found {len(patterns)} patterns"
            
        except Exception as e:
            return False, str(e)
    
    def test_earnings_calendar_monitor(self):
        """Test Earnings Calendar Monitor batch processing."""
        try:
            config_manager = ConfigManager(Path('config'))
            monitor = EarningsCalendarMonitor(config_manager)
            
            # Test earnings monitoring (async method)
            earnings_data = asyncio.run(monitor.monitor_earnings_calendar(self.test_symbols))
            
            # Get metrics
            metrics = monitor.get_monitor_metrics()
            
            earnings_analyzed = len(earnings_data.get('upcoming_earnings', []))
            
            self.logger.info(f"Earnings Calendar Monitor - Earnings analyzed: {earnings_analyzed}")
            self.logger.info(f"Batch metrics: {metrics.get('batch_processing', {})}")
            
            return True, f"Analyzed {earnings_analyzed} earnings events"
            
        except Exception as e:
            return False, str(e)
    
    def run_all_tests(self):
        """Run all batch optimization tests."""
        test_id = f"batch_opt_test_{int(datetime.now().timestamp())}"
        start_time = datetime.now()
        
        self.logger.info(f"Starting batch optimization tests - ID: {test_id}")
        
        tests = [
            ("Volume Spike Scanner", self.test_volume_spike_scanner),
            ("Sector Rotation Detector", self.test_sector_rotation_detector),
            ("Technical Breakout Scanner", self.test_technical_breakout_scanner),
            ("Earnings Calendar Monitor", self.test_earnings_calendar_monitor)
        ]
        
        results = {}
        passed_tests = 0
        
        for test_name, test_func in tests:
            self.logger.info(f"Testing {test_name}...")
            try:
                success, message = test_func()
                if success:
                    results[test_name] = {"status": "passed", "message": message}
                    passed_tests += 1
                    self.logger.info(f"✅ {test_name}: {message}")
                else:
                    results[test_name] = {"status": "failed", "error": message}
                    self.logger.error(f"❌ {test_name}: {message}")
            except Exception as e:
                results[test_name] = {"status": "failed", "error": str(e)}
                self.logger.error(f"❌ {test_name} test failed: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "test_id": test_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "tests_passed": passed_tests,
            "total_tests": len(tests),
            "success_rate": (passed_tests / len(tests)) * 100,
            "results": results
        }
        
        self.logger.info(f"Batch optimization tests completed: {passed_tests}/{len(tests)} passed")
        
        return summary
    
    def print_test_summary(self, summary: Dict[str, Any]):
        """Print a formatted test summary."""
        print("\n" + "="*80)
        print("BATCH PROCESSING OPTIMIZATION TEST RESULTS")
        print("="*80)
        print(f"Test Run ID: {summary['test_id']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Tests Passed: {summary['tests_passed']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        print("\n" + "-"*80)
        print("INDIVIDUAL TEST RESULTS:")
        print("-"*80)
        
        for test_name, result in summary['results'].items():
            if result['status'] == 'passed':
                print(f"✅ {test_name}")
                print(f"   {result['message']}")
            else:
                print(f"❌ {test_name}")
                print(f"   Error: {result['error']}")
        
        print("\n" + "="*80)
        
        # Save detailed results to file
        results_file = f"batch_optimization_test_results_{summary['test_id'].split('_')[-1]}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        if summary['tests_passed'] < summary['total_tests']:
            print(f"\n⚠️  {summary['total_tests'] - summary['tests_passed']} test(s) failed")
        else:
            print(f"\n🎉 All tests passed!")

def main():
    """Main test execution function."""
    tester = BatchOptimizationTester()
    
    try:
        summary = tester.run_all_tests()
        tester.print_test_summary(summary)
        
        # Return exit code based on test results
        return 0 if summary['tests_passed'] == summary['total_tests'] else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)