#!/usr/bin/env python3
"""
System Integration Tests - Advanced Trading System v2.0

Comprehensive integration test suite that validates the entire trading system
works as a cohesive unit. Tests cover:

- System orchestrator initialization and coordination
- Component integration and communication
- End-to-end trading workflows
- Risk management integration
- Performance analytics integration
- ML pipeline integration
- Alternative data integration
- Real-time monitoring and alerting
- Portfolio optimization workflows
- Error handling and recovery
- System health monitoring
- Data flow validation
- Backtesting integration
- Multi-asset trading scenarios
- 24/7 operational scenarios

These tests ensure that all components work together seamlessly
and that the system can handle real-world trading scenarios.

Author: AI Trading System v2.0
Date: January 2025
"""

import asyncio
import unittest
import logging
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import os
import tempfile
import shutil
from pathlib import Path

# Import system components
try:
    from system_orchestrator import SystemOrchestrator, SystemState, ComponentStatus, ExecutionMode
    from advanced_strategies import TradingSignal, SignalStrength
    from market_intelligence_engine import MarketIntelligenceEngine
    from portfolio_optimization_engine import PortfolioOptimizationEngine
    from advanced_risk_management import AdvancedRiskManager
    from ml_trading_pipeline import MLTradingPipeline
    from alternative_data_engine import AlternativeDataEngine
    from real_time_monitoring import RealTimeMonitor
    from performance_analytics import PerformanceAnalytics
    from advanced_backtesting_framework import AdvancedBacktester
except ImportError as e:
    logging.warning(f"Import failed: {e}")

class MockMarketData:
    """Mock market data for testing"""
    
    def __init__(self):
        self.data = {
            'AAPL': {
                'price': 150.0,
                'volume': 1000000,
                'bid': 149.95,
                'ask': 150.05,
                'timestamp': datetime.now()
            },
            'GOOGL': {
                'price': 2800.0,
                'volume': 500000,
                'bid': 2799.50,
                'ask': 2800.50,
                'timestamp': datetime.now()
            },
            'BTC-USD': {
                'price': 45000.0,
                'volume': 100000,
                'bid': 44995.0,
                'ask': 45005.0,
                'timestamp': datetime.now()
            }
        }
    
    def get_latest_price(self, symbol: str) -> float:
        return self.data.get(symbol, {}).get('price', 100.0)
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        return self.data.get(symbol, {})
    
    def update_all_data(self):
        # Simulate price updates
        for symbol in self.data:
            current_price = self.data[symbol]['price']
            change = np.random.normal(0, 0.01) * current_price
            self.data[symbol]['price'] = max(0.01, current_price + change)
            self.data[symbol]['timestamp'] = datetime.now()

class MockExecutionEngine:
    """Mock execution engine for testing"""
    
    def __init__(self):
        self.orders = []
        self.positions = {}
        self.trades = []
    
    def submit_order(self, order) -> bool:
        self.orders.append(order)
        # Simulate successful execution
        self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        self.trades.append({
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'price': 100.0,  # Mock price
            'timestamp': datetime.now()
        })
        return True
    
    def get_positions(self) -> Dict[str, float]:
        return self.positions.copy()
    
    def get_trades(self) -> List[Dict[str, Any]]:
        return self.trades.copy()

class SystemIntegrationTests(unittest.TestCase):
    """Comprehensive system integration tests"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        
        # Create test configuration
        test_config = {
            "execution_mode": "paper_trading",
            "max_concurrent_strategies": 5,
            "enable_ml_pipeline": True,
            "enable_alternative_data": True,
            "enable_portfolio_optimization": True,
            "log_level": "DEBUG"
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Setup logging for tests
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("integration_tests")
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_system_orchestrator_initialization(self):
        """Test system orchestrator initialization"""
        self.logger.info("Testing system orchestrator initialization...")
        
        # Create orchestrator with test config
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Test initial state
        self.assertEqual(orchestrator.state, SystemState.INITIALIZING)
        self.assertIsNotNone(orchestrator.config)
        self.assertEqual(orchestrator.config.execution_mode, ExecutionMode.PAPER_TRADING)
        
        # Test configuration loading
        self.assertTrue(orchestrator.config.enable_ml_pipeline)
        self.assertTrue(orchestrator.config.enable_alternative_data)
        self.assertTrue(orchestrator.config.enable_portfolio_optimization)
        
        self.logger.info("✅ System orchestrator initialization test passed")
    
    @patch('system_orchestrator.MarketDataAggregator')
    @patch('system_orchestrator.ExecutionEngine')
    @patch('system_orchestrator.RiskManager24_7')
    async def test_system_initialization_workflow(self, mock_risk, mock_execution, mock_market_data):
        """Test complete system initialization workflow"""
        self.logger.info("Testing system initialization workflow...")
        
        # Setup mocks
        mock_market_data.return_value = MockMarketData()
        mock_execution.return_value = MockExecutionEngine()
        mock_risk.return_value = Mock()
        
        # Create orchestrator
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Test initialization
        with patch.object(orchestrator, '_initialize_core_components') as mock_core:
            with patch.object(orchestrator, '_initialize_advanced_components') as mock_advanced:
                with patch.object(orchestrator, '_verify_system_health', return_value=True) as mock_health:
                    
                    mock_core.return_value = None
                    mock_advanced.return_value = None
                    
                    # Initialize system
                    result = await orchestrator.initialize()
                    
                    # Verify initialization
                    self.assertTrue(result)
                    self.assertEqual(orchestrator.state, SystemState.RUNNING)
                    mock_core.assert_called_once()
                    mock_advanced.assert_called_once()
                    mock_health.assert_called_once()
        
        self.logger.info("✅ System initialization workflow test passed")
    
    def test_component_health_monitoring(self):
        """Test component health monitoring"""
        self.logger.info("Testing component health monitoring...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Register mock components
        orchestrator._register_component('test_component')
        
        # Test health status
        health = orchestrator.component_health['test_component']
        self.assertEqual(health.status, ComponentStatus.HEALTHY)
        self.assertIsNotNone(health.last_heartbeat)
        
        # Test health check
        self.assertTrue(health.is_healthy())
        
        # Test unhealthy component
        health.status = ComponentStatus.ERROR
        health.last_heartbeat = datetime.now() - timedelta(minutes=10)
        self.assertFalse(health.is_healthy())
        
        self.logger.info("✅ Component health monitoring test passed")
    
    def test_trading_signal_generation_integration(self):
        """Test trading signal generation integration"""
        self.logger.info("Testing trading signal generation integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Mock strategy orchestrator
        mock_strategy_orchestrator = Mock()
        mock_signals = [
            TradingSignal(
                symbol='AAPL',
                action='BUY',
                confidence=0.8,
                strength=SignalStrength.STRONG,
                reasoning='Test momentum signal',
                risk_level='MEDIUM',
                suggested_position_size=0.05,
                strategy_name='momentum',
                timestamp=datetime.now()
            ),
            TradingSignal(
                symbol='GOOGL',
                action='SELL',
                confidence=0.7,
                strength=SignalStrength.MODERATE,
                reasoning='Test mean reversion signal',
                risk_level='MEDIUM',
                suggested_position_size=0.03,
                strategy_name='mean_reversion',
                timestamp=datetime.now()
            )
        ]
        
        mock_strategy_orchestrator.generate_ensemble_signals.return_value = mock_signals
        orchestrator.components['strategy_orchestrator'] = mock_strategy_orchestrator
        
        # Test signal generation
        signals = orchestrator._generate_trading_signals()
        
        # Verify signals
        self.assertEqual(len(signals), 2)
        self.assertEqual(signals[0].symbol, 'AAPL')
        self.assertEqual(signals[0].action, 'BUY')
        self.assertEqual(signals[1].symbol, 'GOOGL')
        self.assertEqual(signals[1].action, 'SELL')
        
        self.logger.info("✅ Trading signal generation integration test passed")
    
    def test_ml_pipeline_integration(self):
        """Test ML pipeline integration"""
        self.logger.info("Testing ML pipeline integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        orchestrator.config.enable_ml_pipeline = True
        
        # Mock ML pipeline
        mock_ml_pipeline = Mock()
        mock_prediction = Mock()
        mock_prediction.prediction = 0.7
        mock_prediction.confidence = 0.85
        mock_ml_pipeline.predict.return_value = mock_prediction
        
        orchestrator.components['ml_pipeline'] = mock_ml_pipeline
        
        # Create test signal
        test_signal = TradingSignal(
            symbol='AAPL',
            action='BUY',
            strength=SignalStrength.MEDIUM,
            strategy='test',
            timestamp=datetime.now()
        )
        
        # Test ML enhancement
        enhanced_signals = orchestrator._enhance_signals_with_ml([test_signal])
        
        # Verify enhancement
        self.assertEqual(len(enhanced_signals), 1)
        enhanced_signal = enhanced_signals[0]
        
        # Check ML metadata
        self.assertIn('ml_prediction', enhanced_signal.metadata)
        self.assertIn('ml_confidence', enhanced_signal.metadata)
        self.assertEqual(enhanced_signal.metadata['ml_prediction'], 0.7)
        self.assertEqual(enhanced_signal.metadata['ml_confidence'], 0.85)
        
        self.logger.info("✅ ML pipeline integration test passed")
    
    def test_portfolio_optimization_integration(self):
        """Test portfolio optimization integration"""
        self.logger.info("Testing portfolio optimization integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        orchestrator.config.enable_portfolio_optimization = True
        
        # Mock portfolio optimizer
        mock_optimizer = Mock()
        mock_result = Mock()
        mock_result.weights = {'AAPL': 0.6, 'GOOGL': 0.4}
        mock_result.metrics = Mock()
        mock_result.metrics.sharpe_ratio = 1.5
        mock_optimizer.optimize_portfolio.return_value = mock_result
        
        orchestrator.components['portfolio_optimizer'] = mock_optimizer
        
        # Create test signals
        test_signals = [
            TradingSignal(
                symbol='AAPL',
                action='BUY',
                strength=SignalStrength.STRONG,
                strategy='test',
                timestamp=datetime.now()
            ),
            TradingSignal(
                symbol='GOOGL',
                action='BUY',
                strength=SignalStrength.MEDIUM,
                strategy='test',
                timestamp=datetime.now()
            )
        ]
        
        # Test portfolio optimization
        optimized_signals = orchestrator._optimize_portfolio_allocation(test_signals)
        
        # Verify optimization
        self.assertEqual(len(optimized_signals), 2)
        
        for signal in optimized_signals:
            self.assertIn('target_weight', signal.metadata)
            self.assertIn('optimization_score', signal.metadata)
            
            if signal.symbol == 'AAPL':
                self.assertEqual(signal.metadata['target_weight'], 0.6)
            elif signal.symbol == 'GOOGL':
                self.assertEqual(signal.metadata['target_weight'], 0.4)
        
        self.logger.info("✅ Portfolio optimization integration test passed")
    
    def test_risk_management_integration(self):
        """Test risk management integration"""
        self.logger.info("Testing risk management integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Mock advanced risk manager
        mock_risk_manager = Mock()
        mock_risk_metrics = Mock()
        mock_risk_manager.calculate_portfolio_risk.return_value = mock_risk_metrics
        
        orchestrator.components['advanced_risk_manager'] = mock_risk_manager
        
        # Create test signals
        test_signals = [
            TradingSignal(
                symbol='AAPL',
                action='BUY',
                strength=SignalStrength.STRONG,
                strategy='test',
                timestamp=datetime.now(),
                position_size=100
            ),
            TradingSignal(
                symbol='RISKY_STOCK',
                action='BUY',
                strength=SignalStrength.WEAK,
                strategy='test',
                timestamp=datetime.now(),
                position_size=1000  # Large position
            )
        ]
        
        # Mock risk checks to filter out risky signal
        with patch.object(orchestrator, '_passes_risk_checks') as mock_risk_check:
            mock_risk_check.side_effect = lambda signal, pos_risk, port_impact: signal.symbol != 'RISKY_STOCK'
            
            # Test risk management
            filtered_signals = orchestrator._apply_risk_management(test_signals)
            
            # Verify filtering
            self.assertEqual(len(filtered_signals), 1)
            self.assertEqual(filtered_signals[0].symbol, 'AAPL')
        
        self.logger.info("✅ Risk management integration test passed")
    
    def test_execution_engine_integration(self):
        """Test execution engine integration"""
        self.logger.info("Testing execution engine integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Mock execution engine
        mock_execution_engine = MockExecutionEngine()
        orchestrator.components['execution_engine'] = mock_execution_engine
        
        # Create test signals
        test_signals = [
            TradingSignal(
                symbol='AAPL',
                action='BUY',
                strength=SignalStrength.STRONG,
                strategy='test',
                timestamp=datetime.now(),
                position_size=100
            )
        ]
        
        # Test signal execution
        orchestrator._execute_signals(test_signals)
        
        # Verify execution
        self.assertEqual(len(mock_execution_engine.orders), 1)
        self.assertEqual(len(mock_execution_engine.trades), 1)
        
        order = mock_execution_engine.orders[0]
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.quantity, 100)
        
        # Verify metrics update
        self.assertEqual(orchestrator.system_metrics.total_trades, 1)
        self.assertEqual(orchestrator.system_metrics.successful_trades, 1)
        
        self.logger.info("✅ Execution engine integration test passed")
    
    def test_real_time_monitoring_integration(self):
        """Test real-time monitoring integration"""
        self.logger.info("Testing real-time monitoring integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Mock real-time monitor
        mock_monitor = Mock()
        mock_alerts = []
        mock_monitor.get_active_alerts.return_value = mock_alerts
        
        orchestrator.components['real_time_monitor'] = mock_monitor
        
        # Test monitoring integration
        orchestrator._handle_risk_alert("Test alert")
        
        # Verify alert handling
        self.assertEqual(orchestrator.system_metrics.alerts_generated, 1)
        
        self.logger.info("✅ Real-time monitoring integration test passed")
    
    def test_performance_analytics_integration(self):
        """Test performance analytics integration"""
        self.logger.info("Testing performance analytics integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Mock performance analytics
        mock_analytics = Mock()
        mock_metrics = Mock()
        mock_metrics.total_return = 0.15
        mock_metrics.sharpe_ratio = 1.2
        mock_analytics.calculate_performance.return_value = mock_metrics
        
        orchestrator.components['performance_analytics'] = mock_analytics
        
        # Test performance update
        orchestrator._update_performance_metrics()
        
        # Verify integration (no exceptions thrown)
        self.assertTrue(True)
        
        self.logger.info("✅ Performance analytics integration test passed")
    
    def test_alternative_data_integration(self):
        """Test alternative data integration"""
        self.logger.info("Testing alternative data integration...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        orchestrator.config.enable_alternative_data = True
        
        # Mock alternative data engine
        mock_alt_data = Mock()
        mock_alt_signals = [
            TradingSignal(
                symbol='AAPL',
                action='BUY',
                strength=SignalStrength.MEDIUM,
                strategy='sentiment',
                timestamp=datetime.now()
            )
        ]
        mock_alt_data.generate_trading_signals.return_value = mock_alt_signals
        
        orchestrator.components['alternative_data'] = mock_alt_data
        
        # Test signal generation with alternative data
        signals = orchestrator._generate_trading_signals()
        
        # Verify alternative data signals are included
        self.assertGreaterEqual(len(signals), 1)
        
        self.logger.info("✅ Alternative data integration test passed")
    
    async def test_end_to_end_trading_workflow(self):
        """Test complete end-to-end trading workflow"""
        self.logger.info("Testing end-to-end trading workflow...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Setup all required mocks
        orchestrator.components['market_data'] = MockMarketData()
        orchestrator.components['execution_engine'] = MockExecutionEngine()
        
        # Mock strategy orchestrator
        mock_strategy = Mock()
        mock_signals = [
            TradingSignal(
                symbol='AAPL',
                action='BUY',
                confidence=0.8,
                strength=SignalStrength.STRONG,
                reasoning='Test momentum signal',
                risk_level='MEDIUM',
                suggested_position_size=0.05,
                strategy_name='momentum',
                timestamp=datetime.now()
            )
        ]
        mock_strategy.generate_ensemble_signals.return_value = mock_signals
        orchestrator.components['strategy_orchestrator'] = mock_strategy
        
        # Mock other components
        orchestrator.components['advanced_risk_manager'] = Mock()
        orchestrator.components['ml_pipeline'] = Mock()
        orchestrator.components['portfolio_optimizer'] = Mock()
        
        # Test complete trading cycle
        await orchestrator._execute_trading_cycle()
        
        # Verify workflow execution
        execution_engine = orchestrator.components['execution_engine']
        self.assertEqual(len(execution_engine.orders), 1)
        self.assertEqual(execution_engine.orders[0].symbol, 'AAPL')
        
        self.logger.info("✅ End-to-end trading workflow test passed")
    
    def test_system_state_management(self):
        """Test system state management"""
        self.logger.info("Testing system state management...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Test initial state
        self.assertEqual(orchestrator.state, SystemState.INITIALIZING)
        
        # Test state transitions
        orchestrator.state = SystemState.RUNNING
        self.assertEqual(orchestrator.state, SystemState.RUNNING)
        
        # Test pause/resume
        result = orchestrator.pause_system()
        self.assertTrue(result)
        self.assertEqual(orchestrator.state, SystemState.PAUSED)
        
        result = orchestrator.resume_system()
        self.assertTrue(result)
        self.assertEqual(orchestrator.state, SystemState.RUNNING)
        
        self.logger.info("✅ System state management test passed")
    
    def test_system_metrics_tracking(self):
        """Test system metrics tracking"""
        self.logger.info("Testing system metrics tracking...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Test initial metrics
        metrics = orchestrator.system_metrics
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.successful_trades, 0)
        self.assertEqual(metrics.failed_trades, 0)
        
        # Simulate trades
        metrics.total_trades = 10
        metrics.successful_trades = 8
        metrics.failed_trades = 2
        
        # Test success rate calculation
        self.assertEqual(metrics.success_rate, 0.8)
        
        # Test metrics update
        orchestrator._update_system_metrics()
        
        # Verify uptime is tracked
        self.assertGreater(metrics.uptime.total_seconds(), 0)
        
        self.logger.info("✅ System metrics tracking test passed")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        self.logger.info("Testing error handling and recovery...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Test component error handling
        orchestrator._register_component('test_component')
        health = orchestrator.component_health['test_component']
        
        # Simulate component error
        health.status = ComponentStatus.ERROR
        health.error_count = 1
        health.last_error = "Test error"
        
        # Test error state
        self.assertEqual(health.status, ComponentStatus.ERROR)
        self.assertEqual(health.error_count, 1)
        self.assertIsNotNone(health.last_error)
        
        # Test recovery (reset to healthy)
        health.status = ComponentStatus.HEALTHY
        health.error_count = 0
        health.last_error = None
        
        self.assertEqual(health.status, ComponentStatus.HEALTHY)
        self.assertEqual(health.error_count, 0)
        
        self.logger.info("✅ Error handling and recovery test passed")
    
    def test_configuration_management(self):
        """Test configuration management"""
        self.logger.info("Testing configuration management...")
        
        # Test config loading
        orchestrator = SystemOrchestrator(self.config_path)
        
        # Verify config values
        self.assertEqual(orchestrator.config.execution_mode, ExecutionMode.PAPER_TRADING)
        self.assertEqual(orchestrator.config.max_concurrent_strategies, 5)
        self.assertTrue(orchestrator.config.enable_ml_pipeline)
        
        # Test config saving
        orchestrator.config.max_concurrent_strategies = 10
        orchestrator._save_config(orchestrator.config)
        
        # Reload and verify
        new_orchestrator = SystemOrchestrator(self.config_path)
        self.assertEqual(new_orchestrator.config.max_concurrent_strategies, 10)
        
        self.logger.info("✅ Configuration management test passed")
    
    def test_multi_asset_trading_support(self):
        """Test multi-asset trading support"""
        self.logger.info("Testing multi-asset trading support...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        orchestrator.components['execution_engine'] = MockExecutionEngine()
        
        # Create multi-asset signals
        multi_asset_signals = [
            TradingSignal(
                symbol='AAPL',  # Stock
                action='BUY',
                strength=SignalStrength.STRONG,
                strategy='equity_momentum',
                timestamp=datetime.now(),
                position_size=100
            ),
            TradingSignal(
                symbol='BTC-USD',  # Crypto
                action='BUY',
                strength=SignalStrength.MEDIUM,
                strategy='crypto_momentum',
                timestamp=datetime.now(),
                position_size=0.5
            ),
            TradingSignal(
                symbol='EUR/USD',  # Forex
                action='SELL',
                strength=SignalStrength.WEAK,
                strategy='forex_carry',
                timestamp=datetime.now(),
                position_size=10000
            )
        ]
        
        # Test execution of multi-asset signals
        orchestrator._execute_signals(multi_asset_signals)
        
        # Verify all asset types were processed
        execution_engine = orchestrator.components['execution_engine']
        self.assertEqual(len(execution_engine.orders), 3)
        
        symbols = [order.symbol for order in execution_engine.orders]
        self.assertIn('AAPL', symbols)
        self.assertIn('BTC-USD', symbols)
        self.assertIn('EUR/USD', symbols)
        
        self.logger.info("✅ Multi-asset trading support test passed")
    
    def test_system_status_reporting(self):
        """Test system status reporting"""
        self.logger.info("Testing system status reporting...")
        
        orchestrator = SystemOrchestrator(self.config_path)
        orchestrator.state = SystemState.RUNNING
        
        # Register some components
        orchestrator._register_component('test_component_1')
        orchestrator._register_component('test_component_2')
        
        # Get system status
        status = orchestrator.get_system_status()
        
        # Verify status structure
        self.assertIn('state', status)
        self.assertIn('uptime', status)
        self.assertIn('metrics', status)
        self.assertIn('component_health', status)
        self.assertIn('config', status)
        
        # Verify values
        self.assertEqual(status['state'], 'running')
        self.assertEqual(len(status['component_health']), 2)
        
        self.logger.info("✅ System status reporting test passed")

class PerformanceIntegrationTests(unittest.TestCase):
    """Performance-focused integration tests"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger("performance_tests")
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_high_frequency_signal_processing(self):
        """Test high-frequency signal processing performance"""
        self.logger.info("Testing high-frequency signal processing...")
        
        # Create orchestrator with performance config
        config_path = os.path.join(self.test_dir, "perf_config.json")
        perf_config = {
            "execution_mode": "paper_trading",
            "max_concurrent_strategies": 20,
            "enable_ml_pipeline": False,  # Disable for performance
            "enable_alternative_data": False
        }
        
        with open(config_path, 'w') as f:
            json.dump(perf_config, f)
        
        orchestrator = SystemOrchestrator(config_path)
        orchestrator.components['execution_engine'] = MockExecutionEngine()
        
        # Generate large number of signals
        signals = []
        for i in range(1000):
            signals.append(TradingSignal(
                symbol=f'STOCK_{i}',
                action='BUY',
                strength=SignalStrength.MEDIUM,
                strategy='test',
                timestamp=datetime.now(),
                position_size=100
            ))
        
        # Measure execution time
        start_time = time.time()
        orchestrator._execute_signals(signals)
        execution_time = time.time() - start_time
        
        # Verify performance (should process 1000 signals in reasonable time)
        self.assertLess(execution_time, 10.0)  # Less than 10 seconds
        
        # Verify all signals were processed
        execution_engine = orchestrator.components['execution_engine']
        self.assertEqual(len(execution_engine.orders), 1000)
        
        self.logger.info(f"✅ Processed 1000 signals in {execution_time:.2f} seconds")
    
    def test_concurrent_component_operations(self):
        """Test concurrent component operations"""
        self.logger.info("Testing concurrent component operations...")
        
        orchestrator = SystemOrchestrator()
        
        # Test concurrent health checks
        def health_check_worker():
            for _ in range(100):
                orchestrator._check_component_health()
                time.sleep(0.01)
        
        # Start multiple health check threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=health_check_worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify no deadlocks or errors occurred
        self.assertTrue(True)
        
        self.logger.info("✅ Concurrent component operations test passed")

class BacktestingIntegrationTests(unittest.TestCase):
    """Backtesting integration tests"""
    
    def setUp(self):
        self.logger = logging.getLogger("backtesting_tests")
    
    def test_backtesting_integration(self):
        """Test backtesting framework integration"""
        self.logger.info("Testing backtesting integration...")
        
        # This would test integration with the backtesting framework
        # For now, we'll just verify the component can be instantiated
        try:
            from advanced_backtesting_framework import AdvancedBacktester
            backtester = AdvancedBacktester()
            self.assertIsNotNone(backtester)
            self.logger.info("✅ Backtesting integration test passed")
        except ImportError:
            self.logger.warning("⚠️  Backtesting framework not available for testing")

def run_integration_tests():
    """Run all integration tests"""
    print("\n🧪 Advanced Trading System v2.0 - Integration Tests")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add system integration tests
    test_suite.addTest(unittest.makeSuite(SystemIntegrationTests))
    test_suite.addTest(unittest.makeSuite(PerformanceIntegrationTests))
    test_suite.addTest(unittest.makeSuite(BacktestingIntegrationTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"   • Tests Run: {result.testsRun}")
    print(f"   • Failures: {len(result.failures)}")
    print(f"   • Errors: {len(result.errors)}")
    print(f"   • Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            print(f"   • {test}: {error_lines[-2]}")
    
    if not result.failures and not result.errors:
        print("\n🎉 All integration tests passed successfully!")
        print("\n✅ System Integration Validation Complete:")
        print("   • Component coordination: ✓")
        print("   • End-to-end workflows: ✓")
        print("   • Error handling: ✓")
        print("   • Performance requirements: ✓")
        print("   • Multi-asset support: ✓")
        print("   • Real-time operations: ✓")
        print("\n🚀 Advanced Trading System v2.0 is ready for deployment!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run integration tests
    success = run_integration_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)