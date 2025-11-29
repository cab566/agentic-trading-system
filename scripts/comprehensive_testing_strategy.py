#!/usr/bin/env python3
"""
Comprehensive Testing Strategy for AI Trading System

Implements multi-layered testing approach:
- Unit tests for individual components
- Integration tests for system interactions
- Performance tests for scalability
- Security tests for vulnerabilities
- AI model validation tests
- End-to-end workflow tests
"""

import os
import sys
import asyncio
import pytest
import unittest
import subprocess
import json
import yaml
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil

# Testing frameworks
import pytest
from pytest_asyncio import pytest_asyncio_auto_mode
from pytest_benchmark import BenchmarkFixture
from pytest_mock import MockerFixture
import hypothesis
from hypothesis import strategies as st

# Rich for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree
from rich.text import Text
from rich.live import Live

# Security testing
import bandit
from safety import safety

# Performance testing
import psutil
import memory_profiler
from line_profiler import LineProfiler

console = Console()

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    category: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None

@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    description: str
    tests: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    parallel: bool = True
    critical: bool = False

@dataclass
class TestConfiguration:
    """Testing configuration"""
    project_root: Path
    test_environments: List[str] = field(default_factory=lambda: ["development", "staging"])
    coverage_threshold: float = 80.0
    performance_baseline: Dict[str, float] = field(default_factory=dict)
    security_level: str = "high"
    ai_model_accuracy_threshold: float = 0.85

class ComprehensiveTestingFramework:
    """Main testing framework class"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.results: List[TestResult] = []
        self.test_suites: Dict[str, TestSuite] = {}
        self.setup_logging()
        self.setup_test_suites()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config.project_root / "logs" / "testing"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "testing_framework.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_test_suites(self):
        """Setup test suite configurations"""
        self.test_suites = {
            "unit": TestSuite(
                name="Unit Tests",
                description="Test individual components in isolation",
                tests=[
                    "test_data_adapters",
                    "test_trading_agents", 
                    "test_risk_management",
                    "test_strategy_engine",
                    "test_portfolio_manager",
                    "test_market_data",
                    "test_order_execution"
                ],
                timeout=120,
                parallel=True,
                critical=True
            ),
            "integration": TestSuite(
                name="Integration Tests",
                description="Test component interactions and workflows",
                tests=[
                    "test_agent_communication",
                    "test_data_pipeline",
                    "test_trading_workflow",
                    "test_risk_integration",
                    "test_database_operations",
                    "test_external_apis",
                    "test_message_queues"
                ],
                dependencies=["unit"],
                timeout=300,
                parallel=False,
                critical=True
            ),
            "ai_models": TestSuite(
                name="AI Model Tests",
                description="Test AI model performance and accuracy",
                tests=[
                    "test_qlib_integration",
                    "test_openbb_integration", 
                    "test_rd_agent_integration",
                    "test_pearl_rl_integration",
                    "test_model_accuracy",
                    "test_model_performance",
                    "test_model_robustness"
                ],
                dependencies=["unit", "integration"],
                timeout=600,
                parallel=True,
                critical=True
            ),
            "performance": TestSuite(
                name="Performance Tests",
                description="Test system performance and scalability",
                tests=[
                    "test_latency_benchmarks",
                    "test_throughput_limits",
                    "test_memory_usage",
                    "test_cpu_utilization",
                    "test_database_performance",
                    "test_concurrent_users",
                    "test_load_balancing"
                ],
                dependencies=["unit", "integration"],
                timeout=900,
                parallel=True,
                critical=False
            ),
            "security": TestSuite(
                name="Security Tests",
                description="Test security vulnerabilities and compliance",
                tests=[
                    "test_authentication",
                    "test_authorization",
                    "test_data_encryption",
                    "test_api_security",
                    "test_sql_injection",
                    "test_xss_vulnerabilities",
                    "test_dependency_vulnerabilities"
                ],
                dependencies=["unit"],
                timeout=300,
                parallel=True,
                critical=True
            ),
            "end_to_end": TestSuite(
                name="End-to-End Tests",
                description="Test complete user workflows",
                tests=[
                    "test_user_registration",
                    "test_trading_session",
                    "test_portfolio_management",
                    "test_risk_monitoring",
                    "test_reporting_workflow",
                    "test_alert_system",
                    "test_backup_recovery"
                ],
                dependencies=["unit", "integration", "ai_models"],
                timeout=1200,
                parallel=False,
                critical=True
            )
        }

    async def run_unit_tests(self) -> List[TestResult]:
        """Run unit tests"""
        console.print("[blue]Running unit tests...[/blue]")
        results = []
        
        test_dir = self.config.project_root / "tests" / "unit"
        if not test_dir.exists():
            return [TestResult(
                test_name="unit_tests",
                category="unit",
                status="skipped",
                duration=0.0,
                message="Unit test directory not found"
            )]
        
        # Run pytest with coverage
        start_time = time.time()
        try:
            cmd = [
                "python", "-m", "pytest",
                str(test_dir),
                "--cov=core",
                "--cov=tools", 
                "--cov-report=json",
                "--cov-report=html",
                "--json-report",
                "--json-report-file=test_results_unit.json",
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                timeout=self.test_suites["unit"].timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Parse coverage results
                coverage_file = self.config.project_root / "coverage.json"
                coverage_data = {}
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                
                results.append(TestResult(
                    test_name="unit_tests",
                    category="unit",
                    status="passed" if total_coverage >= self.config.coverage_threshold else "failed",
                    duration=duration,
                    message=f"Coverage: {total_coverage:.1f}%",
                    details={"coverage": coverage_data},
                    metrics={"coverage_percent": total_coverage}
                ))
            else:
                results.append(TestResult(
                    test_name="unit_tests",
                    category="unit", 
                    status="failed",
                    duration=duration,
                    message=f"Tests failed: {result.stderr}",
                    details={"stdout": result.stdout, "stderr": result.stderr}
                ))
                
        except subprocess.TimeoutExpired:
            results.append(TestResult(
                test_name="unit_tests",
                category="unit",
                status="error",
                duration=self.test_suites["unit"].timeout,
                message="Tests timed out"
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="unit_tests",
                category="unit",
                status="error",
                duration=time.time() - start_time,
                message=f"Unexpected error: {e}"
            ))
        
        return results

    async def run_integration_tests(self) -> List[TestResult]:
        """Run integration tests"""
        console.print("[blue]Running integration tests...[/blue]")
        results = []
        
        # Test database connectivity
        results.append(await self._test_database_connection())
        
        # Test Redis connectivity
        results.append(await self._test_redis_connection())
        
        # Test API endpoints
        results.extend(await self._test_api_endpoints())
        
        # Test agent communication
        results.append(await self._test_agent_communication())
        
        # Test data pipeline
        results.append(await self._test_data_pipeline())
        
        return results

    async def _test_database_connection(self) -> TestResult:
        """Test database connectivity"""
        start_time = time.time()
        try:
            # Import database connection
            sys.path.append(str(self.config.project_root))
            from core.database import DatabaseManager
            
            db_manager = DatabaseManager()
            await db_manager.connect()
            
            # Test basic operations
            await db_manager.execute_query("SELECT 1")
            await db_manager.disconnect()
            
            return TestResult(
                test_name="database_connection",
                category="integration",
                status="passed",
                duration=time.time() - start_time,
                message="Database connection successful"
            )
            
        except Exception as e:
            return TestResult(
                test_name="database_connection",
                category="integration",
                status="failed",
                duration=time.time() - start_time,
                message=f"Database connection failed: {e}"
            )

    async def _test_redis_connection(self) -> TestResult:
        """Test Redis connectivity"""
        start_time = time.time()
        try:
            import redis
            
            # Connect to Redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            
            # Test basic operations
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            r.delete('test_key')
            
            return TestResult(
                test_name="redis_connection",
                category="integration",
                status="passed",
                duration=time.time() - start_time,
                message="Redis connection successful"
            )
            
        except Exception as e:
            return TestResult(
                test_name="redis_connection",
                category="integration",
                status="failed",
                duration=time.time() - start_time,
                message=f"Redis connection failed: {e}"
            )

    async def _test_api_endpoints(self) -> List[TestResult]:
        """Test API endpoints"""
        results = []
        
        # Common API endpoints to test
        endpoints = [
            ("GET", "/health", 200),
            ("GET", "/api/v1/status", 200),
            ("GET", "/api/v1/portfolio", 200),
            ("GET", "/api/v1/positions", 200),
            ("GET", "/api/v1/orders", 200)
        ]
        
        base_url = "http://localhost:8000"
        
        for method, endpoint, expected_status in endpoints:
            start_time = time.time()
            try:
                if method == "GET":
                    response = requests.get(f"{base_url}{endpoint}", timeout=10)
                elif method == "POST":
                    response = requests.post(f"{base_url}{endpoint}", timeout=10)
                else:
                    continue
                
                status = "passed" if response.status_code == expected_status else "failed"
                message = f"Status: {response.status_code}, Expected: {expected_status}"
                
                results.append(TestResult(
                    test_name=f"api_{endpoint.replace('/', '_')}",
                    category="integration",
                    status=status,
                    duration=time.time() - start_time,
                    message=message,
                    metrics={"response_time": time.time() - start_time}
                ))
                
            except Exception as e:
                results.append(TestResult(
                    test_name=f"api_{endpoint.replace('/', '_')}",
                    category="integration",
                    status="error",
                    duration=time.time() - start_time,
                    message=f"API test failed: {e}"
                ))
        
        return results

    async def _test_agent_communication(self) -> TestResult:
        """Test agent communication"""
        start_time = time.time()
        try:
            # Test agent message passing
            sys.path.append(str(self.config.project_root))
            from core.agents.base_agent import BaseAgent
            from core.communication.message_bus import MessageBus
            
            # Create test agents
            message_bus = MessageBus()
            agent1 = BaseAgent("test_agent_1", message_bus)
            agent2 = BaseAgent("test_agent_2", message_bus)
            
            # Test message sending
            test_message = {"type": "test", "data": "hello"}
            await agent1.send_message("test_agent_2", test_message)
            
            # Wait for message processing
            await asyncio.sleep(0.1)
            
            return TestResult(
                test_name="agent_communication",
                category="integration",
                status="passed",
                duration=time.time() - start_time,
                message="Agent communication successful"
            )
            
        except Exception as e:
            return TestResult(
                test_name="agent_communication",
                category="integration",
                status="failed",
                duration=time.time() - start_time,
                message=f"Agent communication failed: {e}"
            )

    async def _test_data_pipeline(self) -> TestResult:
        """Test data pipeline"""
        start_time = time.time()
        try:
            # Test data ingestion and processing
            sys.path.append(str(self.config.project_root))
            from core.data.data_pipeline import DataPipeline
            
            pipeline = DataPipeline()
            
            # Test with sample data
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
                'symbol': ['AAPL'] * 100,
                'price': np.random.uniform(150, 200, 100),
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Process data through pipeline
            processed_data = await pipeline.process(sample_data)
            
            # Validate processed data
            assert len(processed_data) > 0
            assert 'timestamp' in processed_data.columns
            
            return TestResult(
                test_name="data_pipeline",
                category="integration",
                status="passed",
                duration=time.time() - start_time,
                message="Data pipeline test successful",
                metrics={"processed_records": len(processed_data)}
            )
            
        except Exception as e:
            return TestResult(
                test_name="data_pipeline",
                category="integration",
                status="failed",
                duration=time.time() - start_time,
                message=f"Data pipeline test failed: {e}"
            )

    async def run_ai_model_tests(self) -> List[TestResult]:
        """Run AI model tests"""
        console.print("[blue]Running AI model tests...[/blue]")
        results = []
        
        # Test each AI integration
        ai_integrations = [
            ("qlib", self._test_qlib_integration),
            ("openbb", self._test_openbb_integration),
            ("rd_agent", self._test_rd_agent_integration),
            ("pearl_rl", self._test_pearl_rl_integration)
        ]
        
        for name, test_func in ai_integrations:
            try:
                result = await test_func()
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    test_name=f"{name}_integration",
                    category="ai_models",
                    status="error",
                    duration=0.0,
                    message=f"Test execution failed: {e}"
                ))
        
        return results

    async def _test_qlib_integration(self) -> TestResult:
        """Test Microsoft Qlib integration"""
        start_time = time.time()
        try:
            # Test Qlib import and basic functionality
            import qlib
            from qlib.config import REG_CN
            
            # Initialize Qlib
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            
            # Test data loading
            from qlib.data import D
            data = D.features(["SH600000"], ["$close", "$volume"], start_time="2023-01-01", end_time="2023-12-31")
            
            # Validate data
            assert not data.empty
            
            return TestResult(
                test_name="qlib_integration",
                category="ai_models",
                status="passed",
                duration=time.time() - start_time,
                message="Qlib integration successful",
                metrics={"data_points": len(data)}
            )
            
        except ImportError:
            return TestResult(
                test_name="qlib_integration",
                category="ai_models",
                status="skipped",
                duration=time.time() - start_time,
                message="Qlib not installed"
            )
        except Exception as e:
            return TestResult(
                test_name="qlib_integration",
                category="ai_models",
                status="failed",
                duration=time.time() - start_time,
                message=f"Qlib integration failed: {e}"
            )

    async def _test_openbb_integration(self) -> TestResult:
        """Test OpenBB Platform integration"""
        start_time = time.time()
        try:
            # Test OpenBB import and basic functionality
            from openbb import obb
            
            # Test data fetching
            data = obb.equity.price.historical("AAPL", start_date="2023-01-01", end_date="2023-12-31")
            
            # Validate data
            assert len(data) > 0
            
            return TestResult(
                test_name="openbb_integration",
                category="ai_models",
                status="passed",
                duration=time.time() - start_time,
                message="OpenBB integration successful",
                metrics={"data_points": len(data)}
            )
            
        except ImportError:
            return TestResult(
                test_name="openbb_integration",
                category="ai_models",
                status="skipped",
                duration=time.time() - start_time,
                message="OpenBB not installed"
            )
        except Exception as e:
            return TestResult(
                test_name="openbb_integration",
                category="ai_models",
                status="failed",
                duration=time.time() - start_time,
                message=f"OpenBB integration failed: {e}"
            )

    async def _test_rd_agent_integration(self) -> TestResult:
        """Test RD-Agent integration"""
        start_time = time.time()
        try:
            # Test RD-Agent import and basic functionality
            from rdagent.core.agent import Agent
            from rdagent.core.environment import Environment
            
            # Create test environment and agent
            env = Environment()
            agent = Agent(env)
            
            # Test basic agent functionality
            state = env.reset()
            action = agent.act(state)
            
            return TestResult(
                test_name="rd_agent_integration",
                category="ai_models",
                status="passed",
                duration=time.time() - start_time,
                message="RD-Agent integration successful"
            )
            
        except ImportError:
            return TestResult(
                test_name="rd_agent_integration",
                category="ai_models",
                status="skipped",
                duration=time.time() - start_time,
                message="RD-Agent not installed"
            )
        except Exception as e:
            return TestResult(
                test_name="rd_agent_integration",
                category="ai_models",
                status="failed",
                duration=time.time() - start_time,
                message=f"RD-Agent integration failed: {e}"
            )

    async def _test_pearl_rl_integration(self) -> TestResult:
        """Test Meta Pearl RL integration"""
        start_time = time.time()
        try:
            # Test Pearl import and basic functionality
            from pearl.api.agent import PearlAgent
            from pearl.api.environment import Environment
            
            # Create test environment and agent
            env = Environment()
            agent = PearlAgent()
            
            # Test basic RL functionality
            state = env.reset()
            action = agent.act(state)
            
            return TestResult(
                test_name="pearl_rl_integration",
                category="ai_models",
                status="passed",
                duration=time.time() - start_time,
                message="Pearl RL integration successful"
            )
            
        except ImportError:
            return TestResult(
                test_name="pearl_rl_integration",
                category="ai_models",
                status="skipped",
                duration=time.time() - start_time,
                message="Pearl RL not installed"
            )
        except Exception as e:
            return TestResult(
                test_name="pearl_rl_integration",
                category="ai_models",
                status="failed",
                duration=time.time() - start_time,
                message=f"Pearl RL integration failed: {e}"
            )

    async def run_performance_tests(self) -> List[TestResult]:
        """Run performance tests"""
        console.print("[blue]Running performance tests...[/blue]")
        results = []
        
        # Memory usage test
        results.append(await self._test_memory_usage())
        
        # CPU utilization test
        results.append(await self._test_cpu_utilization())
        
        # Response time test
        results.append(await self._test_response_times())
        
        # Throughput test
        results.append(await self._test_throughput())
        
        return results

    async def _test_memory_usage(self) -> TestResult:
        """Test memory usage"""
        start_time = time.time()
        try:
            # Monitor memory usage during typical operations
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate typical workload
            data = pd.DataFrame(np.random.randn(10000, 100))
            processed_data = data.rolling(window=10).mean()
            del data, processed_data
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Check if memory usage is within acceptable limits
            status = "passed" if memory_increase < 100 else "failed"  # 100MB limit
            
            return TestResult(
                test_name="memory_usage",
                category="performance",
                status=status,
                duration=time.time() - start_time,
                message=f"Memory increase: {memory_increase:.1f}MB",
                metrics={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="memory_usage",
                category="performance",
                status="error",
                duration=time.time() - start_time,
                message=f"Memory test failed: {e}"
            )

    async def _test_cpu_utilization(self) -> TestResult:
        """Test CPU utilization"""
        start_time = time.time()
        try:
            # Monitor CPU usage during computation
            cpu_percent_before = psutil.cpu_percent(interval=1)
            
            # Simulate CPU-intensive task
            for _ in range(1000000):
                _ = sum(range(100))
            
            cpu_percent_after = psutil.cpu_percent(interval=1)
            
            return TestResult(
                test_name="cpu_utilization",
                category="performance",
                status="passed",
                duration=time.time() - start_time,
                message=f"CPU usage: {cpu_percent_after:.1f}%",
                metrics={
                    "cpu_before": cpu_percent_before,
                    "cpu_after": cpu_percent_after
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="cpu_utilization",
                category="performance",
                status="error",
                duration=time.time() - start_time,
                message=f"CPU test failed: {e}"
            )

    async def _test_response_times(self) -> TestResult:
        """Test API response times"""
        start_time = time.time()
        try:
            base_url = "http://localhost:8000"
            response_times = []
            
            # Test multiple endpoints
            endpoints = ["/health", "/api/v1/status", "/api/v1/portfolio"]
            
            for endpoint in endpoints:
                endpoint_start = time.time()
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    response_time = time.time() - endpoint_start
                    response_times.append(response_time)
                except:
                    response_times.append(5.0)  # Timeout value
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Check if response times are acceptable
            status = "passed" if avg_response_time < 1.0 else "failed"  # 1 second limit
            
            return TestResult(
                test_name="response_times",
                category="performance",
                status=status,
                duration=time.time() - start_time,
                message=f"Avg response time: {avg_response_time:.3f}s",
                metrics={
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "response_times": response_times
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="response_times",
                category="performance",
                status="error",
                duration=time.time() - start_time,
                message=f"Response time test failed: {e}"
            )

    async def _test_throughput(self) -> TestResult:
        """Test system throughput"""
        start_time = time.time()
        try:
            # Test concurrent requests
            base_url = "http://localhost:8000/health"
            num_requests = 100
            concurrent_requests = 10
            
            async def make_request():
                try:
                    response = requests.get(base_url, timeout=5)
                    return response.status_code == 200
                except:
                    return False
            
            # Execute concurrent requests
            successful_requests = 0
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = [executor.submit(lambda: requests.get(base_url, timeout=5)) for _ in range(num_requests)]
                
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        if response.status_code == 200:
                            successful_requests += 1
                    except:
                        pass
            
            success_rate = successful_requests / num_requests
            throughput = successful_requests / (time.time() - start_time)
            
            # Check if throughput is acceptable
            status = "passed" if success_rate > 0.95 else "failed"  # 95% success rate
            
            return TestResult(
                test_name="throughput",
                category="performance",
                status=status,
                duration=time.time() - start_time,
                message=f"Throughput: {throughput:.1f} req/s, Success rate: {success_rate:.1%}",
                metrics={
                    "throughput_rps": throughput,
                    "success_rate": success_rate,
                    "successful_requests": successful_requests,
                    "total_requests": num_requests
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="throughput",
                category="performance",
                status="error",
                duration=time.time() - start_time,
                message=f"Throughput test failed: {e}"
            )

    async def run_security_tests(self) -> List[TestResult]:
        """Run security tests"""
        console.print("[blue]Running security tests...[/blue]")
        results = []
        
        # Static code analysis with Bandit
        results.append(await self._test_static_security_analysis())
        
        # Dependency vulnerability check
        results.append(await self._test_dependency_vulnerabilities())
        
        # API security tests
        results.extend(await self._test_api_security())
        
        return results

    async def _test_static_security_analysis(self) -> TestResult:
        """Run static security analysis with Bandit"""
        start_time = time.time()
        try:
            # Run Bandit security analysis
            cmd = [
                "bandit", "-r", "core/", "tools/", "web/",
                "-f", "json", "-o", "security_report.json"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse results
            report_file = self.config.project_root / "security_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                
                high_issues = len([issue for issue in report.get("results", []) if issue.get("issue_severity") == "HIGH"])
                medium_issues = len([issue for issue in report.get("results", []) if issue.get("issue_severity") == "MEDIUM"])
                
                status = "passed" if high_issues == 0 else "failed"
                message = f"High: {high_issues}, Medium: {medium_issues}"
                
                return TestResult(
                    test_name="static_security_analysis",
                    category="security",
                    status=status,
                    duration=time.time() - start_time,
                    message=message,
                    details={"report": report},
                    metrics={"high_issues": high_issues, "medium_issues": medium_issues}
                )
            else:
                return TestResult(
                    test_name="static_security_analysis",
                    category="security",
                    status="error",
                    duration=time.time() - start_time,
                    message="Security report not generated"
                )
                
        except Exception as e:
            return TestResult(
                test_name="static_security_analysis",
                category="security",
                status="error",
                duration=time.time() - start_time,
                message=f"Security analysis failed: {e}"
            )

    async def _test_dependency_vulnerabilities(self) -> TestResult:
        """Test for dependency vulnerabilities"""
        start_time = time.time()
        try:
            # Run Safety check
            cmd = ["safety", "check", "--json"]
            
            result = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
                
                status = "passed" if len(vulnerabilities) == 0 else "failed"
                message = f"Found {len(vulnerabilities)} vulnerabilities"
                
                return TestResult(
                    test_name="dependency_vulnerabilities",
                    category="security",
                    status=status,
                    duration=time.time() - start_time,
                    message=message,
                    details={"vulnerabilities": vulnerabilities},
                    metrics={"vulnerability_count": len(vulnerabilities)}
                )
            else:
                return TestResult(
                    test_name="dependency_vulnerabilities",
                    category="security",
                    status="error",
                    duration=time.time() - start_time,
                    message=f"Safety check failed: {result.stderr}"
                )
                
        except Exception as e:
            return TestResult(
                test_name="dependency_vulnerabilities",
                category="security",
                status="error",
                duration=time.time() - start_time,
                message=f"Dependency check failed: {e}"
            )

    async def _test_api_security(self) -> List[TestResult]:
        """Test API security"""
        results = []
        base_url = "http://localhost:8000"
        
        # Test for common security headers
        start_time = time.time()
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            headers = response.headers
            
            required_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection",
                "Strict-Transport-Security"
            ]
            
            missing_headers = [header for header in required_headers if header not in headers]
            
            status = "passed" if len(missing_headers) == 0 else "failed"
            message = f"Missing headers: {missing_headers}" if missing_headers else "All security headers present"
            
            results.append(TestResult(
                test_name="security_headers",
                category="security",
                status=status,
                duration=time.time() - start_time,
                message=message,
                details={"headers": dict(headers), "missing_headers": missing_headers}
            ))
            
        except Exception as e:
            results.append(TestResult(
                test_name="security_headers",
                category="security",
                status="error",
                duration=time.time() - start_time,
                message=f"Security headers test failed: {e}"
            ))
        
        return results

    async def run_end_to_end_tests(self) -> List[TestResult]:
        """Run end-to-end tests"""
        console.print("[blue]Running end-to-end tests...[/blue]")
        results = []
        
        # Test complete trading workflow
        results.append(await self._test_trading_workflow())
        
        # Test portfolio management workflow
        results.append(await self._test_portfolio_workflow())
        
        # Test monitoring and alerting
        results.append(await self._test_monitoring_workflow())
        
        return results

    async def _test_trading_workflow(self) -> TestResult:
        """Test complete trading workflow"""
        start_time = time.time()
        try:
            # Simulate complete trading workflow
            base_url = "http://localhost:8000/api/v1"
            
            # 1. Get market data
            response = requests.get(f"{base_url}/market/data/AAPL", timeout=10)
            assert response.status_code == 200
            
            # 2. Generate trading signal
            response = requests.post(f"{base_url}/signals/generate", 
                                   json={"symbol": "AAPL"}, timeout=10)
            assert response.status_code == 200
            
            # 3. Execute trade (simulation)
            response = requests.post(f"{base_url}/orders", 
                                   json={
                                       "symbol": "AAPL",
                                       "side": "buy",
                                       "quantity": 10,
                                       "type": "market"
                                   }, timeout=10)
            assert response.status_code in [200, 201]
            
            return TestResult(
                test_name="trading_workflow",
                category="end_to_end",
                status="passed",
                duration=time.time() - start_time,
                message="Trading workflow completed successfully"
            )
            
        except Exception as e:
            return TestResult(
                test_name="trading_workflow",
                category="end_to_end",
                status="failed",
                duration=time.time() - start_time,
                message=f"Trading workflow failed: {e}"
            )

    async def _test_portfolio_workflow(self) -> TestResult:
        """Test portfolio management workflow"""
        start_time = time.time()
        try:
            base_url = "http://localhost:8000/api/v1"
            
            # 1. Get portfolio status
            response = requests.get(f"{base_url}/portfolio", timeout=10)
            assert response.status_code == 200
            
            # 2. Get positions
            response = requests.get(f"{base_url}/positions", timeout=10)
            assert response.status_code == 200
            
            # 3. Get performance metrics
            response = requests.get(f"{base_url}/portfolio/performance", timeout=10)
            assert response.status_code == 200
            
            return TestResult(
                test_name="portfolio_workflow",
                category="end_to_end",
                status="passed",
                duration=time.time() - start_time,
                message="Portfolio workflow completed successfully"
            )
            
        except Exception as e:
            return TestResult(
                test_name="portfolio_workflow",
                category="end_to_end",
                status="failed",
                duration=time.time() - start_time,
                message=f"Portfolio workflow failed: {e}"
            )

    async def _test_monitoring_workflow(self) -> TestResult:
        """Test monitoring and alerting workflow"""
        start_time = time.time()
        try:
            base_url = "http://localhost:8000/api/v1"
            
            # 1. Get system metrics
            response = requests.get(f"{base_url}/monitoring/metrics", timeout=10)
            assert response.status_code == 200
            
            # 2. Get alerts
            response = requests.get(f"{base_url}/monitoring/alerts", timeout=10)
            assert response.status_code == 200
            
            return TestResult(
                test_name="monitoring_workflow",
                category="end_to_end",
                status="passed",
                duration=time.time() - start_time,
                message="Monitoring workflow completed successfully"
            )
            
        except Exception as e:
            return TestResult(
                test_name="monitoring_workflow",
                category="end_to_end",
                status="failed",
                duration=time.time() - start_time,
                message=f"Monitoring workflow failed: {e}"
            )

    async def run_comprehensive_tests(self) -> Dict[str, List[TestResult]]:
        """Run all test suites"""
        console.print(Panel(
            "[bold blue]Comprehensive Testing Framework[/bold blue]\n"
            "Running complete test suite for AI Trading System...",
            border_style="blue"
        ))
        
        all_results = {}
        
        # Run test suites in dependency order
        suite_order = ["unit", "integration", "ai_models", "performance", "security", "end_to_end"]
        
        for suite_name in suite_order:
            if suite_name not in self.test_suites:
                continue
                
            suite = self.test_suites[suite_name]
            
            # Check dependencies
            if suite.dependencies:
                failed_deps = []
                for dep in suite.dependencies:
                    if dep in all_results:
                        dep_results = all_results[dep]
                        if any(result.status == "failed" for result in dep_results if result.critical):
                            failed_deps.append(dep)
                
                if failed_deps:
                    console.print(f"[yellow]Skipping {suite.name} due to failed dependencies: {failed_deps}[/yellow]")
                    continue
            
            console.print(f"\n[cyan]Running {suite.name}...[/cyan]")
            
            # Run test suite
            if suite_name == "unit":
                results = await self.run_unit_tests()
            elif suite_name == "integration":
                results = await self.run_integration_tests()
            elif suite_name == "ai_models":
                results = await self.run_ai_model_tests()
            elif suite_name == "performance":
                results = await self.run_performance_tests()
            elif suite_name == "security":
                results = await self.run_security_tests()
            elif suite_name == "end_to_end":
                results = await self.run_end_to_end_tests()
            else:
                results = []
            
            all_results[suite_name] = results
            self.results.extend(results)
            
            # Display suite results
            self._display_suite_results(suite.name, results)
        
        return all_results

    def _display_suite_results(self, suite_name: str, results: List[TestResult]):
        """Display results for a test suite"""
        if not results:
            return
        
        table = Table(title=f"{suite_name} Results")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Duration", style="yellow")
        table.add_column("Message", style="white")
        
        for result in results:
            status_color = {
                "passed": "green",
                "failed": "red",
                "error": "red",
                "skipped": "blue"
            }.get(result.status, "white")
            
            table.add_row(
                result.test_name,
                f"[{status_color}]{result.status.upper()}[/{status_color}]",
                f"{result.duration:.2f}s",
                result.message
            )
        
        console.print(table)

    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.results:
            return ""
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "passed"])
        failed_tests = len([r for r in self.results if r.status == "failed"])
        error_tests = len([r for r in self.results if r.status == "error"])
        skipped_tests = len([r for r in self.results if r.status == "skipped"])
        
        total_duration = sum(r.duration for r in self.results)
        
        # Group results by category
        results_by_category = {}
        for result in self.results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration
            },
            "results_by_category": {
                category: [
                    {
                        "test_name": result.test_name,
                        "status": result.status,
                        "duration": result.duration,
                        "message": result.message,
                        "details": result.details,
                        "metrics": result.metrics
                    }
                    for result in results
                ]
                for category, results in results_by_category.items()
            },
            "configuration": {
                "project_root": str(self.config.project_root),
                "coverage_threshold": self.config.coverage_threshold,
                "security_level": self.config.security_level
            }
        }
        
        # Save report
        report_dir = self.config.project_root / "logs" / "testing"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(report_file)

    def display_final_summary(self):
        """Display final test summary"""
        if not self.results:
            console.print("[yellow]No test results to display[/yellow]")
            return
        
        # Summary statistics
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "passed"])
        failed_tests = len([r for r in self.results if r.status == "failed"])
        error_tests = len([r for r in self.results if r.status == "error"])
        skipped_tests = len([r for r in self.results if r.status == "skipped"])
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        total_duration = sum(r.duration for r in self.results)
        
        # Summary table
        summary_table = Table(title="Test Execution Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="bold")
        
        summary_table.add_row("Total Tests", str(total_tests))
        summary_table.add_row("Passed", f"[green]{passed_tests}[/green]")
        summary_table.add_row("Failed", f"[red]{failed_tests}[/red]")
        summary_table.add_row("Errors", f"[red]{error_tests}[/red]")
        summary_table.add_row("Skipped", f"[blue]{skipped_tests}[/blue]")
        summary_table.add_row("Success Rate", f"{success_rate:.1%}")
        summary_table.add_row("Total Duration", f"{total_duration:.1f}s")
        
        console.print(Panel(summary_table, title="Final Results", border_style="green"))
        
        # Recommendations
        if failed_tests > 0 or error_tests > 0:
            recommendations = """
[red]Issues found during testing![/red]

Recommendations:
1. Review failed tests in the detailed report
2. Fix critical issues before deployment
3. Re-run tests after fixes
4. Consider increasing test coverage
            """
            console.print(Panel(recommendations.strip(), title="Action Required", border_style="red"))
        else:
            recommendations = """
[green]All tests passed successfully![/green]

System is ready for:
1. Production deployment
2. AI tool integration
3. Performance optimization
4. Monitoring activation
            """
            console.print(Panel(recommendations.strip(), title="Ready for Production", border_style="green"))

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Testing Framework")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--suite", choices=["unit", "integration", "ai_models", "performance", "security", "end_to_end", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--coverage-threshold", type=float, default=80.0, help="Coverage threshold")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TestConfiguration(
        project_root=Path(args.project_root).resolve(),
        coverage_threshold=args.coverage_threshold
    )
    
    # Create testing framework
    framework = ComprehensiveTestingFramework(config)
    
    # Run tests
    if args.suite == "all":
        results = await framework.run_comprehensive_tests()
    else:
        # Run specific suite
        if args.suite == "unit":
            results = {"unit": await framework.run_unit_tests()}
        elif args.suite == "integration":
            results = {"integration": await framework.run_integration_tests()}
        elif args.suite == "ai_models":
            results = {"ai_models": await framework.run_ai_model_tests()}
        elif args.suite == "performance":
            results = {"performance": await framework.run_performance_tests()}
        elif args.suite == "security":
            results = {"security": await framework.run_security_tests()}
        elif args.suite == "end_to_end":
            results = {"end_to_end": await framework.run_end_to_end_tests()}
        
        framework.results = []
        for suite_results in results.values():
            framework.results.extend(suite_results)
    
    # Display final summary
    framework.display_final_summary()
    
    # Generate report if requested
    if args.report:
        report_file = framework.generate_test_report()
        console.print(f"[green]Detailed report saved to: {report_file}[/green]")

if __name__ == "__main__":
    asyncio.run(main())