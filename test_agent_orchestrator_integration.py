#!/usr/bin/env python3
"""
Test Agent Orchestrator Integration with Dynamic Discovery Tools
Verifies end-to-end discovery workflows through the agent orchestrator
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentOrchestratorIntegrationTester:
    """Test the integration between agent orchestrator and discovery tools"""
    
    def __init__(self):
        self.test_results = {}
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
    async def test_orchestrator_initialization(self) -> Dict[str, Any]:
        """Test that the agent orchestrator initializes with discovery tools"""
        logger.info("Testing Agent Orchestrator initialization...")
        
        try:
            # Import and initialize the orchestrator
            from core.agent_orchestrator import AgentOrchestrator
            
            # Create orchestrator instance
            orchestrator = AgentOrchestrator()
            
            # Check if discovery tools are initialized
            discovery_tools = [
                'volume_spike_tool',
                'news_discovery_tool', 
                'technical_breakout_tool',
                'earnings_calendar_tool',
                'sector_rotation_tool'
            ]
            
            initialized_tools = []
            missing_tools = []
            
            for tool_name in discovery_tools:
                if hasattr(orchestrator, tool_name):
                    tool = getattr(orchestrator, tool_name)
                    if tool is not None:
                        initialized_tools.append(tool_name)
                    else:
                        missing_tools.append(f"{tool_name} (None)")
                else:
                    missing_tools.append(f"{tool_name} (missing)")
            
            return {
                'status': 'success' if len(missing_tools) == 0 else 'partial',
                'initialized_tools': initialized_tools,
                'missing_tools': missing_tools,
                'total_tools': len(discovery_tools),
                'success_rate': f"{(len(initialized_tools)/len(discovery_tools))*100:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Orchestrator initialization test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_discovery_request_methods(self) -> Dict[str, Any]:
        """Test discovery request methods in the orchestrator"""
        logger.info("Testing discovery request methods...")
        
        try:
            from core.agent_orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator()
            
            # Test discovery request methods
            request_methods = [
                'request_volume_spike_discovery',
                'request_news_driven_discovery',
                'request_technical_breakout_discovery',
                'request_earnings_calendar_discovery',
                'request_sector_rotation_discovery',
                'request_comprehensive_discovery'
            ]
            
            available_methods = []
            missing_methods = []
            
            for method_name in request_methods:
                if hasattr(orchestrator, method_name):
                    method = getattr(orchestrator, method_name)
                    if callable(method):
                        available_methods.append(method_name)
                    else:
                        missing_methods.append(f"{method_name} (not callable)")
                else:
                    missing_methods.append(f"{method_name} (missing)")
            
            return {
                'status': 'success' if len(missing_methods) == 0 else 'partial',
                'available_methods': available_methods,
                'missing_methods': missing_methods,
                'total_methods': len(request_methods),
                'success_rate': f"{(len(available_methods)/len(request_methods))*100:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Discovery request methods test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_volume_spike_discovery_request(self) -> Dict[str, Any]:
        """Test volume spike discovery request"""
        logger.info("Testing volume spike discovery request...")
        
        try:
            from core.agent_orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator()
            
            # Test volume spike discovery request
            task_id = await orchestrator.request_volume_spike_discovery(
                symbols=self.test_symbols[:3],
                volume_threshold=2.0,
                time_window="1d"
            )
            
            # Wait a moment for task to be processed
            await asyncio.sleep(1)
            
            # Check task status
            task_status = orchestrator.get_task_status(task_id)
            
            return {
                'status': 'success',
                'task_id': task_id,
                'task_status': task_status,
                'symbols_requested': self.test_symbols[:3],
                'parameters': {
                    'volume_threshold': 2.0,
                    'time_window': '1d'
                }
            }
            
        except Exception as e:
            logger.error(f"Volume spike discovery request test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_comprehensive_discovery_request(self) -> Dict[str, Any]:
        """Test comprehensive discovery request"""
        logger.info("Testing comprehensive discovery request...")
        
        try:
            from core.agent_orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator()
            
            # Test comprehensive discovery request
            task_id = await orchestrator.request_comprehensive_discovery(
                symbols=self.test_symbols[:2],
                discovery_types=["volume_spike", "technical_breakout", "earnings_calendar"],
                priority=1
            )
            
            # Wait a moment for task to be processed
            await asyncio.sleep(1)
            
            # Check task status
            task_status = orchestrator.get_task_status(task_id)
            
            return {
                'status': 'success',
                'task_id': task_id,
                'task_status': task_status,
                'symbols_requested': self.test_symbols[:2],
                'discovery_types': ["volume_spike", "technical_breakout", "earnings_calendar"]
            }
            
        except Exception as e:
            logger.error(f"Comprehensive discovery request test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_agent_tool_integration(self) -> Dict[str, Any]:
        """Test that research agents have access to discovery tools"""
        logger.info("Testing agent tool integration...")
        
        try:
            from core.agent_orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator()
            
            # Get research agents
            research_agents = orchestrator._create_research_agents()
            
            # Find Market Research Analyst
            market_analyst = None
            for agent in research_agents:
                if "Market Research Analyst" in agent.role:
                    market_analyst = agent
                    break
            
            if not market_analyst:
                return {
                    'status': 'error',
                    'error': 'Market Research Analyst not found'
                }
            
            # Check tools
            expected_tools = [
                'volume_spike_tool',
                'news_discovery_tool',
                'technical_breakout_tool',
                'earnings_calendar_tool',
                'sector_rotation_tool'
            ]
            
            agent_tools = [tool.__class__.__name__ for tool in market_analyst.tools]
            
            found_tools = []
            missing_tools = []
            
            for expected_tool in expected_tools:
                # Check if any tool class name contains the expected tool name
                tool_found = any(expected_tool.replace('_tool', '').replace('_', '').lower() 
                               in tool_name.lower() for tool_name in agent_tools)
                if tool_found:
                    found_tools.append(expected_tool)
                else:
                    missing_tools.append(expected_tool)
            
            return {
                'status': 'success' if len(missing_tools) == 0 else 'partial',
                'agent_role': market_analyst.role,
                'total_tools': len(market_analyst.tools),
                'agent_tools': agent_tools,
                'found_discovery_tools': found_tools,
                'missing_discovery_tools': missing_tools,
                'success_rate': f"{(len(found_tools)/len(expected_tools))*100:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Agent tool integration test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_system_metrics(self) -> Dict[str, Any]:
        """Test system metrics and status"""
        logger.info("Testing system metrics...")
        
        try:
            from core.agent_orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator()
            
            # Get system metrics
            metrics = orchestrator.get_system_metrics()
            
            # Get agent status
            agent_status = orchestrator.get_agent_status()
            
            return {
                'status': 'success',
                'system_metrics': metrics,
                'agent_status': agent_status,
                'metrics_available': metrics is not None,
                'agent_status_available': agent_status is not None
            }
            
        except Exception as e:
            logger.error(f"System metrics test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting Agent Orchestrator integration tests...")
        
        test_suite = {
            'orchestrator_initialization': self.test_orchestrator_initialization,
            'discovery_request_methods': self.test_discovery_request_methods,
            'volume_spike_discovery_request': self.test_volume_spike_discovery_request,
            'comprehensive_discovery_request': self.test_comprehensive_discovery_request,
            'agent_tool_integration': self.test_agent_tool_integration,
            'system_metrics': self.test_system_metrics
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
                elif result['status'] == 'partial':
                    logger.warning(f"⚠️ {test_name}: PARTIAL - Some issues detected")
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
        partial_tests = len([r for r in results.values() if r['status'] == 'partial'])
        failed_tests = len([r for r in results.values() if r['status'] in ['error', 'crashed']])
        
        summary = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'partial_tests': partial_tests,
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
        """Print a formatted integration test report"""
        print("\n" + "="*80)
        print("AGENT ORCHESTRATOR INTEGRATION - TEST REPORT")
        print("="*80)
        
        summary = results['test_summary']
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ✅")
        print(f"   Partial: {summary['partial_tests']} ⚠️")
        print(f"   Failed: {summary['failed_tests']} ❌")
        print(f"   Success Rate: {summary['success_rate']}")
        print(f"   Total Time: {summary['total_execution_time']}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for test_name, result in results['test_results'].items():
            if result['status'] == 'success':
                status_icon = "✅"
            elif result['status'] == 'partial':
                status_icon = "⚠️"
            else:
                status_icon = "❌"
                
            print(f"\n   {status_icon} {test_name.replace('_', ' ').title()}")
            print(f"      Status: {result['status']}")
            print(f"      Time: {result.get('execution_time', 0):.2f}s")
            
            if result['status'] in ['success', 'partial']:
                # Print specific details based on test type
                if 'initialized_tools' in result:
                    print(f"      Initialized Tools: {len(result['initialized_tools'])}")
                    if result['missing_tools']:
                        print(f"      Missing Tools: {result['missing_tools']}")
                
                if 'available_methods' in result:
                    print(f"      Available Methods: {len(result['available_methods'])}")
                    if result['missing_methods']:
                        print(f"      Missing Methods: {result['missing_methods']}")
                
                if 'task_id' in result:
                    print(f"      Task ID: {result['task_id']}")
                    print(f"      Task Status: {result.get('task_status', 'Unknown')}")
                
                if 'found_discovery_tools' in result:
                    print(f"      Discovery Tools Found: {len(result['found_discovery_tools'])}")
                    if result['missing_discovery_tools']:
                        print(f"      Missing Discovery Tools: {result['missing_discovery_tools']}")
                
                if 'success_rate' in result:
                    print(f"      Success Rate: {result['success_rate']}")
                    
            else:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        print(f"\n📈 INTEGRATION HEALTH:")
        if summary['passed_tests'] == summary['total_tests']:
            print("   🟢 Full integration successful - All systems properly connected!")
        elif summary['passed_tests'] + summary['partial_tests'] >= summary['total_tests'] * 0.8:
            print("   🟡 Most integrations working - Minor configuration issues detected")
        else:
            print("   🔴 Integration failures detected - Requires immediate attention")
        
        print("\n" + "="*80)

async def main():
    """Main test execution"""
    tester = AgentOrchestratorIntegrationTester()
    
    try:
        results = await tester.run_all_tests()
        tester.print_report(results)
        
        # Save results to file
        import json
        with open('agent_orchestrator_integration_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: agent_orchestrator_integration_test_results.json")
        
        # Return appropriate exit code
        if results['test_summary']['failed_tests'] == 0:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Integration test execution failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)