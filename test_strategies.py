#!/usr/bin/env python3
"""
Quick test script to validate the strategy assessment framework
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from strategy_assessment_framework import StrategyAssessmentFramework
    from core.config_manager import ConfigManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Available modules:")
    import os
    for item in os.listdir('.'):
        if item.endswith('.py'):
            print(f"  - {item}")
    sys.exit(1)

async def test_strategies():
    """Test the strategy assessment framework."""
    try:
        print("Initializing configuration manager...")
        config_path = Path("config")
        config = ConfigManager(config_path)
        
        print("Creating strategy assessment framework...")
        framework = StrategyAssessmentFramework()
        
        print("Getting enabled strategies from config...")
        enabled_strategies = config.get_enabled_strategies()
        print(f"Enabled strategies: {enabled_strategies}")
        
        print("\nSimulating strategy assessments...")
        await framework.simulate_strategy_assessments(num_strategies=5)
        
        print("\nGenerating comprehensive report...")
        report = await framework.generate_comprehensive_report()
        
        print(f"\nAssessment Summary:")
        print(f"  - Total agents assessed: {len(report.get('agent_assessments', []))}")
        print(f"  - Total strategies evaluated: {len(framework.strategy_performances)}")
        print(f"  - Top performing strategies: {len(report.get('top_performing_strategies', []))}")
        
        for agent in report.get('agent_assessments', []):
            print(f"  - Agent {agent['agent_id']}: Score {agent['overall_score']:.1f}")
            
        print("\nStrategy assessment framework test completed successfully!")
        
    except Exception as e:
        print(f"Error testing strategies: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_strategies())
    sys.exit(0 if success else 1)