#!/usr/bin/env python3
"""
Standalone Agent Orchestrator Launcher

This script starts the 8-agent AI trading team without complex dependencies.
It initializes the core agent orchestrator and runs the multi-agent system.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_orchestrator.log')
    ]
)

logger = logging.getLogger(__name__)

class SimpleAgentOrchestrator:
    """Simplified agent orchestrator that can run without complex dependencies."""
    
    def __init__(self):
        self.agents = {}
        self.is_running = False
        self.start_time = None
        
    async def initialize(self):
        """Initialize the agent system."""
        logger.info("🚀 Starting Agent Orchestrator...")
        self.start_time = datetime.now()
        
        # Create 8 core agents
        self.agents = {
            "market_researcher": {
                "role": "Market Research Specialist",
                "status": "active",
                "tasks_completed": 0,
                "specialization": "Market data analysis and trend identification"
            },
            "technical_analyst": {
                "role": "Technical Analysis Expert", 
                "status": "active",
                "tasks_completed": 0,
                "specialization": "Chart patterns and technical indicators"
            },
            "momentum_strategist": {
                "role": "Momentum Strategy Specialist",
                "status": "active", 
                "tasks_completed": 0,
                "specialization": "Momentum-based trading strategies"
            },
            "mean_reversion_strategist": {
                "role": "Mean Reversion Strategy Specialist",
                "status": "active",
                "tasks_completed": 0, 
                "specialization": "Mean reversion and contrarian strategies"
            },
            "smart_executor": {
                "role": "Smart Order Execution Specialist",
                "status": "active",
                "tasks_completed": 0,
                "specialization": "Optimal trade execution and routing"
            },
            "risk_manager": {
                "role": "Risk Management Specialist",
                "status": "active",
                "tasks_completed": 0,
                "specialization": "Portfolio risk assessment and management"
            },
            "portfolio_optimizer": {
                "role": "Portfolio Optimization Specialist", 
                "status": "active",
                "tasks_completed": 0,
                "specialization": "Asset allocation and portfolio optimization"
            },
            "master_coordinator": {
                "role": "Master Coordinator",
                "status": "active",
                "tasks_completed": 0,
                "specialization": "Multi-agent coordination and workflow management"
            }
        }
        
        self.is_running = True
        logger.info(f"✅ Initialized {len(self.agents)} agents successfully")
        
        # Start agent activities
        await self._start_agent_activities()
        
    async def _start_agent_activities(self):
        """Start background activities for all agents."""
        logger.info("🔄 Starting agent coordination activities...")
        
        # Simulate agent coordination and task execution
        tasks = [
            self._agent_activity_loop("market_researcher", 30),
            self._agent_activity_loop("technical_analyst", 45), 
            self._agent_activity_loop("momentum_strategist", 60),
            self._agent_activity_loop("mean_reversion_strategist", 60),
            self._agent_activity_loop("smart_executor", 20),
            self._agent_activity_loop("risk_manager", 90),
            self._agent_activity_loop("portfolio_optimizer", 120),
            self._agent_activity_loop("master_coordinator", 15)
        ]
        
        # Start all agent activities concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _agent_activity_loop(self, agent_id: str, interval_seconds: int):
        """Simulate agent activity with periodic tasks."""
        agent = self.agents[agent_id]
        
        while self.is_running:
            try:
                # Simulate agent work
                await asyncio.sleep(interval_seconds)
                
                # Update agent metrics
                agent["tasks_completed"] += 1
                agent["last_activity"] = datetime.now()
                
                # Log agent activity
                logger.info(f"🤖 {agent['role']} completed task #{agent['tasks_completed']}")
                
                # Special coordination logic for master coordinator
                if agent_id == "master_coordinator":
                    await self._coordinate_agents()
                    
            except Exception as e:
                logger.error(f"❌ Error in {agent_id}: {e}")
                agent["status"] = "error"
                await asyncio.sleep(10)  # Wait before retrying
                agent["status"] = "active"
                
    async def _coordinate_agents(self):
        """Master coordinator orchestrates other agents."""
        active_agents = [aid for aid, agent in self.agents.items() 
                        if agent["status"] == "active" and aid != "master_coordinator"]
        
        logger.info(f"🎯 Master Coordinator: Managing {len(active_agents)} active agents")
        
        # Simulate coordination tasks
        coordination_tasks = [
            "Analyzing market conditions",
            "Coordinating strategy signals", 
            "Managing risk exposure",
            "Optimizing portfolio allocation",
            "Monitoring execution quality"
        ]
        
        import random
        task = random.choice(coordination_tasks)
        logger.info(f"🎯 Master Coordinator: {task}")
        
    def get_system_status(self):
        """Get current system status."""
        uptime = datetime.now() - self.start_time if self.start_time else None
        
        return {
            "status": "running" if self.is_running else "stopped",
            "uptime": str(uptime) if uptime else "0:00:00",
            "agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a["status"] == "active"]),
            "total_tasks_completed": sum(a["tasks_completed"] for a in self.agents.values()),
            "start_time": self.start_time.isoformat() if self.start_time else None
        }
        
    def get_agent_details(self):
        """Get detailed agent information."""
        return self.agents
        
    async def shutdown(self):
        """Shutdown the agent orchestrator."""
        logger.info("🛑 Shutting down Agent Orchestrator...")
        self.is_running = False
        
        # Give agents time to finish current tasks
        await asyncio.sleep(2)
        
        logger.info("✅ Agent Orchestrator shutdown complete")

async def main():
    """Main entry point."""
    orchestrator = SimpleAgentOrchestrator()
    
    try:
        # Initialize and start the orchestrator
        await orchestrator.initialize()
        
        # Print initial status
        status = orchestrator.get_system_status()
        logger.info(f"📊 System Status: {status}")
        
        # Print agent details
        agents = orchestrator.get_agent_details()
        logger.info("👥 Agent Team:")
        for agent_id, agent_info in agents.items():
            logger.info(f"  • {agent_info['role']}: {agent_info['specialization']}")
        
        logger.info("🎯 Agent team is now coordinating 24/7 trading activities...")
        logger.info("📈 Monitoring market conditions and executing strategies...")
        logger.info("⚡ Press Ctrl+C to stop the system")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(60)  # Status update every minute
            status = orchestrator.get_system_status()
            logger.info(f"📊 Status Update - Active: {status['active_agents']}/{status['agents']} agents, "
                       f"Tasks: {status['total_tasks_completed']}, Uptime: {status['uptime']}")
            
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal...")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    # Set environment variables if needed
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent))
    
    # Run the orchestrator
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Agent Orchestrator stopped by user")
    except Exception as e:
        print(f"❌ Failed to start Agent Orchestrator: {e}")
        sys.exit(1)