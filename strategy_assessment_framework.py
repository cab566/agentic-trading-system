#!/usr/bin/env python3
"""
Strategy Assessment Framework - Agent Decision-Making Evaluation

Comprehensive framework for evaluating agent effectiveness in strategy
development, information gathering, and decision-making processes.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics

class StrategyType(Enum):
    """Types of trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    TREND_FOLLOWING = "trend_following"
    PAIRS_TRADING = "pairs_trading"
    MARKET_MAKING = "market_making"
    NEWS_BASED = "news_based"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"

class AssessmentMetric(Enum):
    """Assessment metrics for strategy evaluation"""
    INFORMATION_QUALITY = "information_quality"
    DECISION_SPEED = "decision_speed"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_TIMING = "market_timing"
    ADAPTABILITY = "adaptability"
    CONSISTENCY = "consistency"
    INNOVATION = "innovation"
    EXECUTION_QUALITY = "execution_quality"

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_id: str
    strategy_type: StrategyType
    agent_id: str
    timestamp: datetime
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Information gathering metrics
    data_sources_used: int = 0
    information_freshness: float = 0.0  # Average age of information in minutes
    information_relevance: float = 0.0  # Relevance score 0-100
    
    # Decision-making metrics
    decision_time: float = 0.0  # Time to make decision in seconds
    confidence_level: float = 0.0  # Agent's confidence 0-100
    risk_score: float = 0.0  # Risk assessment 0-100
    
    # Execution metrics
    execution_delay: float = 0.0  # Delay between decision and execution
    slippage: float = 0.0  # Price slippage percentage
    
    # Market context
    market_volatility: float = 0.0
    market_trend: str = "neutral"
    market_conditions: str = "normal"

@dataclass
class AgentAssessment:
    """Comprehensive agent assessment"""
    agent_id: str
    agent_type: str
    assessment_period: Tuple[datetime, datetime]
    
    # Overall scores (0-100)
    overall_score: float = 0.0
    information_gathering_score: float = 0.0
    decision_making_score: float = 0.0
    risk_management_score: float = 0.0
    adaptability_score: float = 0.0
    
    # Detailed metrics
    strategies_evaluated: int = 0
    successful_strategies: int = 0
    avg_strategy_performance: float = 0.0
    best_strategy_return: float = 0.0
    worst_strategy_return: float = 0.0
    
    # Information metrics
    avg_data_sources: float = 0.0
    avg_information_freshness: float = 0.0
    avg_information_relevance: float = 0.0
    
    # Decision metrics
    avg_decision_time: float = 0.0
    avg_confidence: float = 0.0
    decision_accuracy: float = 0.0
    
    # Recommendations
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class StrategyAssessmentFramework:
    """Framework for assessing agent strategy effectiveness"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.strategy_performances: List[StrategyPerformance] = []
        self.agent_assessments: Dict[str, AgentAssessment] = {}
        self.assessment_history: deque = deque(maxlen=1000)
        self.benchmark_metrics = self._load_benchmark_metrics()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("strategy_assessment")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = Path("logs/strategy_assessment.log")
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_benchmark_metrics(self) -> Dict[str, float]:
        """Load benchmark metrics for comparison"""
        # Default benchmark metrics (can be loaded from config)
        return {
            'min_sharpe_ratio': 1.0,
            'max_drawdown_threshold': 15.0,
            'min_win_rate': 55.0,
            'min_profit_factor': 1.2,
            'max_decision_time': 30.0,  # seconds
            'min_confidence': 70.0,
            'min_information_relevance': 80.0,
            'max_information_age': 300.0,  # 5 minutes
            'min_data_sources': 3
        }
    
    async def assess_strategy_performance(self, strategy_data: Dict[str, Any]) -> StrategyPerformance:
        """Assess individual strategy performance"""
        try:
            # Extract strategy information
            strategy_id = strategy_data.get('strategy_id', 'unknown')
            agent_id = strategy_data.get('agent_id', 'unknown')
            
            # Create performance object
            performance = StrategyPerformance(
                strategy_id=strategy_id,
                strategy_type=StrategyType(strategy_data.get('strategy_type', 'technical')),
                agent_id=agent_id,
                timestamp=datetime.now()
            )
            
            # Calculate performance metrics
            performance.total_return = strategy_data.get('total_return', 0.0)
            performance.sharpe_ratio = strategy_data.get('sharpe_ratio', 0.0)
            performance.max_drawdown = strategy_data.get('max_drawdown', 0.0)
            performance.win_rate = strategy_data.get('win_rate', 0.0)
            performance.profit_factor = strategy_data.get('profit_factor', 0.0)
            
            # Information gathering metrics
            performance.data_sources_used = strategy_data.get('data_sources_used', 0)
            performance.information_freshness = strategy_data.get('information_freshness', 0.0)
            performance.information_relevance = strategy_data.get('information_relevance', 0.0)
            
            # Decision-making metrics
            performance.decision_time = strategy_data.get('decision_time', 0.0)
            performance.confidence_level = strategy_data.get('confidence_level', 0.0)
            performance.risk_score = strategy_data.get('risk_score', 0.0)
            
            # Execution metrics
            performance.execution_delay = strategy_data.get('execution_delay', 0.0)
            performance.slippage = strategy_data.get('slippage', 0.0)
            
            # Market context
            performance.market_volatility = strategy_data.get('market_volatility', 0.0)
            performance.market_trend = strategy_data.get('market_trend', 'neutral')
            performance.market_conditions = strategy_data.get('market_conditions', 'normal')
            
            # Store performance
            self.strategy_performances.append(performance)
            
            # Log assessment
            self.logger.info(f"Assessed strategy {strategy_id} by agent {agent_id}: "
                           f"Return: {performance.total_return:.2f}%, "
                           f"Sharpe: {performance.sharpe_ratio:.2f}, "
                           f"Win Rate: {performance.win_rate:.1f}%")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error assessing strategy performance: {e}")
            raise
    
    async def assess_agent_performance(self, agent_id: str, 
                                     assessment_period_hours: int = 24) -> AgentAssessment:
        """Assess overall agent performance over a period"""
        try:
            # Define assessment period
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=assessment_period_hours)
            
            # Filter strategies for this agent and period
            agent_strategies = [
                perf for perf in self.strategy_performances
                if perf.agent_id == agent_id and start_time <= perf.timestamp <= end_time
            ]
            
            if not agent_strategies:
                self.logger.warning(f"No strategies found for agent {agent_id} in assessment period")
                return AgentAssessment(
                    agent_id=agent_id,
                    agent_type="unknown",
                    assessment_period=(start_time, end_time)
                )
            
            # Create assessment
            assessment = AgentAssessment(
                agent_id=agent_id,
                agent_type=self._determine_agent_type(agent_strategies),
                assessment_period=(start_time, end_time)
            )
            
            # Calculate basic metrics
            assessment.strategies_evaluated = len(agent_strategies)
            assessment.successful_strategies = sum(
                1 for s in agent_strategies if s.total_return > 0
            )
            
            # Performance metrics
            returns = [s.total_return for s in agent_strategies]
            assessment.avg_strategy_performance = statistics.mean(returns) if returns else 0.0
            assessment.best_strategy_return = max(returns) if returns else 0.0
            assessment.worst_strategy_return = min(returns) if returns else 0.0
            
            # Information gathering metrics
            assessment.avg_data_sources = statistics.mean(
                [s.data_sources_used for s in agent_strategies]
            ) if agent_strategies else 0.0
            
            assessment.avg_information_freshness = statistics.mean(
                [s.information_freshness for s in agent_strategies]
            ) if agent_strategies else 0.0
            
            assessment.avg_information_relevance = statistics.mean(
                [s.information_relevance for s in agent_strategies]
            ) if agent_strategies else 0.0
            
            # Decision-making metrics
            assessment.avg_decision_time = statistics.mean(
                [s.decision_time for s in agent_strategies]
            ) if agent_strategies else 0.0
            
            assessment.avg_confidence = statistics.mean(
                [s.confidence_level for s in agent_strategies]
            ) if agent_strategies else 0.0
            
            # Calculate decision accuracy (strategies with positive returns)
            assessment.decision_accuracy = (
                assessment.successful_strategies / assessment.strategies_evaluated * 100
                if assessment.strategies_evaluated > 0 else 0.0
            )
            
            # Calculate component scores
            assessment.information_gathering_score = self._calculate_information_score(agent_strategies)
            assessment.decision_making_score = self._calculate_decision_score(agent_strategies)
            assessment.risk_management_score = self._calculate_risk_score(agent_strategies)
            assessment.adaptability_score = self._calculate_adaptability_score(agent_strategies)
            
            # Calculate overall score
            assessment.overall_score = statistics.mean([
                assessment.information_gathering_score,
                assessment.decision_making_score,
                assessment.risk_management_score,
                assessment.adaptability_score
            ])
            
            # Generate recommendations
            assessment.strengths, assessment.weaknesses, assessment.recommendations = \
                self._generate_recommendations(assessment, agent_strategies)
            
            # Store assessment
            self.agent_assessments[agent_id] = assessment
            self.assessment_history.append(assessment)
            
            # Log assessment
            self.logger.info(f"Agent {agent_id} assessment complete: "
                           f"Overall Score: {assessment.overall_score:.1f}, "
                           f"Strategies: {assessment.strategies_evaluated}, "
                           f"Success Rate: {assessment.decision_accuracy:.1f}%")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing agent {agent_id}: {e}")
            raise
    
    def _determine_agent_type(self, strategies: List[StrategyPerformance]) -> str:
        """Determine agent type based on strategy patterns"""
        if not strategies:
            return "unknown"
        
        # Count strategy types
        type_counts = defaultdict(int)
        for strategy in strategies:
            type_counts[strategy.strategy_type.value] += 1
        
        # Return most common type
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "mixed"
    
    def _calculate_information_score(self, strategies: List[StrategyPerformance]) -> float:
        """Calculate information gathering effectiveness score"""
        if not strategies:
            return 0.0
        
        scores = []
        for strategy in strategies:
            # Data source diversity score
            source_score = min(strategy.data_sources_used / self.benchmark_metrics['min_data_sources'] * 100, 100)
            
            # Information freshness score (lower age is better)
            freshness_score = max(0, 100 - (strategy.information_freshness / self.benchmark_metrics['max_information_age'] * 100))
            
            # Information relevance score
            relevance_score = strategy.information_relevance
            
            # Combined score
            info_score = statistics.mean([source_score, freshness_score, relevance_score])
            scores.append(info_score)
        
        return statistics.mean(scores)
    
    def _calculate_decision_score(self, strategies: List[StrategyPerformance]) -> float:
        """Calculate decision-making effectiveness score"""
        if not strategies:
            return 0.0
        
        scores = []
        for strategy in strategies:
            # Decision speed score (faster is better, up to a point)
            speed_score = max(0, 100 - (strategy.decision_time / self.benchmark_metrics['max_decision_time'] * 100))
            
            # Confidence score
            confidence_score = strategy.confidence_level
            
            # Performance-based score
            performance_score = max(0, min(100, strategy.total_return * 10))  # Scale return to 0-100
            
            # Combined score
            decision_score = statistics.mean([speed_score, confidence_score, performance_score])
            scores.append(decision_score)
        
        return statistics.mean(scores)
    
    def _calculate_risk_score(self, strategies: List[StrategyPerformance]) -> float:
        """Calculate risk management effectiveness score"""
        if not strategies:
            return 0.0
        
        scores = []
        for strategy in strategies:
            # Drawdown score (lower drawdown is better)
            drawdown_score = max(0, 100 - (strategy.max_drawdown / self.benchmark_metrics['max_drawdown_threshold'] * 100))
            
            # Sharpe ratio score
            sharpe_score = min(100, strategy.sharpe_ratio / self.benchmark_metrics['min_sharpe_ratio'] * 100)
            
            # Risk assessment score
            risk_assessment_score = 100 - strategy.risk_score  # Lower risk score is better
            
            # Combined score
            risk_score = statistics.mean([drawdown_score, sharpe_score, risk_assessment_score])
            scores.append(risk_score)
        
        return statistics.mean(scores)
    
    def _calculate_adaptability_score(self, strategies: List[StrategyPerformance]) -> float:
        """Calculate adaptability score based on performance across different market conditions"""
        if not strategies:
            return 0.0
        
        # Group strategies by market conditions
        condition_performance = defaultdict(list)
        for strategy in strategies:
            condition_performance[strategy.market_conditions].append(strategy.total_return)
        
        # Calculate consistency across conditions
        if len(condition_performance) < 2:
            return 50.0  # Neutral score if only one condition
        
        # Calculate average performance per condition
        avg_performances = [statistics.mean(returns) for returns in condition_performance.values()]
        
        # Adaptability is measured by consistency (lower variance is better)
        if len(avg_performances) > 1:
            variance = statistics.variance(avg_performances)
            adaptability_score = max(0, 100 - variance * 10)  # Scale variance
        else:
            adaptability_score = 50.0
        
        return adaptability_score
    
    def _generate_recommendations(self, assessment: AgentAssessment, 
                                strategies: List[StrategyPerformance]) -> Tuple[List[str], List[str], List[str]]:
        """Generate strengths, weaknesses, and recommendations"""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze information gathering
        if assessment.information_gathering_score > 80:
            strengths.append("Excellent information gathering capabilities")
        elif assessment.information_gathering_score < 60:
            weaknesses.append("Poor information gathering effectiveness")
            recommendations.append("Improve data source diversity and information relevance")
        
        # Analyze decision making
        if assessment.decision_making_score > 80:
            strengths.append("Strong decision-making abilities")
        elif assessment.decision_making_score < 60:
            weaknesses.append("Inconsistent decision-making performance")
            recommendations.append("Focus on improving decision confidence and speed")
        
        # Analyze risk management
        if assessment.risk_management_score > 80:
            strengths.append("Effective risk management")
        elif assessment.risk_management_score < 60:
            weaknesses.append("Inadequate risk management")
            recommendations.append("Implement stricter risk controls and drawdown limits")
        
        # Analyze adaptability
        if assessment.adaptability_score > 80:
            strengths.append("High adaptability to market conditions")
        elif assessment.adaptability_score < 60:
            weaknesses.append("Limited adaptability to changing market conditions")
            recommendations.append("Develop more robust strategies for different market regimes")
        
        # Performance-based recommendations
        if assessment.decision_accuracy < 50:
            recommendations.append("Review strategy selection criteria and improve backtesting")
        
        if assessment.avg_decision_time > 60:
            recommendations.append("Optimize decision-making process to reduce latency")
        
        return strengths, weaknesses, recommendations
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive assessment report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'assessment_summary': {
                    'total_strategies_assessed': len(self.strategy_performances),
                    'total_agents_assessed': len(self.agent_assessments),
                    'assessment_period': '24 hours'
                },
                'agent_rankings': [],
                'top_performing_strategies': [],
                'system_recommendations': [],
                'benchmark_comparison': {}
            }
            
            # Rank agents by overall score
            sorted_agents = sorted(
                self.agent_assessments.values(),
                key=lambda x: x.overall_score,
                reverse=True
            )
            
            for i, agent in enumerate(sorted_agents[:10]):  # Top 10
                report['agent_rankings'].append({
                    'rank': i + 1,
                    'agent_id': agent.agent_id,
                    'agent_type': agent.agent_type,
                    'overall_score': agent.overall_score,
                    'strategies_evaluated': agent.strategies_evaluated,
                    'success_rate': agent.decision_accuracy
                })
            
            # Top performing strategies
            top_strategies = sorted(
                self.strategy_performances,
                key=lambda x: x.total_return,
                reverse=True
            )[:10]
            
            for strategy in top_strategies:
                report['top_performing_strategies'].append({
                    'strategy_id': strategy.strategy_id,
                    'agent_id': strategy.agent_id,
                    'strategy_type': strategy.strategy_type.value,
                    'total_return': strategy.total_return,
                    'sharpe_ratio': strategy.sharpe_ratio,
                    'win_rate': strategy.win_rate
                })
            
            # System-wide recommendations
            if sorted_agents:
                avg_info_score = statistics.mean([a.information_gathering_score for a in sorted_agents])
                avg_decision_score = statistics.mean([a.decision_making_score for a in sorted_agents])
                avg_risk_score = statistics.mean([a.risk_management_score for a in sorted_agents])
                
                if avg_info_score < 70:
                    report['system_recommendations'].append(
                        "System-wide improvement needed in information gathering capabilities"
                    )
                
                if avg_decision_score < 70:
                    report['system_recommendations'].append(
                        "Focus on improving decision-making processes across all agents"
                    )
                
                if avg_risk_score < 70:
                    report['system_recommendations'].append(
                        "Strengthen risk management protocols system-wide"
                    )
            
            # Save report
            report_file = Path(f"reports/strategy_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Comprehensive assessment report generated: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    async def get_real_strategy_assessments(self):
        """Get real strategy assessment data from the trading system"""
        # This would connect to real strategy performance data
        # For now, return empty list until real strategies are implemented
        self.logger.info("Real strategy assessment data not yet implemented")
        return []

async def main():
    """Main function for testing the assessment framework"""
    framework = StrategyAssessmentFramework()
    
    print("🧠 Starting Strategy Assessment Framework...")
    
    # Get real strategy assessments (currently returns empty list)
    await framework.get_real_strategy_assessments()
    
    # Generate comprehensive report
    report = await framework.generate_comprehensive_report()
    
    print(f"📊 Assessment complete! Report generated with {len(report['agent_rankings'])} agents assessed.")
    print(f"📈 Top performing agent: {report['agent_rankings'][0]['agent_id'] if report['agent_rankings'] else 'None'}")
    
if __name__ == "__main__":
    asyncio.run(main())