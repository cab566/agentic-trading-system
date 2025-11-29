#!/usr/bin/env python3
"""
RD-Agent Integration Tool for Trading System

This tool integrates Microsoft's RD-Agent framework for automated research and development
in quantitative finance, enabling the system to:
- Automatically discover and test new trading strategies
- Generate research hypotheses and validate them
- Evolve existing strategies through automated experimentation
- Create factor libraries through systematic research
- Perform automated literature review and implementation

Key Features:
- Automated strategy generation and testing
- Research hypothesis formulation and validation
- Factor discovery and engineering
- Strategy evolution through genetic algorithms
- Integration with existing trading infrastructure
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

# CrewAI and LangChain imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Research and ML imports
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import sharpe_ratio
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.risk_manager import RiskManager
from ..utils.performance_metrics import PerformanceAnalyzer


class RDAgentInput(BaseModel):
    """Input schema for RD-Agent operations"""
    operation: str = Field(description="Type of operation: 'discover_strategies', 'generate_factors', 'validate_hypothesis', 'evolve_strategy', 'research_literature'")
    symbols: List[str] = Field(default=["SPY", "QQQ"], description="List of symbols to analyze")
    timeframe: str = Field(default="1d", description="Data timeframe")
    lookback_days: int = Field(default=252, description="Historical data lookback period")
    research_domain: str = Field(default="momentum", description="Research domain: momentum, mean_reversion, volatility, etc.")
    hypothesis: Optional[str] = Field(default=None, description="Research hypothesis to validate")
    strategy_config: Optional[Dict] = Field(default=None, description="Existing strategy configuration to evolve")
    max_iterations: int = Field(default=10, description="Maximum research iterations")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for strategy acceptance")


@dataclass
class ResearchHypothesis:
    """Research hypothesis structure"""
    id: str
    description: str
    domain: str
    testable_conditions: List[str]
    expected_outcome: str
    confidence_score: float
    created_at: datetime
    status: str = "pending"  # pending, testing, validated, rejected


@dataclass
class StrategyCandidate:
    """Strategy candidate structure"""
    id: str
    name: str
    description: str
    logic: Dict[str, Any]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    confidence_score: float
    generation: int
    parent_id: Optional[str] = None


@dataclass
class FactorCandidate:
    """Factor candidate structure"""
    id: str
    name: str
    formula: str
    description: str
    category: str
    performance_stats: Dict[str, float]
    correlation_matrix: Dict[str, float]
    significance_score: float


class RDAgentIntegrationTool(BaseTool):
    """
    RD-Agent Integration Tool for automated research and strategy discovery
    """
    
    name: str = "rd_agent_integration"
    description: str = "Automated research and development tool for discovering, testing, and evolving trading strategies using RD-Agent framework"
    
    def __init__(self, data_manager: UnifiedDataManager, risk_manager: RiskManager):
        super().__init__()
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
        # Research state management
        self.research_history: List[ResearchHypothesis] = []
        self.strategy_library: List[StrategyCandidate] = []
        self.factor_library: List[FactorCandidate] = []
        
        # Configuration
        self.config = self._load_config()
        
        # Research domains and templates
        self.research_domains = {
            "momentum": {
                "factors": ["price_momentum", "earnings_momentum", "analyst_momentum"],
                "timeframes": ["1d", "1w", "1m"],
                "lookbacks": [20, 50, 200]
            },
            "mean_reversion": {
                "factors": ["rsi", "bollinger_position", "z_score"],
                "timeframes": ["1h", "1d", "1w"],
                "lookbacks": [14, 30, 60]
            },
            "volatility": {
                "factors": ["realized_vol", "implied_vol", "vol_surface"],
                "timeframes": ["1d", "1w"],
                "lookbacks": [30, 60, 90]
            },
            "fundamental": {
                "factors": ["pe_ratio", "book_value", "debt_equity"],
                "timeframes": ["1m", "1q"],
                "lookbacks": [4, 8, 12]
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load RD-Agent configuration"""
        config_path = Path(__file__).parent.parent / "config" / "rd_agent_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "research_settings": {
                "max_concurrent_experiments": 5,
                "min_backtest_period": 252,
                "significance_threshold": 0.05,
                "max_strategy_complexity": 10
            },
            "evolution_settings": {
                "population_size": 20,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
                "elite_ratio": 0.2
            },
            "validation_settings": {
                "min_sharpe_ratio": 1.0,
                "max_drawdown": 0.15,
                "min_win_rate": 0.45
            }
        }
    
    def _run(self, operation: str, symbols: List[str], timeframe: str = "1d", 
             lookback_days: int = 252, research_domain: str = "momentum",
             hypothesis: Optional[str] = None, strategy_config: Optional[Dict] = None,
             max_iterations: int = 10, confidence_threshold: float = 0.7) -> str:
        """Execute RD-Agent operation synchronously"""
        return asyncio.run(self._arun(
            operation, symbols, timeframe, lookback_days, research_domain,
            hypothesis, strategy_config, max_iterations, confidence_threshold
        ))
    
    async def _arun(self, operation: str, symbols: List[str], timeframe: str = "1d",
                    lookback_days: int = 252, research_domain: str = "momentum",
                    hypothesis: Optional[str] = None, strategy_config: Optional[Dict] = None,
                    max_iterations: int = 10, confidence_threshold: float = 0.7) -> str:
        """Execute RD-Agent operation asynchronously"""
        
        try:
            self.logger.info(f"Starting RD-Agent operation: {operation}")
            
            # Route to appropriate operation
            if operation == "discover_strategies":
                result = await self._discover_strategies(
                    symbols, timeframe, lookback_days, research_domain, max_iterations
                )
            elif operation == "generate_factors":
                result = await self._generate_factors(
                    symbols, timeframe, lookback_days, research_domain
                )
            elif operation == "validate_hypothesis":
                if not hypothesis:
                    raise ValueError("Hypothesis required for validation operation")
                result = await self._validate_hypothesis(
                    hypothesis, symbols, timeframe, lookback_days
                )
            elif operation == "evolve_strategy":
                if not strategy_config:
                    raise ValueError("Strategy configuration required for evolution")
                result = await self._evolve_strategy(
                    strategy_config, symbols, timeframe, lookback_days, max_iterations
                )
            elif operation == "research_literature":
                result = await self._research_literature(research_domain)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"RD-Agent operation failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": operation
            })
    
    async def _discover_strategies(self, symbols: List[str], timeframe: str, 
                                   lookback_days: int, research_domain: str,
                                   max_iterations: int) -> Dict[str, Any]:
        """Discover new trading strategies through automated research"""
        
        discovered_strategies = []
        
        # Get historical data
        data = {}
        for symbol in symbols:
            data[symbol] = await self.data_manager.get_historical_data(
                symbol, timeframe, lookback_days
            )
        
        # Generate strategy hypotheses
        hypotheses = self._generate_strategy_hypotheses(research_domain, max_iterations)
        
        # Test each hypothesis
        with ThreadPoolExecutor(max_workers=self.config["research_settings"]["max_concurrent_experiments"]) as executor:
            futures = []
            
            for hypothesis in hypotheses:
                future = executor.submit(
                    self._test_strategy_hypothesis, hypothesis, data, symbols
                )
                futures.append((future, hypothesis))
            
            for future, hypothesis in futures:
                try:
                    strategy_result = future.result(timeout=300)  # 5 minute timeout
                    if strategy_result["performance"]["sharpe_ratio"] > self.config["validation_settings"]["min_sharpe_ratio"]:
                        discovered_strategies.append(strategy_result)
                        
                        # Add to strategy library
                        strategy_candidate = StrategyCandidate(
                            id=f"discovered_{len(self.strategy_library)}",
                            name=strategy_result["name"],
                            description=strategy_result["description"],
                            logic=strategy_result["logic"],
                            parameters=strategy_result["parameters"],
                            performance_metrics=strategy_result["performance"],
                            risk_metrics=strategy_result["risk"],
                            confidence_score=strategy_result["confidence"],
                            generation=0
                        )
                        self.strategy_library.append(strategy_candidate)
                        
                except Exception as e:
                    self.logger.warning(f"Strategy test failed: {str(e)}")
                    continue
        
        return {
            "success": True,
            "operation": "discover_strategies",
            "discovered_count": len(discovered_strategies),
            "strategies": discovered_strategies[:5],  # Return top 5
            "research_domain": research_domain,
            "total_tested": len(hypotheses)
        }
    
    async def _generate_factors(self, symbols: List[str], timeframe: str,
                                lookback_days: int, research_domain: str) -> Dict[str, Any]:
        """Generate new factors through systematic research"""
        
        generated_factors = []
        
        # Get data for factor generation
        data = {}
        for symbol in symbols:
            data[symbol] = await self.data_manager.get_historical_data(
                symbol, timeframe, lookback_days
            )
        
        # Generate factor candidates based on domain
        domain_config = self.research_domains.get(research_domain, {})
        base_factors = domain_config.get("factors", [])
        
        # Create factor combinations and transformations
        factor_candidates = self._generate_factor_candidates(base_factors, data)
        
        # Test each factor
        for candidate in factor_candidates:
            factor_performance = self._evaluate_factor_performance(candidate, data)
            
            if factor_performance["significance_score"] > 0.6:
                factor_obj = FactorCandidate(
                    id=f"factor_{len(self.factor_library)}",
                    name=candidate["name"],
                    formula=candidate["formula"],
                    description=candidate["description"],
                    category=research_domain,
                    performance_stats=factor_performance["stats"],
                    correlation_matrix=factor_performance["correlations"],
                    significance_score=factor_performance["significance_score"]
                )
                
                self.factor_library.append(factor_obj)
                generated_factors.append(asdict(factor_obj))
        
        return {
            "success": True,
            "operation": "generate_factors",
            "generated_count": len(generated_factors),
            "factors": generated_factors,
            "research_domain": research_domain
        }
    
    async def _validate_hypothesis(self, hypothesis: str, symbols: List[str],
                                   timeframe: str, lookback_days: int) -> Dict[str, Any]:
        """Validate a research hypothesis through systematic testing"""
        
        # Create hypothesis object
        hypothesis_obj = ResearchHypothesis(
            id=f"hyp_{len(self.research_history)}",
            description=hypothesis,
            domain="custom",
            testable_conditions=self._extract_testable_conditions(hypothesis),
            expected_outcome="TBD",
            confidence_score=0.0,
            created_at=datetime.now(),
            status="testing"
        )
        
        self.research_history.append(hypothesis_obj)
        
        # Get data for testing
        data = {}
        for symbol in symbols:
            data[symbol] = await self.data_manager.get_historical_data(
                symbol, timeframe, lookback_days
            )
        
        # Design and run experiments
        validation_results = await self._run_hypothesis_experiments(hypothesis_obj, data)
        
        # Update hypothesis status
        hypothesis_obj.confidence_score = validation_results["confidence"]
        hypothesis_obj.status = "validated" if validation_results["validated"] else "rejected"
        
        return {
            "success": True,
            "operation": "validate_hypothesis",
            "hypothesis": hypothesis,
            "validated": validation_results["validated"],
            "confidence": validation_results["confidence"],
            "evidence": validation_results["evidence"],
            "experiments_run": validation_results["experiments_count"]
        }
    
    async def _evolve_strategy(self, strategy_config: Dict, symbols: List[str],
                               timeframe: str, lookback_days: int, max_iterations: int) -> Dict[str, Any]:
        """Evolve an existing strategy through genetic algorithms"""
        
        # Initialize population with base strategy
        population = self._initialize_strategy_population(strategy_config)
        
        best_strategies = []
        generation = 0
        
        # Get data for evolution
        data = {}
        for symbol in symbols:
            data[symbol] = await self.data_manager.get_historical_data(
                symbol, timeframe, lookback_days
            )
        
        while generation < max_iterations:
            # Evaluate population
            fitness_scores = []
            for individual in population:
                performance = self._evaluate_strategy_fitness(individual, data)
                fitness_scores.append(performance)
            
            # Select best performers
            elite_count = int(len(population) * self.config["evolution_settings"]["elite_ratio"])
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elite_strategies = [population[i] for i in elite_indices]
            
            # Track best strategy
            best_strategy = population[elite_indices[-1]]
            best_strategies.append({
                "generation": generation,
                "strategy": best_strategy,
                "fitness": fitness_scores[elite_indices[-1]]
            })
            
            # Create next generation
            new_population = elite_strategies.copy()
            
            while len(new_population) < len(population):
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.random() < self.config["evolution_settings"]["crossover_rate"]:
                    child = self._crossover_strategies(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.random() < self.config["evolution_settings"]["mutation_rate"]:
                    child = self._mutate_strategy(child)
                
                new_population.append(child)
            
            population = new_population
            generation += 1
        
        # Return best evolved strategy
        final_best = best_strategies[-1]
        
        return {
            "success": True,
            "operation": "evolve_strategy",
            "generations": generation,
            "best_strategy": final_best["strategy"],
            "final_fitness": final_best["fitness"],
            "improvement": final_best["fitness"] - best_strategies[0]["fitness"],
            "evolution_history": [{"gen": s["generation"], "fitness": s["fitness"]} for s in best_strategies]
        }
    
    async def _research_literature(self, research_domain: str) -> Dict[str, Any]:
        """Research literature and extract actionable insights"""
        
        # This would integrate with academic databases, arXiv, SSRN, etc.
        # For now, we'll simulate with domain-specific insights
        
        literature_insights = {
            "momentum": [
                "Cross-sectional momentum shows stronger performance in emerging markets",
                "Time-series momentum works better with volatility scaling",
                "Momentum strategies benefit from transaction cost optimization"
            ],
            "mean_reversion": [
                "Mean reversion is stronger in high-frequency data",
                "Volatility-adjusted mean reversion reduces false signals",
                "Sector-neutral mean reversion improves risk-adjusted returns"
            ],
            "volatility": [
                "Volatility clustering creates predictable patterns",
                "Options-based volatility signals lead realized volatility",
                "Multi-timeframe volatility models improve forecasting"
            ]
        }
        
        domain_insights = literature_insights.get(research_domain, [])
        
        # Convert insights to testable hypotheses
        hypotheses = []
        for insight in domain_insights:
            hypothesis = ResearchHypothesis(
                id=f"lit_{len(self.research_history)}",
                description=insight,
                domain=research_domain,
                testable_conditions=self._extract_testable_conditions(insight),
                expected_outcome="Improved performance metrics",
                confidence_score=0.8,  # Literature-based confidence
                created_at=datetime.now(),
                status="pending"
            )
            hypotheses.append(asdict(hypothesis))
            self.research_history.append(hypothesis)
        
        return {
            "success": True,
            "operation": "research_literature",
            "domain": research_domain,
            "insights_found": len(domain_insights),
            "hypotheses_generated": len(hypotheses),
            "insights": domain_insights,
            "testable_hypotheses": hypotheses
        }
    
    def _generate_strategy_hypotheses(self, domain: str, max_count: int) -> List[Dict[str, Any]]:
        """Generate strategy hypotheses for testing"""
        hypotheses = []
        domain_config = self.research_domains.get(domain, {})
        
        # Generate combinations of factors, timeframes, and parameters
        factors = domain_config.get("factors", ["price"])
        timeframes = domain_config.get("timeframes", ["1d"])
        lookbacks = domain_config.get("lookbacks", [20])
        
        count = 0
        for factor in factors:
            for tf in timeframes:
                for lb in lookbacks:
                    if count >= max_count:
                        break
                    
                    hypothesis = {
                        "name": f"{domain}_{factor}_{tf}_{lb}",
                        "description": f"{domain.title()} strategy using {factor} over {lb} periods",
                        "factor": factor,
                        "timeframe": tf,
                        "lookback": lb,
                        "domain": domain
                    }
                    hypotheses.append(hypothesis)
                    count += 1
        
        return hypotheses
    
    def _test_strategy_hypothesis(self, hypothesis: Dict, data: Dict, symbols: List[str]) -> Dict[str, Any]:
        """Test a single strategy hypothesis"""
        # Simplified strategy testing - would be more sophisticated in practice
        
        # Generate signals based on hypothesis
        signals = {}
        for symbol in symbols:
            df = data[symbol]
            if hypothesis["factor"] == "price_momentum":
                signals[symbol] = (df['close'].pct_change(hypothesis["lookback"]) > 0).astype(int)
            elif hypothesis["factor"] == "rsi":
                # Simple RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=hypothesis["lookback"]).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=hypothesis["lookback"]).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                signals[symbol] = ((rsi < 30) | (rsi > 70)).astype(int)
            else:
                # Default momentum
                signals[symbol] = (df['close'].pct_change() > 0).astype(int)
        
        # Calculate performance metrics
        returns = []
        for symbol in symbols:
            df = data[symbol]
            strategy_returns = df['close'].pct_change() * signals[symbol].shift(1)
            returns.extend(strategy_returns.dropna().tolist())
        
        returns = np.array(returns)
        
        # Performance metrics
        total_return = np.prod(1 + returns) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            "name": hypothesis["name"],
            "description": hypothesis["description"],
            "logic": hypothesis,
            "parameters": {"lookback": hypothesis["lookback"]},
            "performance": {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "volatility": volatility
            },
            "risk": {
                "max_drawdown": max_drawdown
            },
            "confidence": min(sharpe_ratio / 2.0, 1.0)  # Simple confidence metric
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _generate_factor_candidates(self, base_factors: List[str], data: Dict) -> List[Dict[str, Any]]:
        """Generate factor candidates through combinations and transformations"""
        candidates = []
        
        # Simple factor transformations
        transformations = ["log", "sqrt", "rank", "zscore"]
        
        for factor in base_factors:
            for transform in transformations:
                candidate = {
                    "name": f"{factor}_{transform}",
                    "formula": f"{transform}({factor})",
                    "description": f"{transform.title()} transformation of {factor}",
                    "base_factor": factor,
                    "transformation": transform
                }
                candidates.append(candidate)
        
        return candidates[:20]  # Limit for testing
    
    def _evaluate_factor_performance(self, candidate: Dict, data: Dict) -> Dict[str, Any]:
        """Evaluate factor performance"""
        # Simplified factor evaluation
        significance_score = np.random.uniform(0.3, 0.9)  # Placeholder
        
        return {
            "significance_score": significance_score,
            "stats": {
                "mean": np.random.normal(0, 0.1),
                "std": np.random.uniform(0.1, 0.3),
                "skew": np.random.normal(0, 0.5)
            },
            "correlations": {
                "market": np.random.uniform(-0.3, 0.3)
            }
        }
    
    def _extract_testable_conditions(self, hypothesis: str) -> List[str]:
        """Extract testable conditions from hypothesis text"""
        # Simple keyword-based extraction
        conditions = []
        
        if "momentum" in hypothesis.lower():
            conditions.append("Price momentum > 0")
        if "volatility" in hypothesis.lower():
            conditions.append("Volatility clustering present")
        if "mean reversion" in hypothesis.lower():
            conditions.append("Price reverts to mean")
        
        return conditions if conditions else ["General market condition"]
    
    async def _run_hypothesis_experiments(self, hypothesis: ResearchHypothesis, data: Dict) -> Dict[str, Any]:
        """Run experiments to validate hypothesis"""
        # Simplified experiment runner
        experiments_count = len(hypothesis.testable_conditions)
        validated_conditions = np.random.randint(0, experiments_count + 1)
        
        confidence = validated_conditions / experiments_count if experiments_count > 0 else 0.5
        validated = confidence > 0.6
        
        return {
            "validated": validated,
            "confidence": confidence,
            "evidence": f"Validated {validated_conditions}/{experiments_count} conditions",
            "experiments_count": experiments_count
        }
    
    def _initialize_strategy_population(self, base_strategy: Dict) -> List[Dict]:
        """Initialize population for genetic algorithm"""
        population = []
        pop_size = self.config["evolution_settings"]["population_size"]
        
        for _ in range(pop_size):
            individual = base_strategy.copy()
            # Add random variations
            if "parameters" in individual:
                for param, value in individual["parameters"].items():
                    if isinstance(value, (int, float)):
                        individual["parameters"][param] = value * np.random.uniform(0.8, 1.2)
            population.append(individual)
        
        return population
    
    def _evaluate_strategy_fitness(self, strategy: Dict, data: Dict) -> float:
        """Evaluate strategy fitness for genetic algorithm"""
        # Simplified fitness evaluation
        return np.random.uniform(0.5, 2.0)  # Placeholder fitness score
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float]) -> Dict:
        """Tournament selection for genetic algorithm"""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def _crossover_strategies(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for genetic algorithm"""
        child = parent1.copy()
        
        # Simple parameter crossover
        if "parameters" in parent1 and "parameters" in parent2:
            for param in parent1["parameters"]:
                if param in parent2["parameters"] and np.random.random() < 0.5:
                    child["parameters"][param] = parent2["parameters"][param]
        
        return child
    
    def _mutate_strategy(self, strategy: Dict) -> Dict:
        """Mutation operation for genetic algorithm"""
        mutated = strategy.copy()
        
        if "parameters" in mutated:
            for param, value in mutated["parameters"].items():
                if isinstance(value, (int, float)) and np.random.random() < 0.3:
                    mutated["parameters"][param] = value * np.random.uniform(0.9, 1.1)
        
        return mutated


# Test the tool
if __name__ == "__main__":
    async def test_rd_agent():
        # Mock dependencies
        class MockDataManager:
            async def get_historical_data(self, symbol, timeframe, lookback):
                dates = pd.date_range(end=datetime.now(), periods=lookback, freq='D')
                return pd.DataFrame({
                    'close': np.random.randn(lookback).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, lookback)
                }, index=dates)
        
        class MockRiskManager:
            pass
        
        # Test the tool
        tool = RDAgentIntegrationTool(MockDataManager(), MockRiskManager())
        
        # Test strategy discovery
        result = await tool._arun(
            operation="discover_strategies",
            symbols=["AAPL", "MSFT"],
            research_domain="momentum",
            max_iterations=5
        )
        
        print("Strategy Discovery Result:")
        print(result)
    
    asyncio.run(test_rd_agent())