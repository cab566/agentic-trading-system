#!/usr/bin/env python3
"""
Earnings Surprise Predictor Tool

This tool predicts earnings surprises using alternative data sources including:
- Social media sentiment analysis
- Satellite imagery data
- Supply chain indicators
- Web scraping of company metrics
- Options flow analysis
- Analyst revision patterns
- Economic indicators

Features:
- Multi-source data integration
- Machine learning prediction models
- Real-time earnings monitoring
- Surprise probability scoring
- Historical accuracy tracking
- Alert generation for high-confidence predictions
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.config_manager import ConfigManager
from ..core.data_manager import UnifiedDataManager


class SurpriseDirection(Enum):
    """Direction of earnings surprise"""
    BEAT = "beat"
    MISS = "miss"
    INLINE = "inline"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DataSource(Enum):
    """Alternative data sources"""
    SOCIAL_SENTIMENT = "social_sentiment"
    SATELLITE_DATA = "satellite_data"
    SUPPLY_CHAIN = "supply_chain"
    WEB_METRICS = "web_metrics"
    OPTIONS_FLOW = "options_flow"
    ANALYST_REVISIONS = "analyst_revisions"
    ECONOMIC_INDICATORS = "economic_indicators"
    NEWS_SENTIMENT = "news_sentiment"
    INSIDER_TRADING = "insider_trading"


@dataclass
class EarningsEvent:
    """Earnings event data structure"""
    symbol: str
    company_name: str
    earnings_date: datetime
    fiscal_quarter: str
    fiscal_year: int
    estimated_eps: float
    estimated_revenue: float
    actual_eps: Optional[float] = None
    actual_revenue: Optional[float] = None
    surprise_eps: Optional[float] = None
    surprise_revenue: Optional[float] = None
    market_cap: float = 0.0
    sector: str = ""
    industry: str = ""


@dataclass
class AlternativeDataPoint:
    """Alternative data point"""
    source: DataSource
    symbol: str
    timestamp: datetime
    metric_name: str
    value: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class EarningsPrediction:
    """Earnings prediction result"""
    symbol: str
    earnings_date: datetime
    predicted_eps_surprise: float
    predicted_revenue_surprise: float
    surprise_direction: SurpriseDirection
    confidence_level: ConfidenceLevel
    probability_beat: float
    probability_miss: float
    key_factors: List[str]
    alternative_data_signals: Dict[str, float]
    model_version: str
    prediction_timestamp: datetime


@dataclass
class PredictionAlert:
    """Prediction alert for high-confidence predictions"""
    id: str
    symbol: str
    earnings_date: datetime
    prediction: EarningsPrediction
    alert_reason: str
    recommended_action: str
    risk_factors: List[str]
    created_at: datetime


@dataclass
class EarningsPredictorConfig:
    """Configuration for earnings predictor"""
    prediction_horizon_days: int = 30  # Days ahead to predict
    min_confidence_threshold: float = 0.7  # Minimum confidence for alerts
    model_retrain_frequency: int = 7  # Days between model retraining
    alternative_data_weight: float = 0.4  # Weight of alternative data vs traditional
    social_sentiment_lookback: int = 14  # Days of social sentiment to analyze
    satellite_data_lookback: int = 30  # Days of satellite data to analyze
    options_flow_lookback: int = 7  # Days of options flow to analyze
    analyst_revision_lookback: int = 21  # Days of analyst revisions to analyze
    max_predictions_per_run: int = 50  # Maximum predictions per execution
    enable_model_ensemble: bool = True  # Use ensemble of models
    feature_importance_threshold: float = 0.05  # Minimum feature importance


class EarningsSurprisePredictorInput(BaseModel):
    """Input model for earnings surprise predictor"""
    operation: str = Field(description="Operation: 'predict_earnings', 'analyze_upcoming', 'train_model', 'get_predictions', 'generate_alerts'")
    symbols: List[str] = Field(default=[], description="Symbols to analyze (empty for all upcoming)")
    days_ahead: int = Field(default=30, description="Days ahead to look for earnings")
    include_alternative_data: bool = Field(default=True, description="Include alternative data sources")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for results")
    retrain_model: bool = Field(default=False, description="Force model retraining")


class EarningsSurprisePredictorTool(BaseTool):
    """
    Earnings Surprise Predictor Tool for forecasting earnings beats/misses using alternative data
    """
    
    name: str = "earnings_surprise_predictor"
    description: str = "Predict earnings surprises using alternative data sources and machine learning models"
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.data_manager = UnifiedDataManager()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Model storage
        self.model_path = Path("models/earnings_predictor")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Prediction tracking
        self.predictions: List[EarningsPrediction] = []
        self.prediction_alerts: List[PredictionAlert] = []
        self.alternative_data: List[AlternativeDataPoint] = []
        
        # Models
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Data sources configuration
        self.data_sources = {
            "social_sentiment": {"enabled": True, "weight": 0.15},
            "satellite_data": {"enabled": True, "weight": 0.10},
            "supply_chain": {"enabled": True, "weight": 0.08},
            "web_metrics": {"enabled": True, "weight": 0.12},
            "options_flow": {"enabled": True, "weight": 0.20},
            "analyst_revisions": {"enabled": True, "weight": 0.25},
            "economic_indicators": {"enabled": True, "weight": 0.10}
        }
        
        # Initialize models
        self._initialize_models()
    
    def _load_config(self) -> EarningsPredictorConfig:
        """Load earnings predictor configuration"""
        try:
            config_data = self.config_manager.get_tool_config("earnings_surprise_predictor")
            return EarningsPredictorConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Using default config: {e}")
            return EarningsPredictorConfig()
    
    def _initialize_models(self):
        """Initialize prediction models"""
        
        # Load existing models if available
        try:
            self._load_trained_models()
        except Exception as e:
            self.logger.info(f"No existing models found, will train new ones: {e}")
            self._create_default_models()
    
    def _create_default_models(self):
        """Create default prediction models"""
        
        self.models = {
            "eps_surprise": {
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "gradient_boost": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "linear": LinearRegression()
            },
            "revenue_surprise": {
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "gradient_boost": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "linear": LinearRegression()
            }
        }
        
        self.scalers = {
            "eps_surprise": StandardScaler(),
            "revenue_surprise": StandardScaler()
        }
    
    def _run(self, operation: str, symbols: List[str] = [], days_ahead: int = 30,
             include_alternative_data: bool = True, confidence_threshold: float = 0.7,
             retrain_model: bool = False) -> str:
        """Execute earnings prediction synchronously"""
        return asyncio.run(self._arun(
            operation, symbols, days_ahead, include_alternative_data, 
            confidence_threshold, retrain_model
        ))
    
    async def _arun(self, operation: str, symbols: List[str] = [], days_ahead: int = 30,
                    include_alternative_data: bool = True, confidence_threshold: float = 0.7,
                    retrain_model: bool = False) -> str:
        """Execute earnings prediction asynchronously"""
        
        try:
            self.logger.info(f"Starting earnings prediction: {operation}")
            
            if operation == "predict_earnings":
                result = await self._predict_earnings(
                    symbols, days_ahead, include_alternative_data, confidence_threshold
                )
            elif operation == "analyze_upcoming":
                result = await self._analyze_upcoming_earnings(
                    days_ahead, include_alternative_data
                )
            elif operation == "train_model":
                result = await self._train_prediction_models(retrain_model)
            elif operation == "get_predictions":
                result = await self._get_existing_predictions(symbols, confidence_threshold)
            elif operation == "generate_alerts":
                result = await self._generate_prediction_alerts(confidence_threshold)
            elif operation == "backtest_model":
                result = await self._backtest_predictions(symbols, days_ahead)
            elif operation == "feature_importance":
                result = await self._analyze_feature_importance()
            else:
                result = {
                    "error": f"Unknown operation: {operation}",
                    "available_operations": [
                        "predict_earnings", "analyze_upcoming", "train_model",
                        "get_predictions", "generate_alerts", "backtest_model",
                        "feature_importance"
                    ]
                }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Earnings prediction failed: {e}")
            return json.dumps({"error": str(e)})
    
    async def _predict_earnings(self, symbols: List[str], days_ahead: int,
                               include_alternative_data: bool, confidence_threshold: float) -> Dict[str, Any]:
        """Predict earnings for specified symbols"""
        
        self.logger.info(f"Predicting earnings for {symbols}")
        
        predictions = {}
        
        # Get upcoming earnings events
        if not symbols:
            earnings_events = await self._get_upcoming_earnings(days_ahead)
            symbols = [event.symbol for event in earnings_events]
        else:
            earnings_events = await self._get_earnings_for_symbols(symbols, days_ahead)
        
        for event in earnings_events:
            try:
                # Collect features for prediction
                features = await self._collect_prediction_features(
                    event, include_alternative_data
                )
                
                # Make prediction
                prediction = await self._make_earnings_prediction(event, features)
                
                # Filter by confidence threshold
                if prediction.confidence_level.value in ["high", "very_high"] or \
                   prediction.probability_beat >= confidence_threshold:
                    predictions[event.symbol] = asdict(prediction)
                    self.predictions.append(prediction)
                
            except Exception as e:
                self.logger.error(f"Failed to predict earnings for {event.symbol}: {e}")
                predictions[event.symbol] = {"error": str(e)}
        
        return {
            "operation": "predict_earnings",
            "symbols_analyzed": len(symbols),
            "predictions_generated": len(predictions),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_upcoming_earnings(self, days_ahead: int, include_alternative_data: bool) -> Dict[str, Any]:
        """Analyze all upcoming earnings in the specified timeframe"""
        
        self.logger.info(f"Analyzing upcoming earnings for next {days_ahead} days")
        
        # Get upcoming earnings
        earnings_events = await self._get_upcoming_earnings(days_ahead)
        
        analysis = {
            "total_events": len(earnings_events),
            "by_date": {},
            "by_sector": {},
            "high_confidence_predictions": [],
            "summary_stats": {}
        }
        
        predictions = []
        
        for event in earnings_events[:self.config.max_predictions_per_run]:
            try:
                # Collect features
                features = await self._collect_prediction_features(event, include_alternative_data)
                
                # Make prediction
                prediction = await self._make_earnings_prediction(event, features)
                predictions.append(prediction)
                
                # Organize by date
                date_key = event.earnings_date.strftime("%Y-%m-%d")
                if date_key not in analysis["by_date"]:
                    analysis["by_date"][date_key] = []
                analysis["by_date"][date_key].append({
                    "symbol": event.symbol,
                    "prediction": asdict(prediction)
                })
                
                # Organize by sector
                sector = event.sector or "Unknown"
                if sector not in analysis["by_sector"]:
                    analysis["by_sector"][sector] = []
                analysis["by_sector"][sector].append({
                    "symbol": event.symbol,
                    "prediction": asdict(prediction)
                })
                
                # High confidence predictions
                if prediction.confidence_level.value in ["high", "very_high"]:
                    analysis["high_confidence_predictions"].append({
                        "symbol": event.symbol,
                        "prediction": asdict(prediction)
                    })
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {event.symbol}: {e}")
        
        # Calculate summary statistics
        if predictions:
            beat_predictions = len([p for p in predictions if p.surprise_direction == SurpriseDirection.BEAT])
            miss_predictions = len([p for p in predictions if p.surprise_direction == SurpriseDirection.MISS])
            
            analysis["summary_stats"] = {
                "total_predictions": len(predictions),
                "beat_predictions": beat_predictions,
                "miss_predictions": miss_predictions,
                "inline_predictions": len(predictions) - beat_predictions - miss_predictions,
                "avg_confidence": np.mean([p.probability_beat for p in predictions]),
                "high_confidence_count": len(analysis["high_confidence_predictions"])
            }
        
        return {
            "operation": "analyze_upcoming",
            "days_ahead": days_ahead,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _train_prediction_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Train or retrain prediction models"""
        
        self.logger.info("Training earnings prediction models")
        
        # Check if retraining is needed
        if not force_retrain and self._models_recently_trained():
            return {
                "operation": "train_model",
                "status": "skipped",
                "reason": "Models recently trained",
                "last_training": self._get_last_training_date()
            }
        
        # Collect training data
        training_data = await self._collect_training_data()
        
        if len(training_data) < 100:  # Minimum training samples
            return {
                "operation": "train_model",
                "status": "failed",
                "reason": "Insufficient training data",
                "samples_available": len(training_data)
            }
        
        # Prepare features and targets
        features_df = pd.DataFrame([d["features"] for d in training_data])
        eps_targets = [d["eps_surprise"] for d in training_data]
        revenue_targets = [d["revenue_surprise"] for d in training_data]
        
        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        
        training_results = {}
        
        # Train EPS surprise models
        eps_results = await self._train_model_ensemble(
            features_df, eps_targets, "eps_surprise"
        )
        training_results["eps_surprise"] = eps_results
        
        # Train revenue surprise models
        revenue_results = await self._train_model_ensemble(
            features_df, revenue_targets, "revenue_surprise"
        )
        training_results["revenue_surprise"] = revenue_results
        
        # Save trained models
        self._save_trained_models()
        
        return {
            "operation": "train_model",
            "status": "completed",
            "training_samples": len(training_data),
            "feature_count": len(self.feature_columns),
            "model_performance": training_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_existing_predictions(self, symbols: List[str], confidence_threshold: float) -> Dict[str, Any]:
        """Get existing predictions with optional filtering"""
        
        filtered_predictions = []
        
        for prediction in self.predictions:
            # Filter by symbols if specified
            if symbols and prediction.symbol not in symbols:
                continue
            
            # Filter by confidence
            if prediction.probability_beat < confidence_threshold:
                continue
            
            filtered_predictions.append(asdict(prediction))
        
        return {
            "operation": "get_predictions",
            "total_predictions": len(self.predictions),
            "filtered_predictions": len(filtered_predictions),
            "predictions": filtered_predictions,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_prediction_alerts(self, confidence_threshold: float) -> Dict[str, Any]:
        """Generate alerts for high-confidence predictions"""
        
        self.logger.info("Generating prediction alerts")
        
        new_alerts = []
        
        for prediction in self.predictions:
            # Check if alert already exists
            existing_alert = any(
                alert.symbol == prediction.symbol and 
                alert.earnings_date == prediction.earnings_date
                for alert in self.prediction_alerts
            )
            
            if existing_alert:
                continue
            
            # Check confidence threshold
            if prediction.probability_beat < confidence_threshold:
                continue
            
            # Create alert
            alert = self._create_prediction_alert(prediction)
            if alert:
                new_alerts.append(alert)
                self.prediction_alerts.append(alert)
        
        return {
            "operation": "generate_alerts",
            "new_alerts": len(new_alerts),
            "total_active_alerts": len(self.prediction_alerts),
            "alerts": [asdict(alert) for alert in new_alerts],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _backtest_predictions(self, symbols: List[str], days_back: int) -> Dict[str, Any]:
        """Backtest prediction accuracy"""
        
        self.logger.info(f"Backtesting predictions for {symbols}")
        
        # Get historical earnings data
        historical_events = await self._get_historical_earnings(symbols, days_back)
        
        backtest_results = {
            "total_events": len(historical_events),
            "predictions_made": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "by_symbol": {},
            "confusion_matrix": {
                "beat_predicted_beat": 0,
                "beat_predicted_miss": 0,
                "miss_predicted_beat": 0,
                "miss_predicted_miss": 0
            }
        }
        
        for event in historical_events:
            if event.actual_eps is None or event.surprise_eps is None:
                continue
            
            try:
                # Recreate features as they would have been before earnings
                features = await self._collect_historical_features(event)
                
                # Make prediction
                prediction = await self._make_earnings_prediction(event, features)
                
                # Compare with actual
                actual_direction = (SurpriseDirection.BEAT if event.surprise_eps > 0 
                                 else SurpriseDirection.MISS)
                
                is_correct = prediction.surprise_direction == actual_direction
                
                backtest_results["predictions_made"] += 1
                if is_correct:
                    backtest_results["correct_predictions"] += 1
                
                # Update confusion matrix
                if actual_direction == SurpriseDirection.BEAT:
                    if prediction.surprise_direction == SurpriseDirection.BEAT:
                        backtest_results["confusion_matrix"]["beat_predicted_beat"] += 1
                    else:
                        backtest_results["confusion_matrix"]["beat_predicted_miss"] += 1
                else:
                    if prediction.surprise_direction == SurpriseDirection.BEAT:
                        backtest_results["confusion_matrix"]["miss_predicted_beat"] += 1
                    else:
                        backtest_results["confusion_matrix"]["miss_predicted_miss"] += 1
                
                # By symbol results
                if event.symbol not in backtest_results["by_symbol"]:
                    backtest_results["by_symbol"][event.symbol] = {
                        "total": 0, "correct": 0, "accuracy": 0.0
                    }
                
                backtest_results["by_symbol"][event.symbol]["total"] += 1
                if is_correct:
                    backtest_results["by_symbol"][event.symbol]["correct"] += 1
                
            except Exception as e:
                self.logger.error(f"Backtest failed for {event.symbol}: {e}")
        
        # Calculate final accuracy
        if backtest_results["predictions_made"] > 0:
            backtest_results["accuracy"] = (
                backtest_results["correct_predictions"] / 
                backtest_results["predictions_made"]
            )
        
        # Calculate per-symbol accuracy
        for symbol_data in backtest_results["by_symbol"].values():
            if symbol_data["total"] > 0:
                symbol_data["accuracy"] = symbol_data["correct"] / symbol_data["total"]
        
        return {
            "operation": "backtest_model",
            "backtest_results": backtest_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across models"""
        
        if not self.models or not self.feature_columns:
            return {"error": "No trained models available"}
        
        feature_importance = {}
        
        for target_type, models in self.models.items():
            feature_importance[target_type] = {}
            
            for model_name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance[target_type][model_name] = dict(
                        zip(self.feature_columns, importances)
                    )
        
        # Calculate average importance across models
        avg_importance = {}
        for target_type in feature_importance:
            avg_importance[target_type] = {}
            for feature in self.feature_columns:
                importances = [
                    feature_importance[target_type][model].get(feature, 0)
                    for model in feature_importance[target_type]
                    if feature in feature_importance[target_type][model]
                ]
                avg_importance[target_type][feature] = np.mean(importances) if importances else 0
        
        return {
            "operation": "feature_importance",
            "feature_importance": feature_importance,
            "average_importance": avg_importance,
            "top_features": self._get_top_features(avg_importance),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_upcoming_earnings(self, days_ahead: int) -> List[EarningsEvent]:
        """Get upcoming earnings events (simulated for demo)"""
        
        # In production, this would fetch from earnings calendar APIs
        events = []
        current_date = datetime.now()
        
        # Sample symbols with upcoming earnings
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]
        
        for i, symbol in enumerate(symbols):
            earnings_date = current_date + timedelta(days=np.random.randint(1, days_ahead))
            
            event = EarningsEvent(
                symbol=symbol,
                company_name=f"{symbol} Inc.",
                earnings_date=earnings_date,
                fiscal_quarter="Q4",
                fiscal_year=2024,
                estimated_eps=np.random.uniform(1.0, 5.0),
                estimated_revenue=np.random.uniform(10e9, 100e9),
                market_cap=np.random.uniform(100e9, 3000e9),
                sector=np.random.choice(["Technology", "Consumer", "Healthcare", "Finance"]),
                industry=f"Industry_{i}"
            )
            
            events.append(event)
        
        return events
    
    async def _get_earnings_for_symbols(self, symbols: List[str], days_ahead: int) -> List[EarningsEvent]:
        """Get earnings events for specific symbols"""
        
        all_events = await self._get_upcoming_earnings(days_ahead)
        return [event for event in all_events if event.symbol in symbols]
    
    async def _collect_prediction_features(self, event: EarningsEvent, include_alternative_data: bool) -> Dict[str, float]:
        """Collect features for earnings prediction"""
        
        features = {}
        
        # Basic company features
        features.update(await self._get_company_features(event))
        
        # Market features
        features.update(await self._get_market_features(event))
        
        # Alternative data features
        if include_alternative_data:
            features.update(await self._get_alternative_data_features(event))
        
        return features
    
    async def _get_company_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get company-specific features"""
        
        # Simulated company features
        return {
            "market_cap": event.market_cap,
            "estimated_eps": event.estimated_eps,
            "estimated_revenue": event.estimated_revenue,
            "days_to_earnings": (event.earnings_date - datetime.now()).days,
            "quarter": int(event.fiscal_quarter[1]) if event.fiscal_quarter.startswith('Q') else 1,
            "pe_ratio": np.random.uniform(10, 50),
            "revenue_growth": np.random.uniform(-0.2, 0.5),
            "profit_margin": np.random.uniform(0.05, 0.3),
            "debt_to_equity": np.random.uniform(0.1, 2.0)
        }
    
    async def _get_market_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get market-related features"""
        
        # Simulated market features
        return {
            "spy_return_1w": np.random.uniform(-0.1, 0.1),
            "spy_return_1m": np.random.uniform(-0.2, 0.2),
            "sector_return_1w": np.random.uniform(-0.15, 0.15),
            "sector_return_1m": np.random.uniform(-0.25, 0.25),
            "vix_level": np.random.uniform(15, 40),
            "interest_rate": np.random.uniform(0.02, 0.06),
            "dollar_index": np.random.uniform(95, 110)
        }
    
    async def _get_alternative_data_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get alternative data features"""
        
        features = {}
        
        # Social sentiment features
        if self.data_sources["social_sentiment"]["enabled"]:
            features.update(await self._get_social_sentiment_features(event))
        
        # Satellite data features
        if self.data_sources["satellite_data"]["enabled"]:
            features.update(await self._get_satellite_features(event))
        
        # Supply chain features
        if self.data_sources["supply_chain"]["enabled"]:
            features.update(await self._get_supply_chain_features(event))
        
        # Web metrics features
        if self.data_sources["web_metrics"]["enabled"]:
            features.update(await self._get_web_metrics_features(event))
        
        # Options flow features
        if self.data_sources["options_flow"]["enabled"]:
            features.update(await self._get_options_flow_features(event))
        
        # Analyst revision features
        if self.data_sources["analyst_revisions"]["enabled"]:
            features.update(await self._get_analyst_revision_features(event))
        
        return features
    
    async def _get_social_sentiment_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get social media sentiment features"""
        
        # Simulated social sentiment data
        return {
            "twitter_sentiment": np.random.uniform(-1, 1),
            "reddit_sentiment": np.random.uniform(-1, 1),
            "stocktwits_sentiment": np.random.uniform(-1, 1),
            "social_volume": np.random.uniform(0, 1000),
            "sentiment_trend": np.random.uniform(-0.5, 0.5),
            "influencer_sentiment": np.random.uniform(-1, 1)
        }
    
    async def _get_satellite_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get satellite imagery features"""
        
        # Simulated satellite data (e.g., parking lot activity, shipping activity)
        return {
            "parking_lot_activity": np.random.uniform(0.5, 1.5),
            "shipping_activity": np.random.uniform(0.7, 1.3),
            "factory_activity": np.random.uniform(0.6, 1.4),
            "retail_foot_traffic": np.random.uniform(0.8, 1.2)
        }
    
    async def _get_supply_chain_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get supply chain indicators"""
        
        # Simulated supply chain data
        return {
            "supplier_performance": np.random.uniform(0.7, 1.3),
            "logistics_efficiency": np.random.uniform(0.8, 1.2),
            "inventory_levels": np.random.uniform(0.5, 1.5),
            "raw_material_costs": np.random.uniform(0.9, 1.1)
        }
    
    async def _get_web_metrics_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get web scraping metrics"""
        
        # Simulated web metrics
        return {
            "website_traffic": np.random.uniform(0.8, 1.2),
            "app_downloads": np.random.uniform(0.7, 1.3),
            "job_postings": np.random.uniform(0.6, 1.4),
            "patent_filings": np.random.uniform(0.9, 1.1),
            "news_mentions": np.random.uniform(0.5, 1.5)
        }
    
    async def _get_options_flow_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get options flow features"""
        
        # Simulated options flow data
        return {
            "call_put_ratio": np.random.uniform(0.5, 2.0),
            "unusual_options_volume": np.random.uniform(0.8, 1.2),
            "implied_volatility": np.random.uniform(0.2, 0.8),
            "options_skew": np.random.uniform(-0.2, 0.2),
            "gamma_exposure": np.random.uniform(-1000000, 1000000)
        }
    
    async def _get_analyst_revision_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Get analyst revision features"""
        
        # Simulated analyst data
        return {
            "eps_revisions_up": np.random.randint(0, 10),
            "eps_revisions_down": np.random.randint(0, 10),
            "revenue_revisions_up": np.random.randint(0, 8),
            "revenue_revisions_down": np.random.randint(0, 8),
            "price_target_change": np.random.uniform(-0.2, 0.2),
            "analyst_sentiment": np.random.uniform(-1, 1)
        }
    
    async def _make_earnings_prediction(self, event: EarningsEvent, features: Dict[str, float]) -> EarningsPrediction:
        """Make earnings prediction using trained models"""
        
        # Prepare features for prediction
        feature_vector = self._prepare_feature_vector(features)
        
        # Make predictions using ensemble
        eps_prediction = self._predict_with_ensemble(feature_vector, "eps_surprise")
        revenue_prediction = self._predict_with_ensemble(feature_vector, "revenue_surprise")
        
        # Determine surprise direction
        surprise_direction = (SurpriseDirection.BEAT if eps_prediction > 0 
                            else SurpriseDirection.MISS)
        
        # Calculate confidence
        confidence_level, probability_beat = self._calculate_confidence(
            eps_prediction, revenue_prediction, features
        )
        
        # Identify key factors
        key_factors = self._identify_key_factors(features)
        
        # Extract alternative data signals
        alt_data_signals = {k: v for k, v in features.items() 
                          if any(source in k for source in ["sentiment", "satellite", "supply", "web", "options"])}
        
        return EarningsPrediction(
            symbol=event.symbol,
            earnings_date=event.earnings_date,
            predicted_eps_surprise=eps_prediction,
            predicted_revenue_surprise=revenue_prediction,
            surprise_direction=surprise_direction,
            confidence_level=confidence_level,
            probability_beat=probability_beat,
            probability_miss=1.0 - probability_beat,
            key_factors=key_factors,
            alternative_data_signals=alt_data_signals,
            model_version="v1.0",
            prediction_timestamp=datetime.now()
        )
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for model prediction"""
        
        if not self.feature_columns:
            # Use all features if no trained model exists
            self.feature_columns = list(features.keys())
        
        # Create feature vector with proper ordering
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0.0))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _predict_with_ensemble(self, feature_vector: np.ndarray, target_type: str) -> float:
        """Make prediction using model ensemble"""
        
        if target_type not in self.models or not self.models[target_type]:
            # Fallback to simple heuristic if no trained models
            return np.random.uniform(-0.5, 0.5)
        
        predictions = []
        
        # Scale features if scaler is available
        if target_type in self.scalers:
            try:
                feature_vector = self.scalers[target_type].transform(feature_vector)
            except:
                pass  # Use unscaled features if scaling fails
        
        # Get predictions from each model
        for model_name, model in self.models[target_type].items():
            try:
                pred = model.predict(feature_vector)[0]
                predictions.append(pred)
            except:
                # Skip failed predictions
                continue
        
        # Return ensemble average or fallback
        return np.mean(predictions) if predictions else np.random.uniform(-0.5, 0.5)
    
    def _calculate_confidence(self, eps_pred: float, revenue_pred: float, features: Dict[str, float]) -> Tuple[ConfidenceLevel, float]:
        """Calculate prediction confidence"""
        
        # Simple confidence calculation based on prediction magnitude and feature quality
        eps_confidence = min(abs(eps_pred) * 2, 1.0)
        revenue_confidence = min(abs(revenue_pred) * 0.1, 1.0)
        
        # Factor in alternative data availability
        alt_data_score = len([k for k in features.keys() 
                            if any(source in k for source in ["sentiment", "satellite", "supply"])]) / 20.0
        
        overall_confidence = (eps_confidence + revenue_confidence + alt_data_score) / 3.0
        
        # Determine confidence level
        if overall_confidence >= 0.8:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif overall_confidence >= 0.7:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        # Calculate probability of beat
        probability_beat = 0.5 + (eps_pred * 0.3)  # Simple mapping
        probability_beat = max(0.0, min(1.0, probability_beat))
        
        return confidence_level, probability_beat
    
    def _identify_key_factors(self, features: Dict[str, float]) -> List[str]:
        """Identify key factors driving the prediction"""
        
        # Simple heuristic to identify important factors
        key_factors = []
        
        # Check for strong signals
        if features.get("twitter_sentiment", 0) > 0.5:
            key_factors.append("Positive social media sentiment")
        elif features.get("twitter_sentiment", 0) < -0.5:
            key_factors.append("Negative social media sentiment")
        
        if features.get("analyst_sentiment", 0) > 0.3:
            key_factors.append("Positive analyst revisions")
        elif features.get("analyst_sentiment", 0) < -0.3:
            key_factors.append("Negative analyst revisions")
        
        if features.get("parking_lot_activity", 1) > 1.2:
            key_factors.append("Increased retail activity (satellite data)")
        
        if features.get("call_put_ratio", 1) > 1.5:
            key_factors.append("Bullish options positioning")
        elif features.get("call_put_ratio", 1) < 0.7:
            key_factors.append("Bearish options positioning")
        
        return key_factors[:5]  # Return top 5 factors
    
    def _create_prediction_alert(self, prediction: EarningsPrediction) -> Optional[PredictionAlert]:
        """Create alert for high-confidence prediction"""
        
        if prediction.confidence_level.value not in ["high", "very_high"]:
            return None
        
        alert_reason = f"High confidence {prediction.surprise_direction.value} prediction"
        
        if prediction.surprise_direction == SurpriseDirection.BEAT:
            recommended_action = "Consider long position or call options"
            risk_factors = ["Market volatility", "Sector rotation", "Macro headwinds"]
        else:
            recommended_action = "Consider short position or put options"
            risk_factors = ["Short squeeze risk", "Positive surprise risk", "Sector strength"]
        
        return PredictionAlert(
            id=f"alert_{prediction.symbol}_{int(prediction.prediction_timestamp.timestamp())}",
            symbol=prediction.symbol,
            earnings_date=prediction.earnings_date,
            prediction=prediction,
            alert_reason=alert_reason,
            recommended_action=recommended_action,
            risk_factors=risk_factors,
            created_at=datetime.now()
        )
    
    async def _collect_training_data(self) -> List[Dict[str, Any]]:
        """Collect historical data for model training"""
        
        # In production, this would collect real historical earnings data
        # For demo, generate simulated training data
        training_data = []
        
        for i in range(500):  # Generate 500 training samples
            # Simulate historical earnings event
            features = {
                "market_cap": np.random.uniform(1e9, 1000e9),
                "estimated_eps": np.random.uniform(0.5, 5.0),
                "pe_ratio": np.random.uniform(10, 50),
                "revenue_growth": np.random.uniform(-0.3, 0.6),
                "twitter_sentiment": np.random.uniform(-1, 1),
                "analyst_sentiment": np.random.uniform(-1, 1),
                "call_put_ratio": np.random.uniform(0.3, 3.0),
                "vix_level": np.random.uniform(15, 50),
                "sector_return_1m": np.random.uniform(-0.3, 0.3)
            }
            
            # Simulate actual surprise (with some correlation to features)
            eps_surprise = (
                features["twitter_sentiment"] * 0.3 +
                features["analyst_sentiment"] * 0.4 +
                np.random.normal(0, 0.2)
            )
            
            revenue_surprise = (
                features["revenue_growth"] * 0.5 +
                features["twitter_sentiment"] * 0.2 +
                np.random.normal(0, 0.15)
            )
            
            training_data.append({
                "features": features,
                "eps_surprise": eps_surprise,
                "revenue_surprise": revenue_surprise
            })
        
        return training_data
    
    async def _train_model_ensemble(self, features_df: pd.DataFrame, targets: List[float], target_type: str) -> Dict[str, Any]:
        """Train ensemble of models for a target"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_type] = scaler
        
        results = {}
        
        # Train each model in ensemble
        for model_name, model in self.models[target_type].items():
            try:
                # Train model
                if model_name == "linear":
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[model_name] = {
                    "mse": mse,
                    "r2": r2,
                    "rmse": np.sqrt(mse)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name} for {target_type}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def _save_trained_models(self):
        """Save trained models to disk"""
        
        try:
            # Save models
            model_file = self.model_path / "models.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self.models, f)
            
            # Save scalers
            scaler_file = self.model_path / "scalers.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers, f)
            
            # Save feature columns
            features_file = self.model_path / "features.pkl"
            with open(features_file, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            # Save training timestamp
            timestamp_file = self.model_path / "last_training.txt"
            with open(timestamp_file, 'w') as f:
                f.write(datetime.now().isoformat())
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def _load_trained_models(self):
        """Load trained models from disk"""
        
        # Load models
        model_file = self.model_path / "models.pkl"
        with open(model_file, 'rb') as f:
            self.models = pickle.load(f)
        
        # Load scalers
        scaler_file = self.model_path / "scalers.pkl"
        with open(scaler_file, 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Load feature columns
        features_file = self.model_path / "features.pkl"
        with open(features_file, 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        self.logger.info("Models loaded successfully")
    
    def _models_recently_trained(self) -> bool:
        """Check if models were recently trained"""
        
        timestamp_file = self.model_path / "last_training.txt"
        if not timestamp_file.exists():
            return False
        
        try:
            with open(timestamp_file, 'r') as f:
                last_training = datetime.fromisoformat(f.read().strip())
            
            days_since_training = (datetime.now() - last_training).days
            return days_since_training < self.config.model_retrain_frequency
            
        except:
            return False
    
    def _get_last_training_date(self) -> Optional[str]:
        """Get last training date"""
        
        timestamp_file = self.model_path / "last_training.txt"
        if not timestamp_file.exists():
            return None
        
        try:
            with open(timestamp_file, 'r') as f:
                return f.read().strip()
        except:
            return None
    
    async def _get_historical_earnings(self, symbols: List[str], days_back: int) -> List[EarningsEvent]:
        """Get historical earnings events for backtesting"""
        
        # Simulated historical earnings data
        events = []
        
        for symbol in symbols:
            for i in range(4):  # 4 quarters back
                earnings_date = datetime.now() - timedelta(days=90*i + np.random.randint(0, 30))
                
                estimated_eps = np.random.uniform(1.0, 5.0)
                actual_eps = estimated_eps + np.random.uniform(-0.5, 0.5)
                
                event = EarningsEvent(
                    symbol=symbol,
                    company_name=f"{symbol} Inc.",
                    earnings_date=earnings_date,
                    fiscal_quarter=f"Q{4-i}",
                    fiscal_year=2024 if i < 2 else 2023,
                    estimated_eps=estimated_eps,
                    estimated_revenue=np.random.uniform(10e9, 100e9),
                    actual_eps=actual_eps,
                    actual_revenue=np.random.uniform(10e9, 100e9),
                    surprise_eps=actual_eps - estimated_eps,
                    market_cap=np.random.uniform(100e9, 3000e9),
                    sector=np.random.choice(["Technology", "Consumer", "Healthcare"])
                )
                
                events.append(event)
        
        return events
    
    async def _collect_historical_features(self, event: EarningsEvent) -> Dict[str, float]:
        """Collect features as they would have been before historical earnings"""
        
        # This would recreate the feature state before the earnings announcement
        # For demo, use similar logic as current features but with historical context
        return await self._collect_prediction_features(event, True)
    
    def _get_top_features(self, avg_importance: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """Get top features by importance"""
        
        top_features = {}
        
        for target_type, importances in avg_importance.items():
            # Sort features by importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            top_features[target_type] = sorted_features[:10]  # Top 10 features
        
        return top_features


if __name__ == "__main__":
    async def test_earnings_predictor():
        """Test the earnings surprise predictor"""
        tool = EarningsSurprisePredictorTool()
        
        # Test earnings prediction
        result = await tool._arun(
            operation="predict_earnings",
            symbols=["AAPL", "GOOGL", "MSFT"],
            days_ahead=30,
            include_alternative_data=True,
            confidence_threshold=0.6
        )
        
        print("Earnings Prediction Result:")
        print(result)
    
    asyncio.run(test_earnings_predictor())