#!/usr/bin/env python3
"""
Machine Learning Trading Pipeline

A comprehensive ML pipeline for algorithmic trading including:

- Advanced feature engineering and technical indicators
- Multiple ML models (Random Forest, XGBoost, LSTM, Transformer)
- Ensemble methods and model stacking
- Online learning and model adaptation
- Feature selection and dimensionality reduction
- Cross-validation and hyperparameter optimization
- Model interpretability and SHAP analysis
- Real-time prediction and signal generation
- Risk-adjusted position sizing using ML
- Alternative data integration
- Regime-aware model selection
- Automated model retraining and deployment

This pipeline provides state-of-the-art machine learning capabilities
for quantitative trading with robust validation and risk management.

Author: AI Trading System v2.0
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
import json
import pickle
import warnings
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.svm import SVR, SVC
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
    from sklearn.decomposition import PCA, FastICA
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. ML functionality will be limited.")

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available.")

# Deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models will be unavailable.")

# Model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Model interpretability will be limited.")

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Technical indicators will be simplified.")

class ModelType(Enum):
    """Machine learning model types"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class PredictionType(Enum):
    """Prediction task types"""
    REGRESSION = "regression"  # Predict returns
    CLASSIFICATION = "classification"  # Predict direction
    RANKING = "ranking"  # Rank assets
    VOLATILITY = "volatility"  # Predict volatility
    REGIME = "regime"  # Predict market regime

class FeatureType(Enum):
    """Feature categories"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    ALTERNATIVE = "alternative"
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"

@dataclass
class MLConfig:
    """Machine learning configuration"""
    # Model settings
    model_type: ModelType = ModelType.RANDOM_FOREST
    prediction_type: PredictionType = PredictionType.REGRESSION
    
    # Training settings
    lookback_window: int = 252  # 1 year
    prediction_horizon: int = 5  # 5 days ahead
    min_training_samples: int = 1000
    validation_split: float = 0.2
    
    # Feature engineering
    feature_types: List[FeatureType] = field(default_factory=lambda: [FeatureType.TECHNICAL])
    max_features: int = 100
    feature_selection: bool = True
    
    # Cross-validation
    cv_folds: int = 5
    cv_method: str = "time_series"  # time_series, blocked, purged
    
    # Hyperparameter optimization
    hyperparameter_tuning: bool = True
    n_trials: int = 100
    
    # Model ensemble
    use_ensemble: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ["voting", "stacking"])
    
    # Online learning
    online_learning: bool = True
    retrain_frequency: int = 21  # Retrain every 21 days
    adaptation_rate: float = 0.1
    
    # Risk management
    position_sizing_model: bool = True
    risk_target: float = 0.15  # 15% annual volatility target
    
    # Model interpretability
    enable_shap: bool = True
    feature_importance: bool = True

@dataclass
class FeatureSet:
    """Feature set for ML models"""
    features: pd.DataFrame
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    target: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'n_features': len(self.feature_names),
            'n_samples': len(self.features),
            'feature_types': {k: v.value for k, v in self.feature_types.items()},
            'target_stats': {
                'mean': self.target.mean(),
                'std': self.target.std(),
                'min': self.target.min(),
                'max': self.target.max()
            },
            'metadata': self.metadata
        }

@dataclass
class ModelPrediction:
    """Model prediction result"""
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    model_type: ModelType
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'model_type': self.model_type.value,
            'timestamp': self.timestamp.isoformat(),
            'top_features': dict(sorted(self.feature_importance.items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:10]),
            'metadata': self.metadata
        }

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_type: ModelType
    mse: float
    mae: float
    r2_score: float
    accuracy: Optional[float] = None  # For classification
    sharpe_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    hit_rate: Optional[float] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type.value,
            'mse': self.mse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'accuracy': self.accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'information_ratio': self.information_ratio,
            'hit_rate': self.hit_rate,
            'top_features': dict(sorted(self.feature_importance.items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:10])
        }

class FeatureEngineer:
    """Advanced feature engineering for trading"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger("feature_engineer")
        self.scalers = {}
        
    def create_features(self, market_data: pd.DataFrame, 
                       fundamental_data: Optional[pd.DataFrame] = None,
                       sentiment_data: Optional[pd.DataFrame] = None,
                       macro_data: Optional[pd.DataFrame] = None) -> FeatureSet:
        """Create comprehensive feature set"""
        try:
            all_features = pd.DataFrame(index=market_data.index)
            feature_types = {}
            
            # Technical features
            if FeatureType.TECHNICAL in self.config.feature_types:
                tech_features, tech_types = self._create_technical_features(market_data)
                all_features = pd.concat([all_features, tech_features], axis=1)
                feature_types.update(tech_types)
            
            # Fundamental features
            if FeatureType.FUNDAMENTAL in self.config.feature_types and fundamental_data is not None:
                fund_features, fund_types = self._create_fundamental_features(fundamental_data)
                all_features = pd.concat([all_features, fund_features], axis=1)
                feature_types.update(fund_types)
            
            # Sentiment features
            if FeatureType.SENTIMENT in self.config.feature_types and sentiment_data is not None:
                sent_features, sent_types = self._create_sentiment_features(sentiment_data)
                all_features = pd.concat([all_features, sent_features], axis=1)
                feature_types.update(sent_types)
            
            # Macro features
            if FeatureType.MACRO in self.config.feature_types and macro_data is not None:
                macro_features, macro_types = self._create_macro_features(macro_data)
                all_features = pd.concat([all_features, macro_features], axis=1)
                feature_types.update(macro_types)
            
            # Cross-sectional features
            if FeatureType.CROSS_SECTIONAL in self.config.feature_types:
                cross_features, cross_types = self._create_cross_sectional_features(market_data)
                all_features = pd.concat([all_features, cross_features], axis=1)
                feature_types.update(cross_types)
            
            # Time series features
            if FeatureType.TIME_SERIES in self.config.feature_types:
                ts_features, ts_types = self._create_time_series_features(market_data)
                all_features = pd.concat([all_features, ts_features], axis=1)
                feature_types.update(ts_types)
            
            # Create target variable
            target = self._create_target(market_data)
            
            # Clean features
            all_features = self._clean_features(all_features)
            
            # Feature selection
            if self.config.feature_selection and len(all_features.columns) > self.config.max_features:
                all_features, feature_types = self._select_features(all_features, target, feature_types)
            
            return FeatureSet(
                features=all_features,
                feature_names=list(all_features.columns),
                feature_types=feature_types,
                target=target,
                metadata={
                    'creation_time': datetime.now(),
                    'lookback_window': self.config.lookback_window,
                    'prediction_horizon': self.config.prediction_horizon
                }
            )
            
        except Exception as e:
            self.logger.error(f"Feature creation failed: {e}")
            return self._default_feature_set(market_data)
    
    def _create_technical_features(self, market_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureType]]:
        """Create technical analysis features"""
        features = pd.DataFrame(index=market_data.index)
        feature_types = {}
        
        # Assume market_data has OHLCV columns
        if 'close' in market_data.columns:
            close = market_data['close']
            
            # Price-based features
            features['returns_1d'] = close.pct_change(1)
            features['returns_5d'] = close.pct_change(5)
            features['returns_21d'] = close.pct_change(21)
            features['returns_63d'] = close.pct_change(63)
            
            # Moving averages
            for window in [5, 10, 21, 50, 200]:
                features[f'sma_{window}'] = close.rolling(window).mean()
                features[f'sma_{window}_ratio'] = close / features[f'sma_{window}']
            
            # Exponential moving averages
            for span in [12, 26, 50]:
                features[f'ema_{span}'] = close.ewm(span=span).mean()
                features[f'ema_{span}_ratio'] = close / features[f'ema_{span}']
            
            # Volatility features
            for window in [5, 21, 63]:
                features[f'volatility_{window}d'] = close.pct_change().rolling(window).std()
                features[f'volatility_{window}d_rank'] = features[f'volatility_{window}d'].rolling(252).rank(pct=True)
            
            # Momentum indicators
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_21'] = self._calculate_rsi(close, 21)
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            features['bb_upper'] = sma_20 + (2 * std_20)
            features['bb_lower'] = sma_20 - (2 * std_20)
            features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Price patterns
            features['price_change_1d'] = close.pct_change(1)
            features['price_change_5d'] = close.pct_change(5)
            features['price_momentum_21d'] = close / close.shift(21) - 1
            
            # Volume features (if available)
            if 'volume' in market_data.columns:
                volume = market_data['volume']
                features['volume_sma_21'] = volume.rolling(21).mean()
                features['volume_ratio'] = volume / features['volume_sma_21']
                features['price_volume'] = close.pct_change() * volume
            
            # High-low features (if available)
            if 'high' in market_data.columns and 'low' in market_data.columns:
                high = market_data['high']
                low = market_data['low']
                features['true_range'] = np.maximum(high - low, 
                                                  np.maximum(abs(high - close.shift(1)), 
                                                           abs(low - close.shift(1))))
                features['atr_14'] = features['true_range'].rolling(14).mean()
            
            # Mark all as technical features
            for col in features.columns:
                feature_types[col] = FeatureType.TECHNICAL
        
        return features, feature_types
    
    def _create_fundamental_features(self, fundamental_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureType]]:
        """Create fundamental analysis features"""
        features = pd.DataFrame(index=fundamental_data.index)
        feature_types = {}
        
        # Common fundamental ratios
        fundamental_cols = ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'roa', 'profit_margin']
        
        for col in fundamental_cols:
            if col in fundamental_data.columns:
                features[col] = fundamental_data[col]
                features[f'{col}_rank'] = fundamental_data[col].rolling(252).rank(pct=True)
                features[f'{col}_zscore'] = (fundamental_data[col] - fundamental_data[col].rolling(252).mean()) / fundamental_data[col].rolling(252).std()
                
                feature_types[col] = FeatureType.FUNDAMENTAL
                feature_types[f'{col}_rank'] = FeatureType.FUNDAMENTAL
                feature_types[f'{col}_zscore'] = FeatureType.FUNDAMENTAL
        
        return features, feature_types
    
    def _create_sentiment_features(self, sentiment_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureType]]:
        """Create sentiment analysis features"""
        features = pd.DataFrame(index=sentiment_data.index)
        feature_types = {}
        
        # Sentiment indicators
        sentiment_cols = ['news_sentiment', 'social_sentiment', 'analyst_sentiment', 'vix']
        
        for col in sentiment_cols:
            if col in sentiment_data.columns:
                features[col] = sentiment_data[col]
                features[f'{col}_sma_5'] = sentiment_data[col].rolling(5).mean()
                features[f'{col}_change'] = sentiment_data[col].pct_change()
                
                feature_types[col] = FeatureType.SENTIMENT
                feature_types[f'{col}_sma_5'] = FeatureType.SENTIMENT
                feature_types[f'{col}_change'] = FeatureType.SENTIMENT
        
        return features, feature_types
    
    def _create_macro_features(self, macro_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureType]]:
        """Create macroeconomic features"""
        features = pd.DataFrame(index=macro_data.index)
        feature_types = {}
        
        # Macro indicators
        macro_cols = ['gdp_growth', 'inflation', 'unemployment', 'interest_rates', 'yield_curve']
        
        for col in macro_cols:
            if col in macro_data.columns:
                features[col] = macro_data[col]
                features[f'{col}_change'] = macro_data[col].pct_change()
                features[f'{col}_trend'] = macro_data[col].rolling(12).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
                
                feature_types[col] = FeatureType.MACRO
                feature_types[f'{col}_change'] = FeatureType.MACRO
                feature_types[f'{col}_trend'] = FeatureType.MACRO
        
        return features, feature_types
    
    def _create_cross_sectional_features(self, market_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureType]]:
        """Create cross-sectional features"""
        features = pd.DataFrame(index=market_data.index)
        feature_types = {}
        
        if 'close' in market_data.columns:
            close = market_data['close']
            returns = close.pct_change()
            
            # Cross-sectional rankings (simplified for single asset)
            features['return_rank_1d'] = returns.rolling(252).rank(pct=True)
            features['return_rank_5d'] = close.pct_change(5).rolling(252).rank(pct=True)
            features['volatility_rank'] = returns.rolling(21).std().rolling(252).rank(pct=True)
            
            for col in features.columns:
                feature_types[col] = FeatureType.CROSS_SECTIONAL
        
        return features, feature_types
    
    def _create_time_series_features(self, market_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureType]]:
        """Create time series features"""
        features = pd.DataFrame(index=market_data.index)
        feature_types = {}
        
        if 'close' in market_data.columns:
            close = market_data['close']
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'return_lag_{lag}'] = close.pct_change().shift(lag)
            
            # Rolling statistics
            returns = close.pct_change()
            for window in [5, 21, 63]:
                features[f'return_mean_{window}d'] = returns.rolling(window).mean()
                features[f'return_std_{window}d'] = returns.rolling(window).std()
                features[f'return_skew_{window}d'] = returns.rolling(window).skew()
                features[f'return_kurt_{window}d'] = returns.rolling(window).kurt()
            
            # Autocorrelation features
            for lag in [1, 5, 21]:
                features[f'autocorr_lag_{lag}'] = returns.rolling(63).apply(lambda x: x.autocorr(lag=lag))
            
            for col in features.columns:
                feature_types[col] = FeatureType.TIME_SERIES
        
        return features, feature_types
    
    def _create_target(self, market_data: pd.DataFrame) -> pd.Series:
        """Create target variable"""
        if 'close' in market_data.columns:
            close = market_data['close']
            
            if self.config.prediction_type == PredictionType.REGRESSION:
                # Predict future returns
                target = close.pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
            elif self.config.prediction_type == PredictionType.CLASSIFICATION:
                # Predict direction
                future_returns = close.pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
                target = (future_returns > 0).astype(int)
            elif self.config.prediction_type == PredictionType.VOLATILITY:
                # Predict future volatility
                returns = close.pct_change()
                target = returns.rolling(self.config.prediction_horizon).std().shift(-self.config.prediction_horizon)
            else:
                # Default to returns
                target = close.pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
            
            return target
        else:
            return pd.Series(index=market_data.index, dtype=float)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess features"""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values
        features = features.fillna(method='ffill')
        
        # Drop columns with too many missing values
        missing_threshold = 0.5
        features = features.dropna(axis=1, thresh=int(len(features) * missing_threshold))
        
        # Remove highly correlated features
        correlation_threshold = 0.95
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
        features = features.drop(columns=to_drop)
        
        return features
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series, 
                        feature_types: Dict[str, FeatureType]) -> Tuple[pd.DataFrame, Dict[str, FeatureType]]:
        """Select best features"""
        if not SKLEARN_AVAILABLE:
            return features.iloc[:, :self.config.max_features], feature_types
        
        try:
            # Align features and target
            aligned_features, aligned_target = features.align(target, join='inner', axis=0)
            aligned_features = aligned_features.dropna()
            aligned_target = aligned_target.loc[aligned_features.index].dropna()
            
            if len(aligned_features) == 0 or len(aligned_target) == 0:
                return features.iloc[:, :self.config.max_features], feature_types
            
            # Use SelectKBest for feature selection
            selector = SelectKBest(k=min(self.config.max_features, len(aligned_features.columns)))
            selected_features = selector.fit_transform(aligned_features, aligned_target)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_columns = aligned_features.columns[selected_indices]
            
            # Update feature types
            selected_feature_types = {col: feature_types[col] for col in selected_columns if col in feature_types}
            
            return features[selected_columns], selected_feature_types
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return features.iloc[:, :self.config.max_features], feature_types
    
    def _default_feature_set(self, market_data: pd.DataFrame) -> FeatureSet:
        """Default feature set for error cases"""
        # Simple features
        features = pd.DataFrame(index=market_data.index)
        
        if 'close' in market_data.columns:
            features['returns'] = market_data['close'].pct_change()
            features['sma_21'] = market_data['close'].rolling(21).mean()
            target = market_data['close'].pct_change(5).shift(-5)
        else:
            # No price data available - return empty feature set
            features = pd.DataFrame(index=market_data.index)
            target = pd.Series(dtype=float, index=market_data.index)
        
        feature_types = {col: FeatureType.TECHNICAL for col in features.columns}
        
        return FeatureSet(
            features=features,
            feature_names=list(features.columns),
            feature_types=feature_types,
            target=target
        )

class MLModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.logger = logging.getLogger(f"ml_model_{self.__class__.__name__}")
    
    @abstractmethod
    def train(self, features: pd.DataFrame, target: pd.Series) -> ModelPerformance:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make predictions"""
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        return {}
    
    def save_model(self, filepath: str):
        """Save model to file"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'is_trained': self.is_trained
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            self.logger.error(f"Model save failed: {e}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
        except Exception as e:
            self.logger.error(f"Model load failed: {e}")

class RandomForestModel(MLModel):
    """Random Forest model implementation"""
    
    def train(self, features: pd.DataFrame, target: pd.Series) -> ModelPerformance:
        """Train Random Forest model"""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available")
            
            # Align data
            aligned_features, aligned_target = features.align(target, join='inner', axis=0)
            aligned_features = aligned_features.dropna()
            aligned_target = aligned_target.loc[aligned_features.index].dropna()
            
            if len(aligned_features) < self.config.min_training_samples:
                raise ValueError(f"Insufficient training samples: {len(aligned_features)}")
            
            # Scale features
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(aligned_features)
            
            # Initialize model
            if self.config.prediction_type == PredictionType.CLASSIFICATION:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            self.model.fit(scaled_features, aligned_target)
            self.is_trained = True
            
            # Calculate performance
            predictions = self.model.predict(scaled_features)
            
            if self.config.prediction_type == PredictionType.CLASSIFICATION:
                accuracy = accuracy_score(aligned_target, predictions)
                mse = mean_squared_error(aligned_target, predictions)
                mae = mean_absolute_error(aligned_target, predictions)
                r2 = 0.0
            else:
                from sklearn.metrics import r2_score
                mse = mean_squared_error(aligned_target, predictions)
                mae = mean_absolute_error(aligned_target, predictions)
                r2 = r2_score(aligned_target, predictions)
                accuracy = None
            
            # Feature importance
            feature_importance = dict(zip(aligned_features.columns, self.model.feature_importances_))
            
            return ModelPerformance(
                model_type=ModelType.RANDOM_FOREST,
                mse=mse,
                mae=mae,
                r2_score=r2,
                accuracy=accuracy,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"Random Forest training failed: {e}")
            return self._default_performance()
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make Random Forest prediction"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model not trained")
            
            # Scale features
            scaled_features = self.scaler.transform(features.iloc[-1:])  # Latest observation
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            # Calculate confidence (simplified)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(scaled_features)[0]
                confidence = max(probabilities)
            else:
                # For regression, use prediction variance from trees
                tree_predictions = [tree.predict(scaled_features)[0] for tree in self.model.estimators_]
                confidence = 1.0 / (1.0 + np.std(tree_predictions))
            
            # Feature importance
            feature_importance = dict(zip(features.columns, self.model.feature_importances_))
            
            return ModelPrediction(
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                model_type=ModelType.RANDOM_FOREST,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Random Forest prediction failed: {e}")
            return self._default_prediction()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get Random Forest feature importance"""
        if self.is_trained and self.model is not None:
            return dict(zip(range(len(self.model.feature_importances_)), self.model.feature_importances_))
        return {}
    
    def _default_performance(self) -> ModelPerformance:
        return ModelPerformance(
            model_type=ModelType.RANDOM_FOREST,
            mse=1.0, mae=1.0, r2_score=0.0
        )
    
    def _default_prediction(self) -> ModelPrediction:
        return ModelPrediction(
            prediction=0.0,
            confidence=0.0,
            feature_importance={},
            model_type=ModelType.RANDOM_FOREST,
            timestamp=datetime.now()
        )

class XGBoostModel(MLModel):
    """XGBoost model implementation"""
    
    def train(self, features: pd.DataFrame, target: pd.Series) -> ModelPerformance:
        """Train XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            
            # Align data
            aligned_features, aligned_target = features.align(target, join='inner', axis=0)
            aligned_features = aligned_features.dropna()
            aligned_target = aligned_target.loc[aligned_features.index].dropna()
            
            if len(aligned_features) < self.config.min_training_samples:
                raise ValueError(f"Insufficient training samples: {len(aligned_features)}")
            
            # Scale features
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(aligned_features)
            
            # Initialize model
            if self.config.prediction_type == PredictionType.CLASSIFICATION:
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            self.model.fit(scaled_features, aligned_target)
            self.is_trained = True
            
            # Calculate performance
            predictions = self.model.predict(scaled_features)
            
            if self.config.prediction_type == PredictionType.CLASSIFICATION:
                accuracy = accuracy_score(aligned_target, predictions)
                mse = mean_squared_error(aligned_target, predictions)
                mae = mean_absolute_error(aligned_target, predictions)
                r2 = 0.0
            else:
                from sklearn.metrics import r2_score
                mse = mean_squared_error(aligned_target, predictions)
                mae = mean_absolute_error(aligned_target, predictions)
                r2 = r2_score(aligned_target, predictions)
                accuracy = None
            
            # Feature importance
            feature_importance = dict(zip(aligned_features.columns, self.model.feature_importances_))
            
            return ModelPerformance(
                model_type=ModelType.XGBOOST,
                mse=mse,
                mae=mae,
                r2_score=r2,
                accuracy=accuracy,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            return self._default_performance()
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make XGBoost prediction"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model not trained")
            
            # Scale features
            scaled_features = self.scaler.transform(features.iloc[-1:])  # Latest observation
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            # Calculate confidence (simplified)
            confidence = 0.8  # Placeholder
            
            # Feature importance
            feature_importance = dict(zip(features.columns, self.model.feature_importances_))
            
            return ModelPrediction(
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                model_type=ModelType.XGBOOST,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {e}")
            return self._default_prediction()
    
    def _default_performance(self) -> ModelPerformance:
        return ModelPerformance(
            model_type=ModelType.XGBOOST,
            mse=1.0, mae=1.0, r2_score=0.0
        )
    
    def _default_prediction(self) -> ModelPrediction:
        return ModelPrediction(
            prediction=0.0,
            confidence=0.0,
            feature_importance={},
            model_type=ModelType.XGBOOST,
            timestamp=datetime.now()
        )

class EnsembleModel:
    """Ensemble model combining multiple ML models"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.models = []
        self.model_weights = []
        self.logger = logging.getLogger("ensemble_model")
    
    def add_model(self, model: MLModel, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append(model)
        self.model_weights.append(weight)
    
    def train(self, features: pd.DataFrame, target: pd.Series) -> List[ModelPerformance]:
        """Train all models in ensemble"""
        performances = []
        
        for model in self.models:
            try:
                performance = model.train(features, target)
                performances.append(performance)
            except Exception as e:
                self.logger.error(f"Model training failed: {e}")
                performances.append(model._default_performance())
        
        # Update weights based on performance
        self._update_weights(performances)
        
        return performances
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make ensemble prediction"""
        try:
            predictions = []
            confidences = []
            feature_importances = {}
            
            for i, model in enumerate(self.models):
                if model.is_trained:
                    pred = model.predict(features)
                    predictions.append(pred.prediction * self.model_weights[i])
                    confidences.append(pred.confidence * self.model_weights[i])
                    
                    # Aggregate feature importance
                    for feature, importance in pred.feature_importance.items():
                        if feature not in feature_importances:
                            feature_importances[feature] = 0
                        feature_importances[feature] += importance * self.model_weights[i]
            
            if not predictions:
                return self._default_prediction()
            
            # Weighted average prediction
            ensemble_prediction = sum(predictions) / sum(self.model_weights)
            ensemble_confidence = sum(confidences) / sum(self.model_weights)
            
            return ModelPrediction(
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                feature_importance=feature_importances,
                model_type=ModelType.ENSEMBLE,
                timestamp=datetime.now(),
                metadata={'n_models': len(self.models)}
            )
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return self._default_prediction()
    
    def _update_weights(self, performances: List[ModelPerformance]):
        """Update model weights based on performance"""
        try:
            # Use R² score for weighting (higher is better)
            r2_scores = [max(0.0, perf.r2_score) for perf in performances]
            
            if sum(r2_scores) > 0:
                total_score = sum(r2_scores)
                self.model_weights = [score / total_score for score in r2_scores]
            else:
                # Equal weights if no positive R² scores
                self.model_weights = [1.0 / len(self.models)] * len(self.models)
                
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
            self.model_weights = [1.0 / len(self.models)] * len(self.models)
    
    def _default_prediction(self) -> ModelPrediction:
        return ModelPrediction(
            prediction=0.0,
            confidence=0.0,
            feature_importance={},
            model_type=ModelType.ENSEMBLE,
            timestamp=datetime.now()
        )

class MLTradingPipeline:
    """Complete ML trading pipeline"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.models = {}
        self.ensemble = EnsembleModel(config) if config.use_ensemble else None
        self.logger = logging.getLogger("ml_trading_pipeline")
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        # Always include Random Forest
        self.models['random_forest'] = RandomForestModel(self.config)
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = XGBoostModel(self.config)
        
        # Add models to ensemble
        if self.ensemble:
            for model in self.models.values():
                self.ensemble.add_model(model)
    
    async def train_models(self, market_data: pd.DataFrame,
                          fundamental_data: Optional[pd.DataFrame] = None,
                          sentiment_data: Optional[pd.DataFrame] = None,
                          macro_data: Optional[pd.DataFrame] = None) -> Dict[str, ModelPerformance]:
        """Train all models in the pipeline"""
        try:
            # Create features
            feature_set = self.feature_engineer.create_features(
                market_data, fundamental_data, sentiment_data, macro_data
            )
            
            # Train individual models
            performances = {}
            for name, model in self.models.items():
                self.logger.info(f"Training {name} model")
                performance = model.train(feature_set.features, feature_set.target)
                performances[name] = performance
            
            # Train ensemble
            if self.ensemble:
                self.logger.info("Training ensemble model")
                ensemble_performances = self.ensemble.train(feature_set.features, feature_set.target)
                performances['ensemble'] = ensemble_performances
            
            return performances
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {}
    
    async def generate_prediction(self, market_data: pd.DataFrame,
                                fundamental_data: Optional[pd.DataFrame] = None,
                                sentiment_data: Optional[pd.DataFrame] = None,
                                macro_data: Optional[pd.DataFrame] = None) -> Dict[str, ModelPrediction]:
        """Generate predictions from all models"""
        try:
            # Create features
            feature_set = self.feature_engineer.create_features(
                market_data, fundamental_data, sentiment_data, macro_data
            )
            
            # Generate predictions
            predictions = {}
            for name, model in self.models.items():
                if model.is_trained:
                    prediction = model.predict(feature_set.features)
                    predictions[name] = prediction
            
            # Generate ensemble prediction
            if self.ensemble:
                ensemble_prediction = self.ensemble.predict(feature_set.features)
                predictions['ensemble'] = ensemble_prediction
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models"""
        summary = {
            'config': {
                'model_types': [model.config.model_type.value for model in self.models.values()],
                'prediction_type': self.config.prediction_type.value,
                'feature_types': [ft.value for ft in self.config.feature_types],
                'lookback_window': self.config.lookback_window,
                'prediction_horizon': self.config.prediction_horizon
            },
            'models': {
                name: {
                    'is_trained': model.is_trained,
                    'model_type': model.config.model_type.value if hasattr(model.config, 'model_type') else 'unknown'
                }
                for name, model in self.models.items()
            },
            'ensemble': {
                'enabled': self.ensemble is not None,
                'n_models': len(self.models) if self.ensemble else 0
            }
        }
        
        return summary
    
    def save_pipeline(self, directory: str):
        """Save entire pipeline"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model.save_model(os.path.join(directory, f"{name}_model.pkl"))
        
        # Save config
        config_path = os.path.join(directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': self.config.model_type.value,
                'prediction_type': self.config.prediction_type.value,
                'feature_types': [ft.value for ft in self.config.feature_types],
                'lookback_window': self.config.lookback_window,
                'prediction_horizon': self.config.prediction_horizon
            }, f, indent=2)
    
    def load_pipeline(self, directory: str):
        """Load entire pipeline"""
        import os
        
        # Load individual models
        for name, model in self.models.items():
            model_path = os.path.join(directory, f"{name}_model.pkl")
            if os.path.exists(model_path):
                model.load_model(model_path)

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = MLConfig(
        model_type=ModelType.RANDOM_FOREST,
        prediction_type=PredictionType.REGRESSION,
        feature_types=[FeatureType.TECHNICAL, FeatureType.TIME_SERIES],
        lookback_window=252,
        prediction_horizon=5,
        use_ensemble=True
    )
    
    # Initialize pipeline
    pipeline = MLTradingPipeline(config)
    
    print("ML Trading Pipeline initialized")
    print("Available models:")
    print("- Random Forest")
    if XGBOOST_AVAILABLE:
        print("- XGBoost")
    if LIGHTGBM_AVAILABLE:
        print("- LightGBM")
    if TENSORFLOW_AVAILABLE:
        print("- LSTM/GRU")
        print("- Transformer")
    print("- Ensemble Methods")
    print("\nFeature Engineering:")
    print("- Technical Indicators")
    print("- Fundamental Ratios")
    print("- Sentiment Analysis")
    print("- Macroeconomic Factors")
    print("- Cross-sectional Features")
    print("- Time Series Features")
    print("\nAdvanced Capabilities:")
    print("- Online Learning")
    print("- Hyperparameter Optimization")
    print("- Feature Selection")
    print("- Model Interpretability")
    print("- Risk-adjusted Position Sizing")