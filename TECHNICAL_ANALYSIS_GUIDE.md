# Advanced Technical Analysis Guide

## 🎯 Comprehensive Technical Analysis Framework

This document provides detailed technical specifications for the advanced technical analysis capabilities implemented in the trading system, including custom indicators, signal generation methodologies, and quantitative strategies.

---

## 📋 Table of Contents

1. [Technical Indicators Overview](#technical-indicators-overview)
2. [Custom Indicators](#custom-indicators)
3. [Signal Generation Framework](#signal-generation-framework)
4. [Multi-Timeframe Analysis](#multi-timeframe-analysis)
5. [Pattern Recognition](#pattern-recognition)
6. [Volatility Analysis](#volatility-analysis)
7. [Market Microstructure](#market-microstructure)
8. [Regime Detection](#regime-detection)
9. [Risk-Adjusted Signals](#risk-adjusted-signals)
10. [Backtesting Methodology](#backtesting-methodology)
11. [Performance Metrics](#performance-metrics)
12. [Implementation Examples](#implementation-examples)

---

## 📊 Technical Indicators Overview

### Core Indicator Categories

```python
class IndicatorCategory(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    OSCILLATORS = "oscillators"
    CUSTOM = "custom"
```

### Implemented Indicators

#### 1. Trend Indicators

**Moving Averages**
```python
def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various moving averages
    """
    # Simple Moving Averages
    data['SMA_10'] = talib.SMA(data['close'], timeperiod=10)
    data['SMA_20'] = talib.SMA(data['close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['close'], timeperiod=200)
    
    # Exponential Moving Averages
    data['EMA_12'] = talib.EMA(data['close'], timeperiod=12)
    data['EMA_26'] = talib.EMA(data['close'], timeperiod=26)
    data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
    
    # Weighted Moving Average
    data['WMA_20'] = talib.WMA(data['close'], timeperiod=20)
    
    # Triple Exponential Moving Average
    data['TEMA_20'] = talib.TEMA(data['close'], timeperiod=20)
    
    return data
```

**MACD (Moving Average Convergence Divergence)**
```python
def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD with signal line and histogram
    """
    # Standard MACD (12, 26, 9)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
        data['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # MACD Zero Line Crossovers
    data['MACD_zero_cross'] = np.where(
        (data['MACD'] > 0) & (data['MACD'].shift(1) <= 0), 1,
        np.where((data['MACD'] < 0) & (data['MACD'].shift(1) >= 0), -1, 0)
    )
    
    # MACD Signal Line Crossovers
    data['MACD_signal_cross'] = np.where(
        (data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1)), 1,
        np.where((data['MACD'] < data['MACD_signal']) & (data['MACD'].shift(1) >= data['MACD_signal'].shift(1)), -1, 0)
    )
    
    return data
```

**Parabolic SAR**
```python
def calculate_parabolic_sar(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Parabolic Stop and Reverse
    """
    data['SAR'] = talib.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2)
    
    # SAR Trend Direction
    data['SAR_trend'] = np.where(data['close'] > data['SAR'], 1, -1)
    
    # SAR Trend Changes
    data['SAR_change'] = np.where(
        data['SAR_trend'] != data['SAR_trend'].shift(1), 1, 0
    )
    
    return data
```

#### 2. Momentum Indicators

**RSI (Relative Strength Index)**
```python
def calculate_rsi(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RSI with multiple timeframes
    """
    # Standard RSI (14 periods)
    data['RSI_14'] = talib.RSI(data['close'], timeperiod=14)
    
    # Fast RSI (7 periods)
    data['RSI_7'] = talib.RSI(data['close'], timeperiod=7)
    
    # Slow RSI (21 periods)
    data['RSI_21'] = talib.RSI(data['close'], timeperiod=21)
    
    # RSI Divergence Detection
    data['RSI_divergence'] = detect_rsi_divergence(data)
    
    # RSI Overbought/Oversold Levels
    data['RSI_overbought'] = data['RSI_14'] > 70
    data['RSI_oversold'] = data['RSI_14'] < 30
    
    return data

def detect_rsi_divergence(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Detect bullish and bearish RSI divergences
    """
    divergence = pd.Series(0, index=data.index)
    
    for i in range(window, len(data)):
        # Get recent highs and lows
        recent_data = data.iloc[i-window:i+1]
        
        # Price highs and RSI highs
        price_high_idx = recent_data['high'].idxmax()
        rsi_high_idx = recent_data['RSI_14'].idxmax()
        
        # Price lows and RSI lows
        price_low_idx = recent_data['low'].idxmin()
        rsi_low_idx = recent_data['RSI_14'].idxmin()
        
        # Bearish divergence: Price makes higher high, RSI makes lower high
        if (price_high_idx == recent_data.index[-1] and 
            rsi_high_idx != recent_data.index[-1] and
            recent_data.loc[price_high_idx, 'high'] > recent_data.loc[rsi_high_idx, 'high'] and
            recent_data.loc[price_high_idx, 'RSI_14'] < recent_data.loc[rsi_high_idx, 'RSI_14']):
            divergence.iloc[i] = -1
        
        # Bullish divergence: Price makes lower low, RSI makes higher low
        elif (price_low_idx == recent_data.index[-1] and 
              rsi_low_idx != recent_data.index[-1] and
              recent_data.loc[price_low_idx, 'low'] < recent_data.loc[rsi_low_idx, 'low'] and
              recent_data.loc[price_low_idx, 'RSI_14'] > recent_data.loc[rsi_low_idx, 'RSI_14']):
            divergence.iloc[i] = 1
    
    return divergence
```

**Stochastic Oscillator**
```python
def calculate_stochastic(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator
    """
    # Fast Stochastic
    data['STOCH_K'], data['STOCH_D'] = talib.STOCHF(
        data['high'], data['low'], data['close'],
        fastk_period=14, fastd_period=3, fastd_matype=0
    )
    
    # Slow Stochastic
    data['STOCH_SLOW_K'], data['STOCH_SLOW_D'] = talib.STOCH(
        data['high'], data['low'], data['close'],
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    
    # Stochastic Crossovers
    data['STOCH_cross'] = np.where(
        (data['STOCH_K'] > data['STOCH_D']) & (data['STOCH_K'].shift(1) <= data['STOCH_D'].shift(1)), 1,
        np.where((data['STOCH_K'] < data['STOCH_D']) & (data['STOCH_K'].shift(1) >= data['STOCH_D'].shift(1)), -1, 0)
    )
    
    return data
```

#### 3. Volatility Indicators

**Bollinger Bands**
```python
def calculate_bollinger_bands(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Bollinger Bands with multiple standard deviations
    """
    # Standard Bollinger Bands (20, 2)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(
        data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    
    # Bollinger Band Width
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
    
    # Bollinger Band Position
    data['BB_position'] = (data['close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
    
    # Bollinger Band Squeeze
    data['BB_squeeze'] = data['BB_width'] < data['BB_width'].rolling(20).quantile(0.1)
    
    # Bollinger Band Breakouts
    data['BB_breakout'] = np.where(
        data['close'] > data['BB_upper'], 1,
        np.where(data['close'] < data['BB_lower'], -1, 0)
    )
    
    return data
```

**Average True Range (ATR)**
```python
def calculate_atr(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Average True Range and related metrics
    """
    # Standard ATR (14 periods)
    data['ATR_14'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    
    # ATR Percentage
    data['ATR_pct'] = data['ATR_14'] / data['close'] * 100
    
    # ATR-based Stop Loss Levels
    data['ATR_stop_long'] = data['close'] - (2 * data['ATR_14'])
    data['ATR_stop_short'] = data['close'] + (2 * data['ATR_14'])
    
    # ATR Volatility Regime
    data['ATR_regime'] = np.where(
        data['ATR_pct'] > data['ATR_pct'].rolling(50).quantile(0.8), 'high',
        np.where(data['ATR_pct'] < data['ATR_pct'].rolling(50).quantile(0.2), 'low', 'normal')
    )
    
    return data
```

#### 4. Volume Indicators

**Volume-Weighted Average Price (VWAP)**
```python
def calculate_vwap(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VWAP and related volume indicators
    """
    # Typical Price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # VWAP (reset daily)
    data['date'] = data.index.date
    data['VWAP'] = data.groupby('date').apply(
        lambda x: (x['typical_price'] * x['volume']).cumsum() / x['volume'].cumsum()
    ).reset_index(level=0, drop=True)
    
    # VWAP Bands
    data['VWAP_std'] = data.groupby('date')['typical_price'].transform(
        lambda x: x.expanding().std()
    )
    data['VWAP_upper'] = data['VWAP'] + data['VWAP_std']
    data['VWAP_lower'] = data['VWAP'] - data['VWAP_std']
    
    # Price relative to VWAP
    data['price_vs_VWAP'] = (data['close'] - data['VWAP']) / data['VWAP']
    
    return data
```

**On-Balance Volume (OBV)**
```python
def calculate_obv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume and related indicators
    """
    # Standard OBV
    data['OBV'] = talib.OBV(data['close'], data['volume'])
    
    # OBV Moving Average
    data['OBV_MA'] = data['OBV'].rolling(20).mean()
    
    # OBV Divergence
    data['OBV_divergence'] = detect_obv_divergence(data)
    
    # Volume Trend
    data['volume_trend'] = np.where(
        data['OBV'] > data['OBV_MA'], 1,
        np.where(data['OBV'] < data['OBV_MA'], -1, 0)
    )
    
    return data
```

---

## 🔧 Custom Indicators

### 1. Trend Strength Indicator

```python
class TrendStrengthIndicator:
    """
    Custom indicator to measure trend strength across multiple timeframes
    """
    
    def __init__(self, short_period: int = 10, long_period: int = 50):
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate trend strength score (0-100)
        """
        # Price momentum
        price_momentum = self._calculate_price_momentum(data)
        
        # Moving average alignment
        ma_alignment = self._calculate_ma_alignment(data)
        
        # Volume confirmation
        volume_confirmation = self._calculate_volume_confirmation(data)
        
        # ADX strength
        adx_strength = self._calculate_adx_strength(data)
        
        # Combine components
        trend_strength = (
            price_momentum * 0.3 +
            ma_alignment * 0.3 +
            volume_confirmation * 0.2 +
            adx_strength * 0.2
        )
        
        return trend_strength * 100
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price momentum component"""
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.short_period).mean() / returns.rolling(self.long_period).std()
        return np.tanh(momentum)  # Normalize to [-1, 1]
    
    def _calculate_ma_alignment(self, data: pd.DataFrame) -> pd.Series:
        """Calculate moving average alignment"""
        ma_5 = data['close'].rolling(5).mean()
        ma_10 = data['close'].rolling(10).mean()
        ma_20 = data['close'].rolling(20).mean()
        ma_50 = data['close'].rolling(50).mean()
        
        # Check if MAs are in proper order
        bullish_alignment = (ma_5 > ma_10) & (ma_10 > ma_20) & (ma_20 > ma_50)
        bearish_alignment = (ma_5 < ma_10) & (ma_10 < ma_20) & (ma_20 < ma_50)
        
        alignment = np.where(bullish_alignment, 1, np.where(bearish_alignment, -1, 0))
        return pd.Series(alignment, index=data.index)
    
    def _calculate_volume_confirmation(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume confirmation"""
        volume_ma = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / volume_ma
        
        price_change = data['close'].pct_change()
        volume_confirmation = np.where(
            (price_change > 0) & (volume_ratio > 1), 1,
            np.where((price_change < 0) & (volume_ratio > 1), -1, 0)
        )
        
        return pd.Series(volume_confirmation, index=data.index).rolling(5).mean()
    
    def _calculate_adx_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ADX-based trend strength"""
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        adx_normalized = adx / 100  # Normalize to [0, 1]
        
        # Determine trend direction
        plus_di = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        
        trend_direction = np.where(plus_di > minus_di, 1, -1)
        
        return adx_normalized * trend_direction
```

### 2. Market Regime Detector

```python
class MarketRegimeDetector:
    """
    Detect market regimes using multiple indicators
    """
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
    
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regime: trending, ranging, volatile
        """
        # Calculate regime indicators
        trend_strength = self._calculate_trend_persistence(data)
        volatility_regime = self._calculate_volatility_regime(data)
        correlation_regime = self._calculate_correlation_regime(data)
        
        # Combine indicators to determine regime
        regime = pd.Series('ranging', index=data.index)
        
        # Trending regime
        trending_condition = (
            (trend_strength > 0.6) & 
            (volatility_regime == 'normal')
        )
        regime[trending_condition] = 'trending'
        
        # Volatile regime
        volatile_condition = (
            (volatility_regime == 'high') |
            (correlation_regime == 'crisis')
        )
        regime[volatile_condition] = 'volatile'
        
        return regime
    
    def _calculate_trend_persistence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend persistence score"""
        returns = data['close'].pct_change()
        
        # Calculate Hurst exponent as proxy for trend persistence
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        hurst_values = returns.rolling(self.lookback_period).apply(
            lambda x: hurst_exponent(x.values) if len(x.dropna()) > 50 else np.nan
        )
        
        return hurst_values
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Classify volatility regime"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        
        vol_percentiles = volatility.rolling(self.lookback_period).rank(pct=True)
        
        regime = pd.Series('normal', index=data.index)
        regime[vol_percentiles > 0.8] = 'high'
        regime[vol_percentiles < 0.2] = 'low'
        
        return regime
    
    def _calculate_correlation_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect crisis periods based on correlation breakdown"""
        # This would typically use multiple assets
        # For single asset, use volatility clustering as proxy
        returns = data['close'].pct_change()
        
        # GARCH-like volatility clustering detection
        squared_returns = returns ** 2
        volatility_clustering = squared_returns.rolling(20).corr(squared_returns.shift(1))
        
        regime = pd.Series('normal', index=data.index)
        regime[volatility_clustering > 0.3] = 'crisis'
        
        return regime
```

### 3. Adaptive Moving Average

```python
class AdaptiveMovingAverage:
    """
    Adaptive Moving Average that adjusts to market conditions
    """
    
    def __init__(self, min_period: int = 5, max_period: int = 50):
        self.min_period = min_period
        self.max_period = max_period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate adaptive moving average
        """
        # Calculate market efficiency ratio
        efficiency_ratio = self._calculate_efficiency_ratio(data)
        
        # Adaptive period based on efficiency
        adaptive_period = self.min_period + (self.max_period - self.min_period) * (1 - efficiency_ratio)
        
        # Calculate adaptive MA
        adaptive_ma = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            if i < self.max_period:
                adaptive_ma.iloc[i] = data['close'].iloc[:i+1].mean()
            else:
                period = int(adaptive_period.iloc[i])
                adaptive_ma.iloc[i] = data['close'].iloc[i-period+1:i+1].mean()
        
        return adaptive_ma
    
    def _calculate_efficiency_ratio(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio"""
        price_change = abs(data['close'] - data['close'].shift(period))
        volatility = abs(data['close'].diff()).rolling(period).sum()
        
        efficiency_ratio = price_change / volatility
        efficiency_ratio = efficiency_ratio.fillna(0).clip(0, 1)
        
        return efficiency_ratio
```

---

## 🎯 Signal Generation Framework

### Signal Aggregation System

```python
class SignalAggregator:
    """
    Aggregate signals from multiple indicators with confidence weighting
    """
    
    def __init__(self):
        self.signal_weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volatility': 0.2,
            'volume': 0.15,
            'pattern': 0.1
        }
        self.confidence_threshold = 0.6
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate aggregated trading signals
        """
        signals = pd.DataFrame(index=data.index)
        
        # Individual signal components
        signals['trend_signal'] = self._generate_trend_signals(data)
        signals['momentum_signal'] = self._generate_momentum_signals(data)
        signals['volatility_signal'] = self._generate_volatility_signals(data)
        signals['volume_signal'] = self._generate_volume_signals(data)
        signals['pattern_signal'] = self._generate_pattern_signals(data)
        
        # Calculate confidence scores
        signals['trend_confidence'] = self._calculate_trend_confidence(data)
        signals['momentum_confidence'] = self._calculate_momentum_confidence(data)
        signals['volatility_confidence'] = self._calculate_volatility_confidence(data)
        signals['volume_confidence'] = self._calculate_volume_confidence(data)
        signals['pattern_confidence'] = self._calculate_pattern_confidence(data)
        
        # Aggregate signals with confidence weighting
        signals['aggregated_signal'] = (
            signals['trend_signal'] * signals['trend_confidence'] * self.signal_weights['trend'] +
            signals['momentum_signal'] * signals['momentum_confidence'] * self.signal_weights['momentum'] +
            signals['volatility_signal'] * signals['volatility_confidence'] * self.signal_weights['volatility'] +
            signals['volume_signal'] * signals['volume_confidence'] * self.signal_weights['volume'] +
            signals['pattern_signal'] * signals['pattern_confidence'] * self.signal_weights['pattern']
        )
        
        # Overall confidence
        signals['overall_confidence'] = (
            signals['trend_confidence'] * self.signal_weights['trend'] +
            signals['momentum_confidence'] * self.signal_weights['momentum'] +
            signals['volatility_confidence'] * self.signal_weights['volatility'] +
            signals['volume_confidence'] * self.signal_weights['volume'] +
            signals['pattern_confidence'] * self.signal_weights['pattern']
        )
        
        # Final signal with confidence filter
        signals['final_signal'] = np.where(
            signals['overall_confidence'] >= self.confidence_threshold,
            np.sign(signals['aggregated_signal']),
            0
        )
        
        return signals
    
    def _generate_trend_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend-based signals"""
        # Multiple MA crossover system
        ma_fast = data['close'].rolling(10).mean()
        ma_slow = data['close'].rolling(30).mean()
        
        # MACD signal
        macd_signal = np.where(
            (data['MACD'] > data['MACD_signal']) & (data['MACD'] > 0), 1,
            np.where((data['MACD'] < data['MACD_signal']) & (data['MACD'] < 0), -1, 0)
        )
        
        # ADX trend strength
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        
        adx_signal = np.where(
            (adx > 25) & (plus_di > minus_di), 1,
            np.where((adx > 25) & (plus_di < minus_di), -1, 0)
        )
        
        # Combine trend signals
        ma_signal = np.where(ma_fast > ma_slow, 1, -1)
        trend_signal = (ma_signal + macd_signal + adx_signal) / 3
        
        return pd.Series(trend_signal, index=data.index)
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum-based signals"""
        # RSI signals
        rsi_signal = np.where(
            data['RSI_14'] < 30, 1,  # Oversold
            np.where(data['RSI_14'] > 70, -1, 0)  # Overbought
        )
        
        # Stochastic signals
        stoch_signal = np.where(
            (data['STOCH_K'] < 20) & (data['STOCH_K'] > data['STOCH_D']), 1,
            np.where((data['STOCH_K'] > 80) & (data['STOCH_K'] < data['STOCH_D']), -1, 0)
        )
        
        # Williams %R
        willr = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=14)
        willr_signal = np.where(
            willr < -80, 1,  # Oversold
            np.where(willr > -20, -1, 0)  # Overbought
        )
        
        # Combine momentum signals
        momentum_signal = (rsi_signal + stoch_signal + willr_signal) / 3
        
        return pd.Series(momentum_signal, index=data.index)
    
    def _calculate_trend_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence in trend signals"""
        # ADX strength as confidence measure
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        adx_confidence = np.clip(adx / 50, 0, 1)  # Normalize to [0, 1]
        
        # Trend consistency
        returns = data['close'].pct_change()
        trend_consistency = abs(returns.rolling(20).mean()) / returns.rolling(20).std()
        trend_consistency = np.clip(trend_consistency, 0, 1)
        
        # Combine confidence measures
        confidence = (adx_confidence + trend_consistency) / 2
        
        return pd.Series(confidence, index=data.index).fillna(0)
```

### Multi-Timeframe Signal Coordination

```python
class MultiTimeframeAnalyzer:
    """
    Coordinate signals across multiple timeframes
    """
    
    def __init__(self, timeframes: List[str] = ['1h', '4h', '1d']):
        self.timeframes = timeframes
        self.timeframe_weights = {
            '1h': 0.2,
            '4h': 0.3,
            '1d': 0.5
        }
    
    def analyze_multiple_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze signals across multiple timeframes
        """
        mtf_signals = pd.DataFrame()
        
        for timeframe in self.timeframes:
            if timeframe in data_dict:
                tf_data = data_dict[timeframe]
                
                # Generate signals for this timeframe
                signal_aggregator = SignalAggregator()
                tf_signals = signal_aggregator.generate_signals(tf_data)
                
                # Resample to base timeframe if needed
                if timeframe != self.timeframes[0]:
                    tf_signals = self._resample_signals(tf_signals, timeframe)
                
                # Add timeframe prefix
                tf_signals = tf_signals.add_prefix(f'{timeframe}_')
                
                if mtf_signals.empty:
                    mtf_signals = tf_signals
                else:
                    mtf_signals = mtf_signals.join(tf_signals, how='outer')
        
        # Calculate multi-timeframe consensus
        mtf_signals['mtf_signal'] = self._calculate_mtf_consensus(mtf_signals)
        mtf_signals['mtf_confidence'] = self._calculate_mtf_confidence(mtf_signals)
        
        return mtf_signals
    
    def _calculate_mtf_consensus(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate consensus signal across timeframes"""
        consensus = pd.Series(0.0, index=signals.index)
        
        for timeframe in self.timeframes:
            signal_col = f'{timeframe}_final_signal'
            confidence_col = f'{timeframe}_overall_confidence'
            
            if signal_col in signals.columns and confidence_col in signals.columns:
                weight = self.timeframe_weights.get(timeframe, 0.33)
                weighted_signal = signals[signal_col] * signals[confidence_col] * weight
                consensus += weighted_signal.fillna(0)
        
        return consensus
    
    def _calculate_mtf_confidence(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate overall confidence across timeframes"""
        confidence = pd.Series(0.0, index=signals.index)
        
        for timeframe in self.timeframes:
            confidence_col = f'{timeframe}_overall_confidence'
            
            if confidence_col in signals.columns:
                weight = self.timeframe_weights.get(timeframe, 0.33)
                weighted_confidence = signals[confidence_col] * weight
                confidence += weighted_confidence.fillna(0)
        
        return confidence
```

---

## 📈 Pattern Recognition

### Candlestick Pattern Detection

```python
class CandlestickPatternDetector:
    """
    Detect and score candlestick patterns
    """
    
    def __init__(self):
        self.pattern_functions = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing_bullish': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'piercing_pattern': talib.CDLPIERCING,
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER
        }
    
    def detect_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all candlestick patterns
        """
        patterns = pd.DataFrame(index=data.index)
        
        for pattern_name, pattern_func in self.pattern_functions.items():
            patterns[pattern_name] = pattern_func(
                data['open'], data['high'], data['low'], data['close']
            )
        
        # Calculate pattern strength score
        patterns['pattern_score'] = self._calculate_pattern_score(patterns)
        
        # Pattern-based signals
        patterns['pattern_signal'] = self._generate_pattern_signals(patterns)
        
        return patterns
    
    def _calculate_pattern_score(self, patterns: pd.DataFrame) -> pd.Series:
        """Calculate overall pattern strength score"""
        bullish_patterns = ['hammer', 'engulfing_bullish', 'morning_star', 
                           'three_white_soldiers', 'piercing_pattern']
        bearish_patterns = ['hanging_man', 'shooting_star', 'evening_star',
                           'three_black_crows', 'dark_cloud_cover']
        
        bullish_score = patterns[bullish_patterns].sum(axis=1)
        bearish_score = patterns[bearish_patterns].sum(axis=1)
        
        pattern_score = bullish_score - bearish_score
        
        return pattern_score
    
    def _generate_pattern_signals(self, patterns: pd.DataFrame) -> pd.Series:
        """Generate trading signals from patterns"""
        signals = np.where(
            patterns['pattern_score'] > 100, 1,  # Strong bullish
            np.where(patterns['pattern_score'] < -100, -1, 0)  # Strong bearish
        )
        
        return pd.Series(signals, index=patterns.index)
```

### Chart Pattern Recognition

```python
class ChartPatternDetector:
    """
    Detect chart patterns using price action analysis
    """
    
    def __init__(self, min_pattern_length: int = 20):
        self.min_pattern_length = min_pattern_length
    
    def detect_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect various chart patterns
        """
        patterns = pd.DataFrame(index=data.index)
        
        # Support and resistance levels
        support_resistance = self._find_support_resistance(data)
        patterns = patterns.join(support_resistance)
        
        # Triangle patterns
        triangles = self._detect_triangles(data)
        patterns = patterns.join(triangles)
        
        # Head and shoulders
        head_shoulders = self._detect_head_shoulders(data)
        patterns = patterns.join(head_shoulders)
        
        # Double tops/bottoms
        double_patterns = self._detect_double_patterns(data)
        patterns = patterns.join(double_patterns)
        
        # Flag and pennant patterns
        flag_pennant = self._detect_flag_pennant(data)
        patterns = patterns.join(flag_pennant)
        
        return patterns
    
    def _find_support_resistance(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Find support and resistance levels"""
        patterns = pd.DataFrame(index=data.index)
        
        # Local minima (support)
        local_min = data['low'].rolling(window, center=True).min() == data['low']
        patterns['support_level'] = np.where(local_min, data['low'], np.nan)
        
        # Local maxima (resistance)
        local_max = data['high'].rolling(window, center=True).max() == data['high']
        patterns['resistance_level'] = np.where(local_max, data['high'], np.nan)
        
        # Fill forward support/resistance levels
        patterns['support_level'] = patterns['support_level'].fillna(method='ffill')
        patterns['resistance_level'] = patterns['resistance_level'].fillna(method='ffill')
        
        # Distance from support/resistance
        patterns['distance_from_support'] = (data['close'] - patterns['support_level']) / patterns['support_level']
        patterns['distance_from_resistance'] = (patterns['resistance_level'] - data['close']) / data['close']
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect triangle patterns"""
        patterns = pd.DataFrame(index=data.index)
        
        # Calculate trend lines
        highs_trend = self._calculate_trend_line(data['high'], 'descending')
        lows_trend = self._calculate_trend_line(data['low'], 'ascending')
        
        # Ascending triangle: horizontal resistance, ascending support
        ascending_triangle = (
            (highs_trend['slope'] < 0.001) &  # Flat resistance
            (lows_trend['slope'] > 0.001)     # Rising support
        )
        
        # Descending triangle: descending resistance, horizontal support
        descending_triangle = (
            (highs_trend['slope'] < -0.001) &  # Falling resistance
            (lows_trend['slope'] < 0.001)      # Flat support
        )
        
        # Symmetrical triangle: converging trend lines
        symmetrical_triangle = (
            (highs_trend['slope'] < -0.001) &  # Falling resistance
            (lows_trend['slope'] > 0.001)      # Rising support
        )
        
        patterns['ascending_triangle'] = ascending_triangle
        patterns['descending_triangle'] = descending_triangle
        patterns['symmetrical_triangle'] = symmetrical_triangle
        
        return patterns
    
    def _calculate_trend_line(self, series: pd.Series, direction: str, window: int = 50) -> pd.DataFrame:
        """Calculate trend line slope and R-squared"""
        result = pd.DataFrame(index=series.index)
        result['slope'] = np.nan
        result['r_squared'] = np.nan
        
        for i in range(window, len(series)):
            y = series.iloc[i-window:i].values
            x = np.arange(len(y))
            
            # Filter for relevant points based on direction
            if direction == 'ascending':
                # Use local minima for ascending trend line
                local_min_mask = (y == np.minimum.accumulate(y[::-1])[::-1])
                if local_min_mask.sum() >= 2:
                    x_filtered = x[local_min_mask]
                    y_filtered = y[local_min_mask]
                else:
                    continue
            elif direction == 'descending':
                # Use local maxima for descending trend line
                local_max_mask = (y == np.maximum.accumulate(y[::-1])[::-1])
                if local_max_mask.sum() >= 2:
                    x_filtered = x[local_max_mask]
                    y_filtered = y[local_max_mask]
                else:
                    continue
            else:
                x_filtered = x
                y_filtered = y
            
            if len(x_filtered) >= 2:
                # Linear regression
                slope, intercept = np.polyfit(x_filtered, y_filtered, 1)
                y_pred = slope * x_filtered + intercept
                
                # R-squared
                ss_res = np.sum((y_filtered - y_pred) ** 2)
                ss_tot = np.sum((y_filtered - np.mean(y_filtered)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                result.iloc[i]['slope'] = slope
                result.iloc[i]['r_squared'] = r_squared
        
        return result
```

---

## 📊 Performance Metrics

### Advanced Performance Analytics

```python
class AdvancedPerformanceMetrics:
    """
    Calculate comprehensive performance metrics for technical analysis strategies
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_all_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(returns))
        
        # Distribution metrics
        metrics.update(self._calculate_distribution_metrics(returns))
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        return metrics
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        cumulative_returns = (1 + returns).cumprod()
        
        return {
            'total_return': cumulative_returns.iloc[-1] - 1,
            'annualized_return': (cumulative_returns.iloc[-1] ** (252 / len(returns))) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'avg_daily_return': returns.mean(),
            'median_daily_return': returns.median(),
            'positive_days': (returns > 0).sum() / len(returns),
            'negative_days': (returns < 0).sum() / len(returns)
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics"""
        return {
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'cvar_99': returns[returns <= np.percentile(returns, 1)].mean(),
            'downside_deviation': self._calculate_downside_deviation(returns),
            'upside_deviation': self._calculate_upside_deviation(returns)
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        excess_returns = returns - self.risk_free_rate / 252
        
        return {
            'sharpe_ratio': excess_returns.mean() / returns.std() * np.sqrt(252),
            'sortino_ratio': excess_returns.mean() / self._calculate_downside_deviation(returns) * np.sqrt(252),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'omega_ratio': self._calculate_omega_ratio(returns),
            'information_ratio': returns.mean() / returns.std() * np.sqrt(252)
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-related metrics"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean(),
            'max_drawdown_duration': self._calculate_max_drawdown_duration(drawdown),
            'recovery_factor': (cumulative_returns.iloc[-1] - 1) / abs(drawdown.min()),
            'pain_index': -drawdown.mean()
        }
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return distribution metrics"""
        from scipy import stats
        
        return {
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'jarque_bera_stat': stats.jarque_bera(returns)[0],
            'jarque_bera_pvalue': stats.jarque_bera(returns)[1],
            'tail_ratio': np.percentile(returns, 95) / abs(np.percentile(returns, 5))
        }
    
    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        return np.sqrt(np.mean((downside_returns - target) ** 2))
    
    def _calculate_upside_deviation(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate upside deviation"""
        upside_returns = returns[returns > target]
        return np.sqrt(np.mean((upside_returns - target) ** 2))
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        annualized_return = (cumulative_returns.iloc[-1] ** (252 / len(returns))) - 1
        max_drawdown = abs(drawdown.min())
        
        return annualized_return / max_drawdown if max_drawdown != 0 else 0
    
    def _calculate_omega_ratio(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate Omega ratio"""
        excess_returns = returns - target
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        
        return positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
```

---

## 🎯 Implementation Examples

### Complete Strategy Implementation

```python
class MomentumMeanReversionStrategy:
    """
    Example strategy combining momentum and mean reversion signals
    """
    
    def __init__(self):
        self.signal_aggregator = SignalAggregator()
        self.pattern_detector = CandlestickPatternDetector()
        self.regime_detector = MarketRegimeDetector()
        self.performance_metrics = AdvancedPerformanceMetrics()
        
        # Strategy parameters
        self.momentum_threshold = 0.6
        self.mean_reversion_threshold = 0.7
        self.min_confidence = 0.5
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using the complete framework
        """
        # Calculate all technical indicators
        data = self._calculate_all_indicators(data)
        
        # Detect market regime
        data['market_regime'] = self.regime_detector.detect_regime(data)
        
        # Generate base signals
        signals = self.signal_aggregator.generate_signals(data)
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(data)
        signals = signals.join(patterns[['pattern_signal', 'pattern_score']])
        
        # Regime-adjusted signals
        signals['regime_adjusted_signal'] = self._adjust_signals_for_regime(
            signals, data['market_regime']
        )
        
        # Position sizing based on volatility
        signals['position_size'] = self._calculate_position_size(data, signals)
        
        # Final trading decisions
        signals['trade_signal'] = self._generate_trade_signals(signals)
        
        return signals
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required technical indicators"""
        # Moving averages
        data = calculate_moving_averages(data)
        
        # MACD
        data = calculate_macd(data)
        
        # RSI
        data = calculate_rsi(data)
        
        # Bollinger Bands
        data = calculate_bollinger_bands(data)
        
        # ATR
        data = calculate_atr(data)
        
        # Volume indicators
        data = calculate_vwap(data)
        data = calculate_obv(data)
        
        # Stochastic
        data = calculate_stochastic(data)
        
        return data
    
    def _adjust_signals_for_regime(self, signals: pd.DataFrame, regime: pd.Series) -> pd.Series:
        """Adjust signals based on market regime"""
        adjusted_signals = signals['final_signal'].copy()
        
        # In trending markets, favor momentum signals
        trending_mask = regime == 'trending'
        momentum_boost = np.where(
            trending_mask & (signals['momentum_signal'] != 0),
            signals['momentum_signal'] * 1.2,  # Boost momentum signals
            signals['final_signal']
        )
        
        # In ranging markets, favor mean reversion
        ranging_mask = regime == 'ranging'
        mean_reversion_boost = np.where(
            ranging_mask & (signals['volatility_signal'] != 0),
            signals['volatility_signal'] * 1.2,  # Boost mean reversion signals
            momentum_boost
        )
        
        # In volatile markets, reduce position sizes
        volatile_mask = regime == 'volatile'
        volatility_adjustment = np.where(
            volatile_mask,
            mean_reversion_boost * 0.5,  # Reduce signal strength
            mean_reversion_boost
        )
        
        return pd.Series(volatility_adjustment, index=signals.index)
    
    def _calculate_position_size(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate position size based on volatility and confidence"""
        # Base position size (1% risk per trade)
        base_risk = 0.01
        
        # Adjust for volatility (ATR-based)
        volatility_adjustment = 1 / (data['ATR_pct'] / data['ATR_pct'].rolling(50).mean())
        volatility_adjustment = np.clip(volatility_adjustment, 0.5, 2.0)
        
        # Adjust for signal confidence
        confidence_adjustment = signals['overall_confidence']
        
        # Calculate final position size
        position_size = base_risk * volatility_adjustment * confidence_adjustment
        position_size = np.clip(position_size, 0.001, 0.05)  # Min 0.1%, Max 5%
        
        return pd.Series(position_size, index=signals.index)
    
    def _generate_trade_signals(self, signals: pd.DataFrame) -> pd.Series:
        """Generate final trade signals with filters"""
        trade_signals = pd.Series(0, index=signals.index)
        
        # Entry conditions
        long_entry = (
            (signals['regime_adjusted_signal'] > self.momentum_threshold) &
            (signals['overall_confidence'] > self.min_confidence) &
            (signals['pattern_signal'] >= 0)  # No bearish patterns
        )
        
        short_entry = (
            (signals['regime_adjusted_signal'] < -self.momentum_threshold) &
            (signals['overall_confidence'] > self.min_confidence) &
            (signals['pattern_signal'] <= 0)  # No bullish patterns
        )
        
        trade_signals[long_entry] = 1
        trade_signals[short_entry] = -1
        
        return trade_signals
    
    def backtest_strategy(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """
        Backtest the complete strategy
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Simulate trades
        portfolio_value = [initial_capital]
        positions = [0]
        trades = []
        
        current_position = 0
        current_value = initial_capital
        
        for i in range(1, len(data)):
            signal = signals['trade_signal'].iloc[i]
            position_size = signals['position_size'].iloc[i]
            price = data['close'].iloc[i]
            
            # Position change
            if signal != 0 and signal != current_position:
                # Close existing position
                if current_position != 0:
                    pnl = current_position * (price - entry_price) * shares
                    current_value += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data.index[i],
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': current_position,
                        'shares': shares,
                        'pnl': pnl,
                        'return': pnl / (entry_price * shares)
                    })
                
                # Open new position
                if signal != 0:
                    risk_amount = current_value * position_size
                    shares = risk_amount / price
                    entry_price = price
                    entry_date = data.index[i]
                    current_position = signal
                else:
                    current_position = 0
            
            # Update portfolio value
            if current_position != 0:
                unrealized_pnl = current_position * (price - entry_price) * shares
                portfolio_value.append(current_value + unrealized_pnl)
            else:
                portfolio_value.append(current_value)
            
            positions.append(current_position)
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_value).pct_change().dropna()
        metrics = self.performance_metrics.calculate_all_metrics(portfolio_returns)
        
        return {
            'portfolio_value': portfolio_value,
            'positions': positions,
            'trades': trades,
            'metrics': metrics,
            'signals': signals
        }
```

---

## 🔧 Configuration and Usage

### Strategy Configuration

```yaml
# technical_analysis_config.yaml
strategy:
  name: "MomentumMeanReversion"
  timeframes: ["1h", "4h", "1d"]
  
indicators:
  trend:
    - name: "MACD"
      params: {fast: 12, slow: 26, signal: 9}
    - name: "ADX"
      params: {period: 14}
    - name: "MovingAverages"
      params: {periods: [10, 20, 50, 200]}
  
  momentum:
    - name: "RSI"
      params: {period: 14}
    - name: "Stochastic"
      params: {k_period: 14, d_period: 3}
    - name: "Williams_R"
      params: {period: 14}
  
  volatility:
    - name: "BollingerBands"
      params: {period: 20, std_dev: 2}
    - name: "ATR"
      params: {period: 14}
  
  volume:
    - name: "VWAP"
    - name: "OBV"

signal_generation:
  aggregation_weights:
    trend: 0.30
    momentum: 0.25
    volatility: 0.20
    volume: 0.15
    pattern: 0.10
  
  confidence_threshold: 0.60
  signal_threshold: 0.70

risk_management:
  max_position_size: 0.05  # 5% max risk per trade
  min_position_size: 0.001  # 0.1% min risk per trade
  volatility_adjustment: true
  regime_adjustment: true

backtesting:
  initial_capital: 100000
  commission: 0.001  # 0.1% per trade
  slippage: 0.0005   # 0.05% slippage
```

### Usage Examples

```python
# Example 1: Basic Technical Analysis
from technical_analysis import TechnicalAnalyzer

# Initialize analyzer
analyzer = TechnicalAnalyzer()

# Load data
data = pd.read_csv('AAPL_1h.csv', index_col='timestamp', parse_dates=True)

# Calculate indicators
data_with_indicators = analyzer.calculate_all_indicators(data)

# Generate signals
signals = analyzer.generate_signals(data_with_indicators)

print(f"Generated {signals['trade_signal'].abs().sum()} trading signals")
```

```python
# Example 2: Multi-Timeframe Analysis
from technical_analysis import MultiTimeframeAnalyzer

# Initialize multi-timeframe analyzer
mtf_analyzer = MultiTimeframeAnalyzer(['1h', '4h', '1d'])

# Load data for different timeframes
data_dict = {
    '1h': pd.read_csv('AAPL_1h.csv', index_col='timestamp', parse_dates=True),
    '4h': pd.read_csv('AAPL_4h.csv', index_col='timestamp', parse_dates=True),
    '1d': pd.read_csv('AAPL_1d.csv', index_col='timestamp', parse_dates=True)
}

# Analyze across timeframes
mtf_signals = mtf_analyzer.analyze_multiple_timeframes(data_dict)

print(f"Multi-timeframe consensus: {mtf_signals['mtf_signal'].iloc[-1]:.2f}")
```

```python
# Example 3: Complete Strategy Backtest
from technical_analysis import MomentumMeanReversionStrategy

# Initialize strategy
strategy = MomentumMeanReversionStrategy()

# Load historical data
data = pd.read_csv('AAPL_daily.csv', index_col='date', parse_dates=True)

# Run backtest
results = strategy.backtest_strategy(data, initial_capital=100000)

# Print performance summary
print("\n=== BACKTEST RESULTS ===")
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Annualized Return: {results['metrics']['annualized_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
print(f"Number of Trades: {len(results['trades'])}")
```

---

## 📈 Advanced Features

### 1. Dynamic Parameter Optimization

```python
class ParameterOptimizer:
    """
    Optimize technical indicator parameters using walk-forward analysis
    """
    
    def __init__(self, strategy_class, optimization_metric='sharpe_ratio'):
        self.strategy_class = strategy_class
        self.optimization_metric = optimization_metric
    
    def optimize_parameters(self, data: pd.DataFrame, param_ranges: Dict, 
                          walk_forward_periods: int = 252) -> Dict:
        """
        Optimize parameters using walk-forward analysis
        """
        from itertools import product
        from sklearn.model_selection import ParameterGrid
        
        # Generate parameter combinations
        param_grid = ParameterGrid(param_ranges)
        
        best_params = None
        best_score = -np.inf
        optimization_results = []
        
        # Walk-forward optimization
        for start_idx in range(0, len(data) - walk_forward_periods, walk_forward_periods // 4):
            end_idx = start_idx + walk_forward_periods
            train_data = data.iloc[start_idx:end_idx]
            
            if len(train_data) < walk_forward_periods // 2:
                continue
            
            period_best_params = None
            period_best_score = -np.inf
            
            # Test each parameter combination
            for params in param_grid:
                try:
                    # Initialize strategy with parameters
                    strategy = self.strategy_class(**params)
                    
                    # Run backtest
                    results = strategy.backtest_strategy(train_data)
                    score = results['metrics'][self.optimization_metric]
                    
                    if score > period_best_score:
                        period_best_score = score
                        period_best_params = params
                        
                except Exception as e:
                    continue
            
            if period_best_params:
                optimization_results.append({
                    'period': f"{data.index[start_idx]} to {data.index[end_idx-1]}",
                    'params': period_best_params,
                    'score': period_best_score
                })
        
        # Select overall best parameters
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x['score'])
            best_params = best_result['params']
            best_score = best_result['score']
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': optimization_results
        }
```

### 2. Real-Time Signal Generation

```python
class RealTimeSignalGenerator:
    """
    Generate trading signals in real-time
    """
    
    def __init__(self, strategy, update_frequency: int = 60):
        self.strategy = strategy
        self.update_frequency = update_frequency  # seconds
        self.data_buffer = pd.DataFrame()
        self.last_signals = pd.DataFrame()
        
    def start_real_time_analysis(self, data_source):
        """
        Start real-time signal generation
        """
        import time
        
        while True:
            try:
                # Get latest data
                new_data = data_source.get_latest_data()
                
                if not new_data.empty:
                    # Update data buffer
                    self.data_buffer = pd.concat([self.data_buffer, new_data])
                    
                    # Keep only recent data (e.g., last 1000 bars)
                    if len(self.data_buffer) > 1000:
                        self.data_buffer = self.data_buffer.tail(1000)
                    
                    # Generate signals
                    if len(self.data_buffer) >= 100:  # Minimum data required
                        signals = self.strategy.generate_signals(self.data_buffer)
                        
                        # Check for new signals
                        latest_signal = signals.iloc[-1]
                        
                        if self._is_new_signal(latest_signal):
                            self._process_new_signal(latest_signal)
                        
                        self.last_signals = signals
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                print(f"Error in real-time analysis: {e}")
                time.sleep(self.update_frequency)
    
    def _is_new_signal(self, signal) -> bool:
        """
        Check if this is a new trading signal
        """
        if self.last_signals.empty:
            return signal['trade_signal'] != 0
        
        last_signal = self.last_signals.iloc[-1]
        return (
            signal['trade_signal'] != last_signal['trade_signal'] and
            signal['trade_signal'] != 0
        )
    
    def _process_new_signal(self, signal):
        """
        Process new trading signal
        """
        signal_info = {
            'timestamp': signal.name,
            'signal': signal['trade_signal'],
            'confidence': signal['overall_confidence'],
            'position_size': signal['position_size']
        }
        
        print(f"NEW SIGNAL: {signal_info}")
        
        # Here you would typically:
        # 1. Send signal to execution engine
        # 2. Log signal to database
        # 3. Send notifications
        # 4. Update risk management system
```

### 3. Machine Learning Integration

```python
class MLEnhancedTechnicalAnalysis:
    """
    Enhance technical analysis with machine learning
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.signal_classifier = None
        self.regime_classifier = None
    
    def train_signal_classifier(self, historical_data: pd.DataFrame, 
                              future_returns: pd.Series):
        """
        Train ML model to classify signal quality
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Generate technical features
        features = self.feature_engineer.create_technical_features(historical_data)
        
        # Create labels based on future returns
        labels = self._create_signal_labels(future_returns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.signal_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.signal_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.signal_classifier.score(X_train, y_train)
        test_score = self.signal_classifier.score(X_test, y_test)
        
        print(f"Signal Classifier - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(
                features.columns, 
                self.signal_classifier.feature_importances_
            ))
        }
    
    def enhance_signals_with_ml(self, data: pd.DataFrame, 
                               base_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance technical signals with ML predictions
        """
        if self.signal_classifier is None:
            return base_signals
        
        # Generate features
        features = self.feature_engineer.create_technical_features(data)
        
        # Get ML predictions
        ml_predictions = self.signal_classifier.predict_proba(features)[:, 1]  # Probability of positive signal
        ml_confidence = np.max(self.signal_classifier.predict_proba(features), axis=1)
        
        # Enhance signals
        enhanced_signals = base_signals.copy()
        enhanced_signals['ml_prediction'] = ml_predictions
        enhanced_signals['ml_confidence'] = ml_confidence
        
        # Combine technical and ML signals
        enhanced_signals['combined_signal'] = (
            enhanced_signals['final_signal'] * 0.6 +
            (ml_predictions - 0.5) * 2 * 0.4  # Convert to [-1, 1] range
        )
        
        enhanced_signals['combined_confidence'] = (
            enhanced_signals['overall_confidence'] * 0.7 +
            ml_confidence * 0.3
        )
        
        return enhanced_signals
    
    def _create_signal_labels(self, future_returns: pd.Series, 
                            threshold: float = 0.02) -> pd.Series:
        """
        Create binary labels for signal classification
        """
        # Look ahead 5 periods for signal validation
        forward_returns = future_returns.rolling(5).sum().shift(-5)
        
        labels = np.where(
            forward_returns > threshold, 1,  # Good signal
            np.where(forward_returns < -threshold, 0, 0.5)  # Bad signal, Neutral
        )
        
        # Convert to binary (remove neutral)
        binary_labels = labels[labels != 0.5]
        
        return pd.Series(binary_labels, index=future_returns.index[labels != 0.5])
```

---

## 🎯 Best Practices

### 1. Signal Validation

- **Out-of-Sample Testing**: Always validate signals on unseen data
- **Walk-Forward Analysis**: Use rolling windows for parameter optimization
- **Cross-Validation**: Implement time-series aware cross-validation
- **Regime Awareness**: Adjust signals based on market conditions

### 2. Risk Management Integration

- **Position Sizing**: Use volatility-adjusted position sizing
- **Stop Losses**: Implement ATR-based stop losses
- **Correlation Monitoring**: Monitor signal correlation across assets
- **Drawdown Control**: Implement maximum drawdown limits

### 3. Performance Monitoring

- **Real-Time Metrics**: Monitor performance in real-time
- **Signal Decay**: Track signal effectiveness over time
- **Regime Performance**: Analyze performance by market regime
- **Attribution Analysis**: Understand sources of returns

---

**📚 This comprehensive technical analysis guide provides the foundation for implementing sophisticated trading strategies with advanced signal generation, pattern recognition, and performance evaluation capabilities. The modular design allows for easy customization and extension of the analysis framework.**