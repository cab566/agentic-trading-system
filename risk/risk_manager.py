#!/usr/bin/env python3
"""
Risk Manager

Comprehensive risk management system for trading operations.
Handles position sizing, risk assessment, portfolio risk, and risk monitoring.

Author: AI Trading System v2.0
Date: January 2025
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(Enum):
    """Types of risk"""
    MARKET = "market"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"

@dataclass
class RiskMetrics:
    """Risk metrics container"""
    var_1d: float = 0.0  # 1-day Value at Risk
    var_5d: float = 0.0  # 5-day Value at Risk
    expected_shortfall: float = 0.0  # Expected Shortfall (CVaR)
    max_drawdown: float = 0.0  # Maximum Drawdown
    sharpe_ratio: float = 0.0  # Sharpe Ratio
    sortino_ratio: float = 0.0  # Sortino Ratio
    beta: float = 0.0  # Market Beta
    volatility: float = 0.0  # Annualized Volatility
    correlation_risk: float = 0.0  # Portfolio Correlation Risk

@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    position_size: float
    market_value: float
    risk_level: RiskLevel
    var_contribution: float
    concentration_risk: float
    liquidity_risk: float
    volatility_risk: float
    recommendations: List[str]

@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment"""
    total_value: float
    risk_metrics: RiskMetrics
    risk_level: RiskLevel
    position_risks: List[PositionRisk]
    sector_concentration: Dict[str, float]
    currency_exposure: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime

class RiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(self, config_manager=None):
        """Initialize the risk manager"""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_position_size = 0.10  # 10% max position size
        self.max_sector_concentration = 0.25  # 25% max sector concentration
        self.max_portfolio_var = 0.05  # 5% max daily VaR
        self.confidence_level = 0.95  # 95% confidence level for VaR
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.LOW: 0.02,
            RiskLevel.MEDIUM: 0.05,
            RiskLevel.HIGH: 0.10,
            RiskLevel.CRITICAL: 0.20
        }
        
        self.logger.info("Risk Manager initialized")
    
    def assess_position_risk(self, symbol: str, position_size: float, 
                           market_data: pd.DataFrame, 
                           portfolio_value: float) -> PositionRisk:
        """
        Assess risk for an individual position
        
        Args:
            symbol: Stock symbol
            position_size: Position size (shares or dollar amount)
            market_data: Historical market data
            portfolio_value: Total portfolio value
            
        Returns:
            PositionRisk object with risk assessment
        """
        try:
            if market_data.empty:
                return self._create_default_position_risk(symbol, position_size)
            
            # Calculate position metrics
            current_price = market_data['close'].iloc[-1] if 'close' in market_data.columns else 0
            market_value = position_size * current_price
            position_weight = market_value / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate VaR contribution
            var_1d = self._calculate_var(returns, confidence_level=self.confidence_level)
            var_contribution = market_value * var_1d
            
            # Assess concentration risk
            concentration_risk = min(position_weight / self.max_position_size, 1.0)
            
            # Assess liquidity risk (simplified - based on volume)
            avg_volume = market_data['volume'].mean() if 'volume' in market_data.columns else 0
            liquidity_risk = self._assess_liquidity_risk(position_size, avg_volume, current_price)
            
            # Determine overall risk level
            risk_score = max(concentration_risk, volatility, liquidity_risk)
            risk_level = self._determine_risk_level(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_position_recommendations(
                position_weight, volatility, liquidity_risk, concentration_risk
            )
            
            return PositionRisk(
                symbol=symbol,
                position_size=position_size,
                market_value=market_value,
                risk_level=risk_level,
                var_contribution=var_contribution,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                volatility_risk=volatility,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing position risk for {symbol}: {e}")
            return self._create_default_position_risk(symbol, position_size)
    
    def assess_portfolio_risk(self, positions: Dict[str, Dict], 
                            market_data: Dict[str, pd.DataFrame]) -> PortfolioRisk:
        """
        Assess overall portfolio risk
        
        Args:
            positions: Dictionary of positions {symbol: {size, value, etc.}}
            market_data: Dictionary of market data {symbol: DataFrame}
            
        Returns:
            PortfolioRisk object with comprehensive risk assessment
        """
        try:
            # Calculate total portfolio value
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            
            if total_value == 0:
                return self._create_default_portfolio_risk()
            
            # Assess individual position risks
            position_risks = []
            for symbol, position in positions.items():
                symbol_data = market_data.get(symbol, pd.DataFrame())
                position_risk = self.assess_position_risk(
                    symbol, position.get('size', 0), symbol_data, total_value
                )
                position_risks.append(position_risk)
            
            # Calculate portfolio-level metrics
            risk_metrics = self._calculate_portfolio_metrics(positions, market_data, total_value)
            
            # Assess sector concentration
            sector_concentration = self._calculate_sector_concentration(positions)
            
            # Assess currency exposure (simplified - assume USD for now)
            currency_exposure = {"USD": 1.0}
            
            # Determine overall portfolio risk level
            portfolio_var = max(risk_metrics.var_1d, risk_metrics.var_5d)
            risk_level = self._determine_risk_level(abs(portfolio_var) / total_value)
            
            # Generate portfolio recommendations
            recommendations = self._generate_portfolio_recommendations(
                risk_metrics, sector_concentration, position_risks
            )
            
            return PortfolioRisk(
                total_value=total_value,
                risk_metrics=risk_metrics,
                risk_level=risk_level,
                position_risks=position_risks,
                sector_concentration=sector_concentration,
                currency_exposure=currency_exposure,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
            return self._create_default_portfolio_risk()
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, risk_amount: float) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Stock symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            risk_amount: Maximum amount to risk on this trade
            
        Returns:
            Recommended position size (number of shares)
        """
        try:
            if entry_price <= 0 or stop_loss <= 0 or risk_amount <= 0:
                return 0.0
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share == 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / risk_per_share
            
            # Apply maximum position size constraint
            max_shares = (risk_amount * self.max_position_size) / entry_price
            position_size = min(position_size, max_shares)
            
            return max(0, position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def validate_trade(self, symbol: str, trade_type: str, quantity: float, 
                      price: float, portfolio_value: float) -> Dict[str, Any]:
        """
        Validate a proposed trade against risk parameters
        
        Args:
            symbol: Stock symbol
            trade_type: 'buy' or 'sell'
            quantity: Number of shares
            price: Trade price
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with validation results
        """
        try:
            trade_value = quantity * price
            position_weight = trade_value / portfolio_value if portfolio_value > 0 else 0
            
            validation = {
                'approved': True,
                'warnings': [],
                'errors': [],
                'risk_level': RiskLevel.LOW,
                'position_weight': position_weight,
                'trade_value': trade_value
            }
            
            # Check position size limits
            if position_weight > self.max_position_size:
                validation['errors'].append(
                    f"Position size ({position_weight:.2%}) exceeds maximum allowed "
                    f"({self.max_position_size:.2%})"
                )
                validation['approved'] = False
                validation['risk_level'] = RiskLevel.CRITICAL
            
            # Check for concentration risk
            if position_weight > self.max_position_size * 0.8:
                validation['warnings'].append(
                    f"Position size ({position_weight:.2%}) approaching maximum limit"
                )
                validation['risk_level'] = RiskLevel.HIGH
            
            # Additional validations for buy orders
            if trade_type.lower() == 'buy':
                # Check available cash (simplified)
                if trade_value > portfolio_value * 0.95:  # Assume 95% cash utilization limit
                    validation['errors'].append("Insufficient cash for trade")
                    validation['approved'] = False
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating trade for {symbol}: {e}")
            return {
                'approved': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'risk_level': RiskLevel.CRITICAL
            }
    
    def monitor_risk_limits(self, portfolio_risk: PortfolioRisk) -> List[Dict[str, Any]]:
        """
        Monitor portfolio against risk limits and generate alerts
        
        Args:
            portfolio_risk: Current portfolio risk assessment
            
        Returns:
            List of risk alerts
        """
        alerts = []
        
        try:
            # Check VaR limits
            portfolio_var_pct = abs(portfolio_risk.risk_metrics.var_1d) / portfolio_risk.total_value
            if portfolio_var_pct > self.max_portfolio_var:
                alerts.append({
                    'type': 'var_breach',
                    'severity': 'high',
                    'message': f"Portfolio VaR ({portfolio_var_pct:.2%}) exceeds limit ({self.max_portfolio_var:.2%})",
                    'timestamp': datetime.now()
                })
            
            # Check sector concentration
            for sector, concentration in portfolio_risk.sector_concentration.items():
                if concentration > self.max_sector_concentration:
                    alerts.append({
                        'type': 'concentration_risk',
                        'severity': 'medium',
                        'message': f"Sector {sector} concentration ({concentration:.2%}) exceeds limit ({self.max_sector_concentration:.2%})",
                        'timestamp': datetime.now()
                    })
            
            # Check individual position risks
            for position_risk in portfolio_risk.position_risks:
                if position_risk.risk_level == RiskLevel.CRITICAL:
                    alerts.append({
                        'type': 'position_risk',
                        'severity': 'critical',
                        'message': f"Position {position_risk.symbol} has critical risk level",
                        'timestamp': datetime.now()
                    })
            
            # Check maximum drawdown
            if portfolio_risk.risk_metrics.max_drawdown < -0.20:  # 20% drawdown threshold
                alerts.append({
                    'type': 'drawdown',
                    'severity': 'high',
                    'message': f"Maximum drawdown ({portfolio_risk.risk_metrics.max_drawdown:.2%}) exceeds threshold",
                    'timestamp': datetime.now()
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error monitoring risk limits: {e}")
            return []
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if returns.empty:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if returns.empty:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _calculate_portfolio_metrics(self, positions: Dict, market_data: Dict, 
                                   total_value: float) -> RiskMetrics:
        """Calculate portfolio-level risk metrics"""
        try:
            # Collect all returns
            all_returns = []
            weights = []
            
            for symbol, position in positions.items():
                if symbol in market_data and not market_data[symbol].empty:
                    returns = market_data[symbol]['close'].pct_change().dropna()
                    if not returns.empty:
                        all_returns.append(returns)
                        weight = position.get('value', 0) / total_value
                        weights.append(weight)
            
            if not all_returns:
                return RiskMetrics()
            
            # Calculate portfolio returns (simplified equal weighting if weights not available)
            if len(all_returns) == 1:
                portfolio_returns = all_returns[0]
            else:
                # Simple average for now - in production, use proper weighted returns
                portfolio_returns = pd.concat(all_returns, axis=1).mean(axis=1)
            
            # Calculate metrics
            var_1d = self._calculate_var(portfolio_returns)
            var_5d = var_1d * np.sqrt(5)  # Simplified scaling
            expected_shortfall = self._calculate_expected_shortfall(portfolio_returns)
            
            # Calculate other metrics
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns = portfolio_returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return RiskMetrics(
                var_1d=var_1d * total_value,
                var_5d=var_5d * total_value,
                expected_shortfall=expected_shortfall * total_value,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                volatility=volatility,
                beta=0.0,  # Would need market data to calculate
                correlation_risk=0.0  # Simplified for now
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return RiskMetrics()
    
    def _assess_liquidity_risk(self, position_size: float, avg_volume: float, 
                             price: float) -> float:
        """Assess liquidity risk for a position"""
        if avg_volume == 0 or price == 0:
            return 1.0  # Maximum liquidity risk
        
        # Calculate position as percentage of average daily volume
        daily_dollar_volume = avg_volume * price
        position_value = position_size * price
        
        if daily_dollar_volume == 0:
            return 1.0
        
        volume_ratio = position_value / daily_dollar_volume
        
        # Risk increases exponentially with volume ratio
        return min(volume_ratio * 2, 1.0)
    
    def _calculate_sector_concentration(self, positions: Dict) -> Dict[str, float]:
        """Calculate sector concentration using real sector data from financial providers"""
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        
        if total_value == 0:
            return {}
        
        # Get real sector data from financial data providers
        sector_map = self._get_real_sector_data(list(positions.keys()))
        
        sector_values = {}
        for symbol, position in positions.items():
            sector = sector_map.get(symbol, 'Other')
            value = position.get('value', 0)
            sector_values[sector] = sector_values.get(sector, 0) + value
        
        # Convert to percentages
        return {sector: value / total_value for sector, value in sector_values.items()}
    
    def _get_real_sector_data(self, symbols: List[str]) -> Dict[str, str]:
        """Get real sector classifications from financial data providers"""
        try:
            import yfinance as yf
            sector_map = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get sector from Yahoo Finance
                    sector = info.get('sector', 'Other')
                    if not sector or sector == 'None':
                        # Try industry as fallback
                        industry = info.get('industry', 'Other')
                        # Map common industries to sectors
                        sector = self._map_industry_to_sector(industry)
                    
                    sector_map[symbol] = sector
                    
                except Exception as e:
                    logger.warning(f"Could not get sector data for {symbol}: {e}")
                    sector_map[symbol] = 'Other'
            
            return sector_map
            
        except ImportError:
            logger.warning("yfinance not available, using fallback sector mapping")
            return self._get_fallback_sector_data(symbols)
        except Exception as e:
            logger.error(f"Error getting real sector data: {e}")
            return self._get_fallback_sector_data(symbols)
    
    def _map_industry_to_sector(self, industry: str) -> str:
        """Map industry to broader sector categories"""
        industry_lower = industry.lower()
        
        if any(term in industry_lower for term in ['software', 'technology', 'internet', 'semiconductor', 'computer']):
            return 'Technology'
        elif any(term in industry_lower for term in ['bank', 'financial', 'insurance', 'credit']):
            return 'Financials'
        elif any(term in industry_lower for term in ['healthcare', 'pharmaceutical', 'biotech', 'medical']):
            return 'Healthcare'
        elif any(term in industry_lower for term in ['energy', 'oil', 'gas', 'renewable']):
            return 'Energy'
        elif any(term in industry_lower for term in ['retail', 'consumer', 'restaurant', 'automotive']):
            return 'Consumer Discretionary'
        elif any(term in industry_lower for term in ['utility', 'electric', 'water']):
            return 'Utilities'
        elif any(term in industry_lower for term in ['real estate', 'reit']):
            return 'Real Estate'
        elif any(term in industry_lower for term in ['material', 'mining', 'chemical', 'steel']):
            return 'Materials'
        elif any(term in industry_lower for term in ['industrial', 'manufacturing', 'aerospace', 'defense']):
            return 'Industrials'
        elif any(term in industry_lower for term in ['telecom', 'communication']):
            return 'Communication Services'
        else:
            return 'Other'
    
    def _get_fallback_sector_data(self, symbols: List[str]) -> Dict[str, str]:
        """Fallback sector mapping for common symbols when real data is unavailable"""
        # Basic fallback mapping for common symbols
        fallback_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'NFLX': 'Communication Services',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
            'V': 'Financials', 'MA': 'Financials', 'NVDA': 'Technology', 'META': 'Communication Services'
        }
        
        return {symbol: fallback_map.get(symbol, 'Other') for symbol in symbols}

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on risk score"""
        if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_position_recommendations(self, position_weight: float, 
                                         volatility: float, liquidity_risk: float,
                                         concentration_risk: float) -> List[str]:
        """Generate recommendations for individual positions"""
        recommendations = []
        
        if concentration_risk > 0.8:
            recommendations.append("Consider reducing position size to manage concentration risk")
        
        if volatility > 0.5:  # 50% annualized volatility
            recommendations.append("High volatility detected - consider tighter stop losses")
        
        if liquidity_risk > 0.5:
            recommendations.append("Low liquidity - consider gradual position adjustments")
        
        if position_weight > self.max_position_size * 0.8:
            recommendations.append("Position approaching size limit - monitor closely")
        
        return recommendations
    
    def _generate_portfolio_recommendations(self, risk_metrics: RiskMetrics,
                                          sector_concentration: Dict[str, float],
                                          position_risks: List[PositionRisk]) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        # VaR recommendations
        if abs(risk_metrics.var_1d) > self.max_portfolio_var * 0.8:
            recommendations.append("Portfolio VaR approaching limit - consider risk reduction")
        
        # Concentration recommendations
        for sector, concentration in sector_concentration.items():
            if concentration > self.max_sector_concentration * 0.8:
                recommendations.append(f"High {sector} sector concentration - consider diversification")
        
        # Sharpe ratio recommendations
        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review strategy performance")
        
        # Drawdown recommendations
        if risk_metrics.max_drawdown < -0.15:
            recommendations.append("Significant drawdown detected - review risk management")
        
        # High-risk position recommendations
        critical_positions = [pr for pr in position_risks if pr.risk_level == RiskLevel.CRITICAL]
        if critical_positions:
            symbols = [pr.symbol for pr in critical_positions]
            recommendations.append(f"Critical risk positions detected: {', '.join(symbols)}")
        
        return recommendations
    
    def _create_default_position_risk(self, symbol: str, position_size: float) -> PositionRisk:
        """Create default position risk when data is unavailable"""
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            market_value=0.0,
            risk_level=RiskLevel.MEDIUM,
            var_contribution=0.0,
            concentration_risk=0.0,
            liquidity_risk=0.0,
            volatility_risk=0.0,
            recommendations=["Insufficient data for risk assessment"]
        )
    
    def _create_default_portfolio_risk(self) -> PortfolioRisk:
        """Create default portfolio risk when data is unavailable"""
        return PortfolioRisk(
            total_value=0.0,
            risk_metrics=RiskMetrics(),
            risk_level=RiskLevel.MEDIUM,
            position_risks=[],
            sector_concentration={},
            currency_exposure={},
            recommendations=["Insufficient data for portfolio risk assessment"],
            timestamp=datetime.now()
        )
    
    def get_risk_summary(self, portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
        """
        Get a summary of portfolio risk metrics
        
        Args:
            portfolio_risk: Portfolio risk assessment
            
        Returns:
            Dictionary with risk summary
        """
        try:
            high_risk_positions = [
                pr for pr in portfolio_risk.position_risks 
                if pr.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ]
            
            return {
                'overall_risk_level': portfolio_risk.risk_level.value,
                'total_portfolio_value': portfolio_risk.total_value,
                'daily_var': portfolio_risk.risk_metrics.var_1d,
                'var_percentage': abs(portfolio_risk.risk_metrics.var_1d) / portfolio_risk.total_value if portfolio_risk.total_value > 0 else 0,
                'max_drawdown': portfolio_risk.risk_metrics.max_drawdown,
                'sharpe_ratio': portfolio_risk.risk_metrics.sharpe_ratio,
                'volatility': portfolio_risk.risk_metrics.volatility,
                'high_risk_positions': len(high_risk_positions),
                'sector_concentrations': portfolio_risk.sector_concentration,
                'key_recommendations': portfolio_risk.recommendations[:3],  # Top 3 recommendations
                'timestamp': portfolio_risk.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk summary: {e}")
            return {}

# Convenience functions
def calculate_position_size(entry_price: float, stop_loss: float, 
                          risk_amount: float, max_position_pct: float = 0.10) -> float:
    """
    Convenience function to calculate position size
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_amount: Amount to risk
        max_position_pct: Maximum position size as percentage
        
    Returns:
        Recommended position size
    """
    manager = RiskManager()
    manager.max_position_size = max_position_pct
    return manager.calculate_position_size("", entry_price, stop_loss, risk_amount)

def assess_trade_risk(symbol: str, quantity: float, price: float, 
                     portfolio_value: float) -> Dict[str, Any]:
    """
    Convenience function to assess trade risk
    
    Args:
        symbol: Stock symbol
        quantity: Trade quantity
        price: Trade price
        portfolio_value: Portfolio value
        
    Returns:
        Risk assessment results
    """
    manager = RiskManager()
    return manager.validate_trade(symbol, "buy", quantity, price, portfolio_value)

if __name__ == "__main__":
    # Test the risk manager
    manager = RiskManager()
    
    # Test position size calculation
    position_size = manager.calculate_position_size("AAPL", 150.0, 140.0, 1000.0)
    print(f"Recommended position size: {position_size}")
    
    # Test trade validation
    validation = manager.validate_trade("AAPL", 100, 150.0, 100000.0)
    print(f"Trade validation: {validation}")
    
    print("Risk Manager test completed")