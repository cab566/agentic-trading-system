#!/usr/bin/env python3
"""
Cross-Asset Arbitrage Detector

This tool identifies arbitrage opportunities across different asset classes,
exchanges, and markets including stocks, ETFs, options, futures, crypto, and forex.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import math

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.config_manager import ConfigManager


class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    PRICE_ARBITRAGE = "price_arbitrage"  # Same asset, different exchanges
    ETF_ARBITRAGE = "etf_arbitrage"  # ETF vs underlying basket
    PAIRS_TRADING = "pairs_trading"  # Correlated assets divergence
    CALENDAR_SPREAD = "calendar_spread"  # Same asset, different expiries
    CURRENCY_ARBITRAGE = "currency_arbitrage"  # Cross-currency opportunities
    MERGER_ARBITRAGE = "merger_arbitrage"  # M&A spread trading
    CONVERTIBLE_ARBITRAGE = "convertible_arbitrage"  # Convertible bonds vs stock
    INDEX_ARBITRAGE = "index_arbitrage"  # Index vs components
    CRYPTO_ARBITRAGE = "crypto_arbitrage"  # Crypto across exchanges
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"  # Mean reversion opportunities


class RiskLevel(Enum):
    """Risk levels for arbitrage opportunities"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data structure"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    asset_1: str
    asset_2: str
    exchange_1: str
    exchange_2: str
    price_1: float
    price_2: float
    spread: float
    spread_percentage: float
    expected_profit: float
    risk_level: RiskLevel
    confidence: float
    timestamp: datetime
    expiry_time: Optional[datetime]
    execution_complexity: str
    capital_required: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArbitrageDetectorConfig(BaseModel):
    """Configuration for arbitrage detection"""
    min_spread_percentage: float = Field(default=0.1, description="Minimum spread percentage to consider")
    max_risk_level: str = Field(default="medium", description="Maximum risk level to include")
    min_confidence: float = Field(default=0.6, description="Minimum confidence threshold")
    max_capital_required: float = Field(default=100000, description="Maximum capital required")
    update_interval: int = Field(default=60, description="Update interval in seconds")
    exchanges: List[str] = Field(default=[], description="Exchanges to monitor")
    asset_classes: List[str] = Field(default=[], description="Asset classes to include")


class CrossAssetArbitrageDetectorTool(BaseTool):
    """Cross-asset arbitrage detector for trading opportunities"""
    
    name: str = "cross_asset_arbitrage_detector"
    description: str = "Identifies arbitrage opportunities across different asset classes, exchanges, and markets"
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        super().__init__()
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Opportunity storage
        self.opportunities: List[ArbitrageOpportunity] = []
        self.historical_spreads: Dict[str, List[float]] = {}
        
        # Exchange configurations
        self.exchanges = self._load_exchange_configs()
        
        # Asset correlations for pairs trading
        self.correlation_matrix = {}
        self.correlation_threshold = 0.7
        
        # Performance tracking
        self.detection_stats = {
            'total_scans': 0,
            'opportunities_found': 0,
            'avg_spread': 0,
            'last_scan_time': None
        }
    
    def _load_config(self) -> ArbitrageDetectorConfig:
        """Load arbitrage detector configuration"""
        try:
            config_dict = self.config_manager.get_config('arbitrage_detector')
            return ArbitrageDetectorConfig(**config_dict)
        except Exception as e:
            self.logger.warning(f"Could not load arbitrage detector config: {e}")
            return ArbitrageDetectorConfig(
                exchanges=['NYSE', 'NASDAQ', 'BATS', 'IEX'],
                asset_classes=['stocks', 'etfs', 'options', 'crypto']
            )
    
    def _load_exchange_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load exchange-specific configurations"""
        return {
            'NYSE': {'fees': 0.0005, 'latency_ms': 1, 'reliability': 0.99},
            'NASDAQ': {'fees': 0.0005, 'latency_ms': 1, 'reliability': 0.99},
            'BATS': {'fees': 0.0003, 'latency_ms': 2, 'reliability': 0.98},
            'IEX': {'fees': 0.0009, 'latency_ms': 3, 'reliability': 0.97},
            'BINANCE': {'fees': 0.001, 'latency_ms': 10, 'reliability': 0.95},
            'COINBASE': {'fees': 0.005, 'latency_ms': 15, 'reliability': 0.96},
            'KRAKEN': {'fees': 0.0026, 'latency_ms': 20, 'reliability': 0.94}
        }
    
    def _run(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Synchronous arbitrage detection execution"""
        return asyncio.run(self._arun(action, parameters))
    
    async def _arun(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Asynchronous arbitrage detection execution"""
        try:
            parameters = parameters or {}
            
            if action == 'detect_arbitrage':
                return await self._detect_arbitrage(parameters)
            elif action == 'get_opportunities':
                return await self._get_opportunities(parameters)
            elif action == 'analyze_spread_history':
                return await self._analyze_spread_history(parameters)
            elif action == 'detect_pairs_trading':
                return await self._detect_pairs_trading(parameters)
            elif action == 'detect_etf_arbitrage':
                return await self._detect_etf_arbitrage(parameters)
            elif action == 'detect_crypto_arbitrage':
                return await self._detect_crypto_arbitrage(parameters)
            elif action == 'get_stats':
                return await self._get_stats()
            else:
                return json.dumps({
                    'error': f'Unknown action: {action}',
                    'available_actions': [
                        'detect_arbitrage', 'get_opportunities', 'analyze_spread_history',
                        'detect_pairs_trading', 'detect_etf_arbitrage', 'detect_crypto_arbitrage', 'get_stats'
                    ]
                })
                
        except Exception as e:
            self.logger.error(f"Arbitrage detection failed: {e}")
            return json.dumps({'error': str(e)})
    
    async def _detect_arbitrage(self, parameters: Dict[str, Any]) -> str:
        """Detect all types of arbitrage opportunities"""
        asset_classes = parameters.get('asset_classes', self.config.asset_classes)
        exchanges = parameters.get('exchanges', self.config.exchanges)
        
        self.detection_stats['total_scans'] += 1
        self.detection_stats['last_scan_time'] = datetime.now()
        
        all_opportunities = []
        
        # Detect different types of arbitrage
        if 'stocks' in asset_classes:
            stock_opportunities = await self._detect_stock_arbitrage(exchanges)
            all_opportunities.extend(stock_opportunities)
        
        if 'etfs' in asset_classes:
            etf_opportunities = await self._detect_etf_arbitrage({})
            all_opportunities.extend(json.loads(etf_opportunities).get('opportunities', []))
        
        if 'crypto' in asset_classes:
            crypto_opportunities = await self._detect_crypto_arbitrage({})
            all_opportunities.extend(json.loads(crypto_opportunities).get('opportunities', []))
        
        if 'pairs' in asset_classes:
            pairs_opportunities = await self._detect_pairs_trading({})
            all_opportunities.extend(json.loads(pairs_opportunities).get('opportunities', []))
        
        # Filter opportunities based on configuration
        filtered_opportunities = self._filter_opportunities(all_opportunities)
        
        # Update statistics
        self.detection_stats['opportunities_found'] += len(filtered_opportunities)
        if filtered_opportunities:
            spreads = [opp.get('spread_percentage', 0) for opp in filtered_opportunities]
            self.detection_stats['avg_spread'] = np.mean(spreads)
        
        return json.dumps({
            'opportunities': filtered_opportunities,
            'total_found': len(filtered_opportunities),
            'scan_timestamp': datetime.now().isoformat(),
            'asset_classes_scanned': asset_classes,
            'exchanges_scanned': exchanges
        }, indent=2)
    
    async def _detect_stock_arbitrage(self, exchanges: List[str]) -> List[Dict[str, Any]]:
        """Detect price arbitrage opportunities in stocks"""
        opportunities = []
        
        # Sample stocks to check (in production, use a larger universe)
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        for symbol in symbols:
            try:
                # Get prices from different exchanges (simulated)
                exchange_prices = await self._get_multi_exchange_prices(symbol, exchanges)
                
                if len(exchange_prices) < 2:
                    continue
                
                # Find arbitrage opportunities
                for i, (exchange1, price1) in enumerate(exchange_prices.items()):
                    for exchange2, price2 in list(exchange_prices.items())[i+1:]:
                        if price1 != price2:
                            spread = abs(price1 - price2)
                            spread_percentage = (spread / min(price1, price2)) * 100
                            
                            if spread_percentage >= self.config.min_spread_percentage:
                                # Calculate expected profit after fees
                                fee1 = self.exchanges[exchange1]['fees'] * price1
                                fee2 = self.exchanges[exchange2]['fees'] * price2
                                expected_profit = spread - fee1 - fee2
                                
                                if expected_profit > 0:
                                    opportunity = {
                                        'opportunity_id': f"{symbol}_{exchange1}_{exchange2}_{int(datetime.now().timestamp())}",
                                        'arbitrage_type': ArbitrageType.PRICE_ARBITRAGE.value,
                                        'asset_1': symbol,
                                        'asset_2': symbol,
                                        'exchange_1': exchange1,
                                        'exchange_2': exchange2,
                                        'price_1': price1,
                                        'price_2': price2,
                                        'spread': spread,
                                        'spread_percentage': spread_percentage,
                                        'expected_profit': expected_profit,
                                        'risk_level': self._assess_risk_level(spread_percentage, exchange1, exchange2).value,
                                        'confidence': self._calculate_confidence(exchange1, exchange2, spread_percentage),
                                        'timestamp': datetime.now().isoformat(),
                                        'execution_complexity': 'Low - simultaneous buy/sell',
                                        'capital_required': max(price1, price2) * 100,  # Assume 100 shares
                                        'metadata': {
                                            'fees': {'exchange1': fee1, 'exchange2': fee2},
                                            'latency_risk': max(
                                                self.exchanges[exchange1]['latency_ms'],
                                                self.exchanges[exchange2]['latency_ms']
                                            )
                                        }
                                    }
                                    opportunities.append(opportunity)
            
            except Exception as e:
                self.logger.error(f"Failed to detect arbitrage for {symbol}: {e}")
        
        return opportunities
    
    async def _detect_etf_arbitrage(self, parameters: Dict[str, Any]) -> str:
        """Detect ETF arbitrage opportunities"""
        opportunities = []
        
        # Sample ETFs and their major holdings
        etf_holdings = {
            'SPY': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'QQQ': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'IWM': ['AMC', 'GME', 'KOSS', 'EXPR', 'BBBY']  # Simplified
        }
        
        for etf_symbol, holdings in etf_holdings.items():
            try:
                # Get ETF price
                etf_data = await self.data_manager.get_price_data(
                    etf_symbol, '1m', datetime.now() - timedelta(minutes=5), datetime.now()
                )
                
                if etf_data.data.empty:
                    continue
                
                etf_price = etf_data.data['Close'].iloc[-1]
                
                # Calculate theoretical NAV from holdings (simplified)
                nav = await self._calculate_nav(holdings)
                
                if nav > 0:
                    spread = abs(etf_price - nav)
                    spread_percentage = (spread / nav) * 100
                    
                    if spread_percentage >= self.config.min_spread_percentage:
                        opportunity = {
                            'opportunity_id': f"ETF_{etf_symbol}_{int(datetime.now().timestamp())}",
                            'arbitrage_type': ArbitrageType.ETF_ARBITRAGE.value,
                            'asset_1': etf_symbol,
                            'asset_2': 'Basket',
                            'exchange_1': 'Primary',
                            'exchange_2': 'Calculated',
                            'price_1': etf_price,
                            'price_2': nav,
                            'spread': spread,
                            'spread_percentage': spread_percentage,
                            'expected_profit': spread * 0.8,  # Account for execution costs
                            'risk_level': RiskLevel.MEDIUM.value,
                            'confidence': 0.7,
                            'timestamp': datetime.now().isoformat(),
                            'execution_complexity': 'High - requires basket trading',
                            'capital_required': etf_price * 1000,  # Assume 1000 shares
                            'metadata': {
                                'holdings': holdings,
                                'nav': nav,
                                'premium_discount': 'Premium' if etf_price > nav else 'Discount'
                            }
                        }
                        opportunities.append(opportunity)
            
            except Exception as e:
                self.logger.error(f"Failed to detect ETF arbitrage for {etf_symbol}: {e}")
        
        return json.dumps({
            'opportunities': opportunities,
            'etfs_analyzed': list(etf_holdings.keys())
        }, indent=2)
    
    async def _detect_pairs_trading(self, parameters: Dict[str, Any]) -> str:
        """Detect pairs trading opportunities"""
        opportunities = []
        
        # Sample correlated pairs
        pairs = [
            ('AAPL', 'MSFT'),
            ('JPM', 'BAC'),
            ('XOM', 'CVX'),
            ('KO', 'PEP'),
            ('WMT', 'TGT')
        ]
        
        for symbol1, symbol2 in pairs:
            try:
                # Get historical data for correlation analysis
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                data1 = await self.data_manager.get_price_data(symbol1, '1h', start_date, end_date)
                data2 = await self.data_manager.get_price_data(symbol2, '1h', start_date, end_date)
                
                if data1.data.empty or data2.data.empty:
                    continue
                
                # Calculate correlation and spread
                prices1 = data1.data['Close']
                prices2 = data2.data['Close']
                
                correlation = prices1.corr(prices2)
                
                if correlation > self.correlation_threshold:
                    # Calculate normalized spread
                    ratio = prices1 / prices2
                    mean_ratio = ratio.mean()
                    std_ratio = ratio.std()
                    current_ratio = prices1.iloc[-1] / prices2.iloc[-1]
                    
                    z_score = (current_ratio - mean_ratio) / std_ratio
                    
                    if abs(z_score) > 2:  # 2 standard deviations
                        spread_percentage = abs(z_score) * std_ratio / mean_ratio * 100
                        
                        opportunity = {
                            'opportunity_id': f"PAIRS_{symbol1}_{symbol2}_{int(datetime.now().timestamp())}",
                            'arbitrage_type': ArbitrageType.PAIRS_TRADING.value,
                            'asset_1': symbol1,
                            'asset_2': symbol2,
                            'exchange_1': 'Primary',
                            'exchange_2': 'Primary',
                            'price_1': prices1.iloc[-1],
                            'price_2': prices2.iloc[-1],
                            'spread': abs(current_ratio - mean_ratio),
                            'spread_percentage': spread_percentage,
                            'expected_profit': spread_percentage * 0.5,  # Conservative estimate
                            'risk_level': RiskLevel.MEDIUM.value,
                            'confidence': min(0.9, correlation),
                            'timestamp': datetime.now().isoformat(),
                            'execution_complexity': 'Medium - long/short pair',
                            'capital_required': (prices1.iloc[-1] + prices2.iloc[-1]) * 100,
                            'metadata': {
                                'correlation': correlation,
                                'z_score': z_score,
                                'mean_ratio': mean_ratio,
                                'std_ratio': std_ratio,
                                'signal': 'Long' if z_score < -2 else 'Short'
                            }
                        }
                        opportunities.append(opportunity)
            
            except Exception as e:
                self.logger.error(f"Failed to detect pairs trading for {symbol1}/{symbol2}: {e}")
        
        return json.dumps({
            'opportunities': opportunities,
            'pairs_analyzed': pairs
        }, indent=2)
    
    async def _detect_crypto_arbitrage(self, parameters: Dict[str, Any]) -> str:
        """Detect cryptocurrency arbitrage opportunities"""
        opportunities = []
        
        # Sample crypto pairs and exchanges
        crypto_symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        crypto_exchanges = ['BINANCE', 'COINBASE', 'KRAKEN']
        
        for symbol in crypto_symbols:
            try:
                # Get prices from different crypto exchanges (simulated)
                exchange_prices = await self._get_crypto_prices(symbol, crypto_exchanges)
                
                if len(exchange_prices) < 2:
                    continue
                
                # Find arbitrage opportunities
                for i, (exchange1, price1) in enumerate(exchange_prices.items()):
                    for exchange2, price2 in list(exchange_prices.items())[i+1:]:
                        if price1 != price2:
                            spread = abs(price1 - price2)
                            spread_percentage = (spread / min(price1, price2)) * 100
                            
                            if spread_percentage >= self.config.min_spread_percentage:
                                # Account for higher crypto fees and risks
                                fee1 = self.exchanges[exchange1]['fees'] * price1
                                fee2 = self.exchanges[exchange2]['fees'] * price2
                                withdrawal_fee = price1 * 0.001  # Estimated withdrawal fee
                                
                                expected_profit = spread - fee1 - fee2 - withdrawal_fee
                                
                                if expected_profit > 0:
                                    opportunity = {
                                        'opportunity_id': f"CRYPTO_{symbol}_{exchange1}_{exchange2}_{int(datetime.now().timestamp())}",
                                        'arbitrage_type': ArbitrageType.CRYPTO_ARBITRAGE.value,
                                        'asset_1': f"{symbol}-{exchange1}",
                                        'asset_2': f"{symbol}-{exchange2}",
                                        'exchange_1': exchange1,
                                        'exchange_2': exchange2,
                                        'price_1': price1,
                                        'price_2': price2,
                                        'spread': spread,
                                        'spread_percentage': spread_percentage,
                                        'expected_profit': expected_profit,
                                        'risk_level': RiskLevel.HIGH.value,  # Crypto is higher risk
                                        'confidence': 0.6,  # Lower confidence due to volatility
                                        'timestamp': datetime.now().isoformat(),
                                        'execution_complexity': 'High - cross-exchange transfer',
                                        'capital_required': max(price1, price2) * 10,  # Assume 10 units
                                        'metadata': {
                                            'withdrawal_fee': withdrawal_fee,
                                            'transfer_time_minutes': 15,  # Estimated transfer time
                                            'volatility_risk': 'High'
                                        }
                                    }
                                    opportunities.append(opportunity)
            
            except Exception as e:
                self.logger.error(f"Failed to detect crypto arbitrage for {symbol}: {e}")
        
        return json.dumps({
            'opportunities': opportunities,
            'crypto_symbols_analyzed': crypto_symbols,
            'exchanges_analyzed': crypto_exchanges
        }, indent=2)
    
    async def _get_opportunities(self, parameters: Dict[str, Any]) -> str:
        """Get current arbitrage opportunities"""
        limit = parameters.get('limit', 20)
        arbitrage_type = parameters.get('type')
        min_profit = parameters.get('min_profit', 0)
        
        # Filter opportunities
        filtered_opportunities = []
        for opp in self.opportunities:
            if arbitrage_type and opp.arbitrage_type.value != arbitrage_type:
                continue
            if opp.expected_profit < min_profit:
                continue
            filtered_opportunities.append(opp)
        
        # Sort by expected profit (highest first)
        filtered_opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
        filtered_opportunities = filtered_opportunities[:limit]
        
        # Convert to JSON-serializable format
        opportunities_data = []
        for opp in filtered_opportunities:
            opportunities_data.append({
                'opportunity_id': opp.opportunity_id,
                'arbitrage_type': opp.arbitrage_type.value,
                'asset_1': opp.asset_1,
                'asset_2': opp.asset_2,
                'exchange_1': opp.exchange_1,
                'exchange_2': opp.exchange_2,
                'spread_percentage': opp.spread_percentage,
                'expected_profit': opp.expected_profit,
                'risk_level': opp.risk_level.value,
                'confidence': opp.confidence,
                'timestamp': opp.timestamp.isoformat(),
                'execution_complexity': opp.execution_complexity,
                'capital_required': opp.capital_required,
                'metadata': opp.metadata
            })
        
        return json.dumps({
            'opportunities': opportunities_data,
            'total_count': len(self.opportunities),
            'filtered_count': len(filtered_opportunities)
        }, indent=2)
    
    async def _analyze_spread_history(self, parameters: Dict[str, Any]) -> str:
        """Analyze historical spread data"""
        pair = parameters.get('pair')
        if not pair:
            return json.dumps({'error': 'Pair parameter required (e.g., "AAPL_NYSE_NASDAQ")'})
        
        spreads = self.historical_spreads.get(pair, [])
        
        if not spreads:
            return json.dumps({'message': f'No historical data for {pair}'})
        
        # Calculate statistics
        stats = {
            'pair': pair,
            'data_points': len(spreads),
            'avg_spread': np.mean(spreads),
            'max_spread': np.max(spreads),
            'min_spread': np.min(spreads),
            'std_spread': np.std(spreads),
            'percentiles': {
                '25th': np.percentile(spreads, 25),
                '50th': np.percentile(spreads, 50),
                '75th': np.percentile(spreads, 75),
                '95th': np.percentile(spreads, 95)
            }
        }
        
        return json.dumps(stats, indent=2)
    
    async def _get_stats(self) -> str:
        """Get arbitrage detection statistics"""
        return json.dumps({
            'detection_stats': self.detection_stats,
            'total_opportunities': len(self.opportunities),
            'opportunities_by_type': self._get_opportunity_counts_by_type(),
            'opportunities_by_risk': self._get_opportunity_counts_by_risk(),
            'avg_expected_profit': np.mean([opp.expected_profit for opp in self.opportunities]) if self.opportunities else 0
        }, indent=2)
    
    def _filter_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter opportunities based on configuration"""
        filtered = []
        
        for opp in opportunities:
            # Check minimum spread
            if opp.get('spread_percentage', 0) < self.config.min_spread_percentage:
                continue
            
            # Check risk level
            risk_levels = ['low', 'medium', 'high', 'very_high']
            max_risk_index = risk_levels.index(self.config.max_risk_level)
            opp_risk_index = risk_levels.index(opp.get('risk_level', 'high'))
            
            if opp_risk_index > max_risk_index:
                continue
            
            # Check confidence
            if opp.get('confidence', 0) < self.config.min_confidence:
                continue
            
            # Check capital requirements
            if opp.get('capital_required', 0) > self.config.max_capital_required:
                continue
            
            filtered.append(opp)
        
        return filtered
    
    def _assess_risk_level(self, spread_percentage: float, exchange1: str, exchange2: str) -> RiskLevel:
        """Assess risk level for an arbitrage opportunity"""
        # Base risk on spread size and exchange reliability
        reliability1 = self.exchanges.get(exchange1, {}).get('reliability', 0.9)
        reliability2 = self.exchanges.get(exchange2, {}).get('reliability', 0.9)
        
        avg_reliability = (reliability1 + reliability2) / 2
        
        if spread_percentage > 2.0 or avg_reliability < 0.95:
            return RiskLevel.HIGH
        elif spread_percentage > 1.0 or avg_reliability < 0.98:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_confidence(self, exchange1: str, exchange2: str, spread_percentage: float) -> float:
        """Calculate confidence score for an arbitrage opportunity"""
        # Base confidence on exchange reliability and spread size
        reliability1 = self.exchanges.get(exchange1, {}).get('reliability', 0.9)
        reliability2 = self.exchanges.get(exchange2, {}).get('reliability', 0.9)
        
        avg_reliability = (reliability1 + reliability2) / 2
        
        # Higher spreads might indicate stale data or execution risk
        spread_factor = max(0.5, 1 - (spread_percentage - 0.5) * 0.1)
        
        return min(0.95, avg_reliability * spread_factor)
    
    def _get_opportunity_counts_by_type(self) -> Dict[str, int]:
        """Get opportunity counts by arbitrage type"""
        counts = {}
        for opp in self.opportunities:
            opp_type = opp.arbitrage_type.value
            counts[opp_type] = counts.get(opp_type, 0) + 1
        return counts
    
    def _get_opportunity_counts_by_risk(self) -> Dict[str, int]:
        """Get opportunity counts by risk level"""
        counts = {}
        for opp in self.opportunities:
            risk = opp.risk_level.value
            counts[risk] = counts.get(risk, 0) + 1
        return counts
    
    # Helper methods for data fetching (simulated - replace with real APIs)
    
    async def _get_multi_exchange_prices(self, symbol: str, exchanges: List[str]) -> Dict[str, float]:
        """Get prices from multiple exchanges (simulated)"""
        base_price = np.random.uniform(100, 300)  # Simulate base price
        prices = {}
        
        for exchange in exchanges:
            # Add small random variations to simulate price differences
            variation = np.random.uniform(-0.02, 0.02)  # ±2% variation
            prices[exchange] = base_price * (1 + variation)
        
        return prices
    
    async def _get_crypto_prices(self, symbol: str, exchanges: List[str]) -> Dict[str, float]:
        """Get crypto prices from multiple exchanges (simulated)"""
        base_prices = {'BTC': 45000, 'ETH': 3000, 'ADA': 1.2, 'DOT': 25, 'LINK': 15}
        base_price = base_prices.get(symbol, 100)
        prices = {}
        
        for exchange in exchanges:
            # Crypto has higher price variations between exchanges
            variation = np.random.uniform(-0.05, 0.05)  # ±5% variation
            prices[exchange] = base_price * (1 + variation)
        
        return prices
    
    async def _calculate_nav(self, holdings: List[str]) -> float:
        """Calculate Net Asset Value for ETF holdings (simplified)"""
        total_value = 0
        
        for symbol in holdings:
            try:
                # Get current price (simulated)
                price = np.random.uniform(50, 300)
                weight = 1 / len(holdings)  # Equal weight for simplicity
                total_value += price * weight
            except Exception as e:
                self.logger.error(f"Failed to get price for {symbol}: {e}")
        
        return total_value


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from ..core.config_manager import ConfigManager
    from ..core.data_manager import UnifiedDataManager
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    detector = CrossAssetArbitrageDetectorTool(config_manager, data_manager)
    
    # Test arbitrage detection
    result = detector._run('detect_arbitrage')
    print("Arbitrage Detection:", result)
    
    # Get opportunities
    result = detector._run('get_opportunities')
    print("Opportunities:", result)