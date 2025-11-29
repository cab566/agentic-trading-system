#!/usr/bin/env python3
"""
CrewAI Tools for Trading System

This module provides specialized tools for CrewAI agents to interact with
market data, execute trades, perform analysis, and manage portfolios.
"""

from .market_data_tool import MarketDataTool
from .technical_analysis_tool import TechnicalAnalysisTool
from .risk_analysis_tool import RiskAnalysisTool
from .order_management_tool import OrderManagementTool
from .news_analysis_tool import NewsAnalysisTool
from .portfolio_management_tool import PortfolioManagementTool
from .research_tool import ResearchTool

__all__ = [
    'MarketDataTool',
    'TechnicalAnalysisTool',
    'RiskAnalysisTool',
    'OrderManagementTool',
    'NewsAnalysisTool',
    'PortfolioManagementTool',
    'ResearchTool'
]

__version__ = '1.0.0'
__author__ = 'Trading System v2'
__description__ = 'CrewAI tools for automated trading system'