#!/usr/bin/env python3
"""
Research Tool for CrewAI Trading System

Provides agents with comprehensive market research capabilities
including fundamental analysis, sector analysis, and economic data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.data_manager import UnifiedDataManager, DataRequest


class ResearchType(Enum):
    """Types of research analysis."""
    FUNDAMENTAL = "fundamental"
    SECTOR = "sector"
    ECONOMIC = "economic"
    PEER_COMPARISON = "peer_comparison"
    VALUATION = "valuation"
    EARNINGS = "earnings"
    ANALYST_CONSENSUS = "analyst_consensus"
    ESG = "esg"
    INSIDER_TRADING = "insider_trading"
    INSTITUTIONAL = "institutional"


class AnalysisDepth(Enum):
    """Depth of analysis."""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP_DIVE = "deep_dive"


@dataclass
class CompanyProfile:
    """Company profile data structure."""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    employees: int = 0
    description: str = ""
    website: str = ""
    headquarters: str = ""
    founded: Optional[int] = None
    ceo: str = ""
    exchange: str = ""
    currency: str = "USD"
    country: str = ""
    ipo_date: Optional[datetime] = None
    fiscal_year_end: str = ""
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class FinancialMetrics:
    """Financial metrics data structure."""
    symbol: str
    period: str  # 'annual', 'quarterly', 'ttm'
    revenue: float = 0.0
    gross_profit: float = 0.0
    operating_income: float = 0.0
    net_income: float = 0.0
    total_assets: float = 0.0
    total_debt: float = 0.0
    shareholders_equity: float = 0.0
    free_cash_flow: float = 0.0
    
    # Per-share metrics
    eps: float = 0.0
    book_value_per_share: float = 0.0
    revenue_per_share: float = 0.0
    
    # Ratios
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    roe: float = 0.0  # Return on Equity
    roa: float = 0.0  # Return on Assets
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0
    
    # Growth rates (YoY %)
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    
    report_date: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)


class ResearchInput(BaseModel):
    """Input schema for research requests."""
    action: str = Field(
        ...,
        description="Research action: 'analyze', 'compare', 'screen', 'sector', 'economic', 'valuation', 'earnings'"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Primary symbol to research"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="List of symbols for comparison or screening"
    )
    research_type: Optional[str] = Field(
        default="fundamental",
        description="Type of research: 'fundamental', 'sector', 'economic', 'peer_comparison', 'valuation'"
    )
    analysis_depth: Optional[str] = Field(
        default="standard",
        description="Analysis depth: 'quick', 'standard', 'comprehensive', 'deep_dive'"
    )
    sector: Optional[str] = Field(
        default=None,
        description="Sector for sector analysis or screening"
    )
    industry: Optional[str] = Field(
        default=None,
        description="Industry for industry analysis"
    )
    market_cap_min: Optional[float] = Field(
        default=None,
        description="Minimum market cap for screening"
    )
    market_cap_max: Optional[float] = Field(
        default=None,
        description="Maximum market cap for screening"
    )
    pe_ratio_max: Optional[float] = Field(
        default=None,
        description="Maximum P/E ratio for screening"
    )
    dividend_yield_min: Optional[float] = Field(
        default=None,
        description="Minimum dividend yield for screening"
    )
    revenue_growth_min: Optional[float] = Field(
        default=None,
        description="Minimum revenue growth rate for screening"
    )
    debt_to_equity_max: Optional[float] = Field(
        default=None,
        description="Maximum debt-to-equity ratio for screening"
    )
    include_financials: Optional[bool] = Field(
        default=True,
        description="Include financial metrics in analysis"
    )
    include_ratios: Optional[bool] = Field(
        default=True,
        description="Include financial ratios in analysis"
    )
    include_growth: Optional[bool] = Field(
        default=True,
        description="Include growth metrics in analysis"
    )
    include_valuation: Optional[bool] = Field(
        default=True,
        description="Include valuation metrics in analysis"
    )
    periods: Optional[int] = Field(
        default=4,
        description="Number of periods for historical analysis"
    )
    benchmark: Optional[str] = Field(
        default="SPY",
        description="Benchmark for comparison"
    )
    currency: Optional[str] = Field(
        default="USD",
        description="Currency for financial data"
    )


class ResearchTool(BaseTool):
    """
    Research Tool for CrewAI agents.
    
    Provides comprehensive market research including:
    - Fundamental analysis of companies
    - Sector and industry analysis
    - Peer comparison and screening
    - Valuation analysis
    - Economic data analysis
    - Earnings analysis and forecasts
    - ESG and sustainability metrics
    """
    
    name: str = "research_tool"
    description: str = (
        "Conduct comprehensive market research and analysis. Provides fundamental "
        "analysis, sector research, peer comparisons, valuation analysis, and "
        "economic data for informed investment decisions."
    )
    args_schema: type[ResearchInput] = ResearchInput
    data_manager: UnifiedDataManager = Field(default=None, exclude=True)
    logger: Any = Field(default=None, exclude=True)
    company_profiles: Dict = Field(default_factory=dict, exclude=True)
    financial_metrics: Dict = Field(default_factory=dict, exclude=True)
    sector_data: Dict = Field(default_factory=dict, exclude=True)
    sector_mappings: Dict = Field(default_factory=dict, exclude=True)
    
    def __init__(self, data_manager: UnifiedDataManager, **kwargs):
        """
        Initialize the research tool.
        
        Args:
            data_manager: Unified data manager instance
        """
        super().__init__(**kwargs)
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Research cache (in production, this would be a database)
        self.company_profiles: Dict[str, CompanyProfile] = {}
        self.financial_metrics: Dict[str, List[FinancialMetrics]] = {}
        self.sector_data: Dict[str, Dict[str, Any]] = {}
        
        # Sector mappings
        self.sector_mappings = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK'],
            'consumer_discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW'],
            'consumer_staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC'],
            'industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX'],
            'materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG'],
            'utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE'],
            'real_estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'DLR']
        }
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async execution."""
        try:
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async method
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(**kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**kwargs))
                
        except Exception as e:
            self.logger.error(f"Error in research tool: {e}")
            return f"Error processing research request: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution of research analysis."""
        try:
            # Parse input
            input_data = ResearchInput(**kwargs)
            
            # Route to appropriate handler
            if input_data.action == "analyze":
                return await self._analyze_company(input_data)
            elif input_data.action == "compare":
                return await self._compare_companies(input_data)
            elif input_data.action == "screen":
                return await self._screen_stocks(input_data)
            elif input_data.action == "sector":
                return await self._analyze_sector(input_data)
            elif input_data.action == "economic":
                return await self._analyze_economic_data(input_data)
            elif input_data.action == "valuation":
                return await self._analyze_valuation(input_data)
            elif input_data.action == "earnings":
                return await self._analyze_earnings(input_data)
            else:
                return f"Error: Unknown action '{input_data.action}'"
                
        except Exception as e:
            self.logger.error(f"Error in async research: {e}")
            return f"Error processing research request: {str(e)}"
    
    async def _analyze_company(self, input_data: ResearchInput) -> str:
        """Comprehensive company analysis."""
        try:
            if not input_data.symbol:
                return "Error: No symbol provided for company analysis"
            
            symbol = input_data.symbol.upper()
            
            # Get company profile
            profile = await self._get_company_profile(symbol)
            
            # Get financial metrics
            financials = await self._get_financial_metrics(symbol, input_data.periods)
            
            if not financials:
                return f"Error: Unable to fetch financial data for {symbol}"
            
            # Generate analysis report
            result = f"Company Analysis Report: {symbol}\n"
            result += "=" * 50 + "\n\n"
            
            # Company overview
            if profile:
                result += "Company Overview:\n"
                result += f"  Name: {profile.name}\n"
                result += f"  Sector: {profile.sector}\n"
                result += f"  Industry: {profile.industry}\n"
                result += f"  Market Cap: ${profile.market_cap:,.0f}\n"
                if profile.employees > 0:
                    result += f"  Employees: {profile.employees:,}\n"
                if profile.headquarters:
                    result += f"  Headquarters: {profile.headquarters}\n"
                result += "\n"
            
            # Latest financial metrics
            latest = financials[0]  # Most recent
            result += f"Financial Snapshot ({latest.period}):\n"
            
            if input_data.include_financials:
                result += "  Revenue: ${:,.0f}\n".format(latest.revenue)
                result += "  Net Income: ${:,.0f}\n".format(latest.net_income)
                result += "  Free Cash Flow: ${:,.0f}\n".format(latest.free_cash_flow)
                result += "  Total Assets: ${:,.0f}\n".format(latest.total_assets)
                result += "  Total Debt: ${:,.0f}\n".format(latest.total_debt)
                result += "\n"
            
            if input_data.include_ratios:
                result += "Key Ratios:\n"
                result += f"  P/E Ratio: {latest.pe_ratio:.2f}\n"
                result += f"  P/B Ratio: {latest.pb_ratio:.2f}\n"
                result += f"  P/S Ratio: {latest.ps_ratio:.2f}\n"
                result += f"  Debt/Equity: {latest.debt_to_equity:.2f}\n"
                result += f"  Current Ratio: {latest.current_ratio:.2f}\n"
                result += f"  ROE: {latest.roe:.1%}\n"
                result += f"  ROA: {latest.roa:.1%}\n"
                result += "\n"
            
            if input_data.include_growth and len(financials) > 1:
                result += "Growth Analysis:\n"
                result += f"  Revenue Growth: {latest.revenue_growth:.1%}\n"
                result += f"  Earnings Growth: {latest.earnings_growth:.1%}\n"
                
                # Calculate multi-period growth
                if len(financials) >= 4:
                    revenue_cagr = self._calculate_cagr(
                        [f.revenue for f in reversed(financials[:4])]
                    )
                    earnings_cagr = self._calculate_cagr(
                        [f.net_income for f in reversed(financials[:4]) if f.net_income > 0]
                    )
                    result += f"  Revenue CAGR (3Y): {revenue_cagr:.1%}\n"
                    if earnings_cagr is not None:
                        result += f"  Earnings CAGR (3Y): {earnings_cagr:.1%}\n"
                result += "\n"
            
            # Profitability analysis
            result += "Profitability Analysis:\n"
            result += f"  Gross Margin: {latest.gross_margin:.1%}\n"
            result += f"  Operating Margin: {latest.operating_margin:.1%}\n"
            result += f"  Net Margin: {latest.net_margin:.1%}\n"
            result += "\n"
            
            # Financial health assessment
            result += "Financial Health Assessment:\n"
            
            # Debt analysis
            if latest.debt_to_equity < 0.3:
                result += "✅ Low debt levels - strong balance sheet\n"
            elif latest.debt_to_equity < 0.6:
                result += "⚠️ Moderate debt levels - manageable\n"
            else:
                result += "❌ High debt levels - potential concern\n"
            
            # Liquidity analysis
            if latest.current_ratio > 2.0:
                result += "✅ Strong liquidity position\n"
            elif latest.current_ratio > 1.0:
                result += "⚠️ Adequate liquidity\n"
            else:
                result += "❌ Potential liquidity concerns\n"
            
            # Profitability analysis
            if latest.roe > 0.15:
                result += "✅ Strong return on equity\n"
            elif latest.roe > 0.10:
                result += "⚠️ Moderate return on equity\n"
            else:
                result += "❌ Low return on equity\n"
            
            # Valuation assessment
            if input_data.include_valuation:
                result += "\nValuation Assessment:\n"
                
                # P/E analysis (sector-relative would be better)
                if latest.pe_ratio < 15:
                    result += "✅ Potentially undervalued (low P/E)\n"
                elif latest.pe_ratio < 25:
                    result += "⚠️ Fairly valued\n"
                else:
                    result += "❌ Potentially overvalued (high P/E)\n"
                
                # P/B analysis
                if latest.pb_ratio < 1.0:
                    result += "✅ Trading below book value\n"
                elif latest.pb_ratio < 3.0:
                    result += "⚠️ Reasonable price-to-book ratio\n"
                else:
                    result += "❌ High price-to-book ratio\n"
            
            # Historical trend analysis
            if len(financials) > 1 and input_data.analysis_depth in ['comprehensive', 'deep_dive']:
                result += "\nHistorical Trends:\n"
                result += f"{'Period':<12} {'Revenue':<12} {'Net Income':<12} {'ROE':<8} {'Debt/Eq':<8}\n"
                result += "-" * 60 + "\n"
                
                for financial in financials[:min(4, len(financials))]:
                    result += f"{financial.period:<12} "
                    result += f"${financial.revenue/1e9:.1f}B{'':<4} "
                    result += f"${financial.net_income/1e9:.1f}B{'':<4} "
                    result += f"{financial.roe:.1%}{'':<3} "
                    result += f"{financial.debt_to_equity:.2f}\n"
            
            # Investment thesis
            result += "\nInvestment Considerations:\n"
            
            strengths = []
            concerns = []
            
            # Analyze strengths and concerns
            if latest.revenue_growth > 0.10:
                strengths.append("Strong revenue growth")
            elif latest.revenue_growth < 0:
                concerns.append("Declining revenue")
            
            if latest.net_margin > 0.15:
                strengths.append("High profit margins")
            elif latest.net_margin < 0.05:
                concerns.append("Low profit margins")
            
            if latest.debt_to_equity < 0.3:
                strengths.append("Conservative debt levels")
            elif latest.debt_to_equity > 0.8:
                concerns.append("High debt burden")
            
            if latest.roe > 0.15:
                strengths.append("Strong return on equity")
            elif latest.roe < 0.08:
                concerns.append("Low return on equity")
            
            if strengths:
                result += "\nStrengths:\n"
                for strength in strengths:
                    result += f"• {strength}\n"
            
            if concerns:
                result += "\nConcerns:\n"
                for concern in concerns:
                    result += f"• {concern}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing company: {e}")
            return f"Error analyzing company: {str(e)}"
    
    async def _compare_companies(self, input_data: ResearchInput) -> str:
        """Compare multiple companies side by side."""
        try:
            if not input_data.symbols or len(input_data.symbols) < 2:
                return "Error: At least 2 symbols required for comparison"
            
            symbols = [s.upper() for s in input_data.symbols]
            
            # Get financial data for all companies
            all_financials = {}
            all_profiles = {}
            
            for symbol in symbols:
                profile = await self._get_company_profile(symbol)
                financials = await self._get_financial_metrics(symbol, 1)  # Latest only
                
                if profile:
                    all_profiles[symbol] = profile
                if financials:
                    all_financials[symbol] = financials[0]
            
            if not all_financials:
                return "Error: Unable to fetch financial data for comparison"
            
            # Generate comparison report
            result = f"Company Comparison Report\n"
            result += "=" * 50 + "\n\n"
            
            result += f"Comparing: {', '.join(symbols)}\n\n"
            
            # Company overview comparison
            result += "Company Overview:\n"
            result += f"{'Symbol':<8} {'Name':<25} {'Sector':<20} {'Market Cap':<15}\n"
            result += "-" * 70 + "\n"
            
            for symbol in symbols:
                profile = all_profiles.get(symbol)
                if profile:
                    result += f"{symbol:<8} {profile.name[:24]:<25} {profile.sector[:19]:<20} ${profile.market_cap/1e9:.1f}B\n"
                else:
                    result += f"{symbol:<8} {'N/A':<25} {'N/A':<20} {'N/A':<15}\n"
            result += "\n"
            
            # Financial metrics comparison
            result += "Financial Metrics Comparison:\n"
            
            metrics = [
                ('Revenue (B)', 'revenue', 1e9, '${:.1f}B'),
                ('Net Income (B)', 'net_income', 1e9, '${:.1f}B'),
                ('P/E Ratio', 'pe_ratio', 1, '{:.1f}'),
                ('P/B Ratio', 'pb_ratio', 1, '{:.2f}'),
                ('ROE', 'roe', 1, '{:.1%}'),
                ('ROA', 'roa', 1, '{:.1%}'),
                ('Debt/Equity', 'debt_to_equity', 1, '{:.2f}'),
                ('Net Margin', 'net_margin', 1, '{:.1%}'),
                ('Revenue Growth', 'revenue_growth', 1, '{:.1%}')
            ]
            
            for metric_name, attr_name, divisor, format_str in metrics:
                result += f"\n{metric_name}:\n"
                values = []
                
                for symbol in symbols:
                    financial = all_financials.get(symbol)
                    if financial and hasattr(financial, attr_name):
                        value = getattr(financial, attr_name)
                        if value is not None:
                            formatted_value = format_str.format(value / divisor)
                            values.append((symbol, value, formatted_value))
                        else:
                            values.append((symbol, 0, 'N/A'))
                    else:
                        values.append((symbol, 0, 'N/A'))
                
                # Sort by value (descending for most metrics)
                if attr_name not in ['pe_ratio', 'pb_ratio', 'debt_to_equity']:
                    values.sort(key=lambda x: x[1], reverse=True)
                else:
                    values.sort(key=lambda x: x[1] if x[1] > 0 else float('inf'))
                
                for i, (symbol, value, formatted) in enumerate(values):
                    rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                    result += f"  {rank_symbol} {symbol}: {formatted}\n"
            
            # Valuation comparison
            result += "\nValuation Summary:\n"
            
            valuation_scores = {}
            for symbol in symbols:
                financial = all_financials.get(symbol)
                if financial:
                    score = 0
                    
                    # P/E score (lower is better)
                    if 0 < financial.pe_ratio < 15:
                        score += 3
                    elif 15 <= financial.pe_ratio < 25:
                        score += 2
                    elif financial.pe_ratio >= 25:
                        score += 1
                    
                    # P/B score (lower is better)
                    if 0 < financial.pb_ratio < 1:
                        score += 3
                    elif 1 <= financial.pb_ratio < 3:
                        score += 2
                    elif financial.pb_ratio >= 3:
                        score += 1
                    
                    # ROE score (higher is better)
                    if financial.roe > 0.15:
                        score += 3
                    elif financial.roe > 0.10:
                        score += 2
                    elif financial.roe > 0.05:
                        score += 1
                    
                    # Growth score (higher is better)
                    if financial.revenue_growth > 0.15:
                        score += 3
                    elif financial.revenue_growth > 0.05:
                        score += 2
                    elif financial.revenue_growth > 0:
                        score += 1
                    
                    valuation_scores[symbol] = score
            
            # Rank by valuation score
            ranked_valuations = sorted(valuation_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (symbol, score) in enumerate(ranked_valuations):
                rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                stars = "⭐" * min(score // 2, 5)
                result += f"  {rank_symbol} {symbol}: {stars} (Score: {score}/12)\n"
            
            # Investment recommendation
            result += "\nComparative Analysis:\n"
            
            if ranked_valuations:
                best_symbol = ranked_valuations[0][0]
                best_financial = all_financials[best_symbol]
                
                result += f"• Best Overall Value: {best_symbol}\n"
                
                # Growth leader
                growth_leader = max(all_financials.items(), 
                                  key=lambda x: x[1].revenue_growth if x[1].revenue_growth else -1)
                result += f"• Growth Leader: {growth_leader[0]} ({growth_leader[1].revenue_growth:.1%} revenue growth)\n"
                
                # Profitability leader
                profit_leader = max(all_financials.items(), 
                                  key=lambda x: x[1].net_margin if x[1].net_margin else -1)
                result += f"• Profitability Leader: {profit_leader[0]} ({profit_leader[1].net_margin:.1%} net margin)\n"
                
                # Quality leader (ROE)
                quality_leader = max(all_financials.items(), 
                                   key=lambda x: x[1].roe if x[1].roe else -1)
                result += f"• Quality Leader: {quality_leader[0]} ({quality_leader[1].roe:.1%} ROE)\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error comparing companies: {e}")
            return f"Error comparing companies: {str(e)}"
    
    async def _get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """Get company profile information."""
        try:
            # Check cache first
            if symbol in self.company_profiles:
                profile = self.company_profiles[symbol]
                # Check if data is recent (within 24 hours)
                if (datetime.now() - profile.last_updated).total_seconds() < 86400:
                    return profile
            
            # Fetch company profile data
            request = DataRequest(
                symbol=symbol,
                data_type="profile"
            )
            
            response = await self.data_manager.get_data(request)
            
            if response.error or response.data is None:
                return None
            
            # Parse profile data (assuming it's in the first row)
            data = response.data.iloc[0] if not response.data.empty else None
            if data is None:
                return None
            
            profile = CompanyProfile(
                symbol=symbol,
                name=data.get('name', symbol),
                sector=data.get('sector', 'Unknown'),
                industry=data.get('industry', 'Unknown'),
                market_cap=float(data.get('market_cap', 0)),
                employees=int(data.get('employees', 0)),
                description=data.get('description', ''),
                website=data.get('website', ''),
                headquarters=data.get('headquarters', ''),
                ceo=data.get('ceo', ''),
                exchange=data.get('exchange', ''),
                country=data.get('country', '')
            )
            
            # Cache the profile
            self.company_profiles[symbol] = profile
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error getting company profile for {symbol}: {e}")
            return None
    
    async def _get_financial_metrics(self, symbol: str, periods: int = 4) -> List[FinancialMetrics]:
        """Get financial metrics for a company."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{periods}"
            if symbol in self.financial_metrics:
                cached_data = self.financial_metrics[symbol]
                if len(cached_data) >= periods:
                    # Check if data is recent (within 6 hours)
                    if (datetime.now() - cached_data[0].last_updated).total_seconds() < 21600:
                        return cached_data[:periods]
            
            # Fetch financial data
            request = DataRequest(
                symbol=symbol,
                data_type="financials",
                limit=periods
            )
            
            response = await self.data_manager.get_data(request)
            
            if response.error or response.data is None or response.data.empty:
                return []
            
            # Parse financial data
            financial_metrics = []
            
            for _, row in response.data.iterrows():
                metrics = FinancialMetrics(
                    symbol=symbol,
                    period=row.get('period', 'ttm'),
                    revenue=float(row.get('revenue', 0)),
                    gross_profit=float(row.get('gross_profit', 0)),
                    operating_income=float(row.get('operating_income', 0)),
                    net_income=float(row.get('net_income', 0)),
                    total_assets=float(row.get('total_assets', 0)),
                    total_debt=float(row.get('total_debt', 0)),
                    shareholders_equity=float(row.get('shareholders_equity', 0)),
                    free_cash_flow=float(row.get('free_cash_flow', 0)),
                    eps=float(row.get('eps', 0)),
                    book_value_per_share=float(row.get('book_value_per_share', 0)),
                    pe_ratio=float(row.get('pe_ratio', 0)),
                    pb_ratio=float(row.get('pb_ratio', 0)),
                    ps_ratio=float(row.get('ps_ratio', 0)),
                    debt_to_equity=float(row.get('debt_to_equity', 0)),
                    current_ratio=float(row.get('current_ratio', 0)),
                    quick_ratio=float(row.get('quick_ratio', 0)),
                    roe=float(row.get('roe', 0)),
                    roa=float(row.get('roa', 0)),
                    gross_margin=float(row.get('gross_margin', 0)),
                    operating_margin=float(row.get('operating_margin', 0)),
                    net_margin=float(row.get('net_margin', 0)),
                    revenue_growth=float(row.get('revenue_growth', 0)),
                    earnings_growth=float(row.get('earnings_growth', 0))
                )
                
                financial_metrics.append(metrics)
            
            # Cache the results
            self.financial_metrics[symbol] = financial_metrics
            
            return financial_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting financial metrics for {symbol}: {e}")
            return []
    
    def _calculate_cagr(self, values: List[float]) -> Optional[float]:
        """Calculate Compound Annual Growth Rate."""
        try:
            if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
                return None
            
            years = len(values) - 1
            cagr = (values[-1] / values[0]) ** (1 / years) - 1
            return cagr
            
        except Exception as e:
            self.logger.error(f"Error calculating CAGR: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    from core.config_manager import ConfigManager
    from core.data_manager import UnifiedDataManager
    
    async def test_research_tool():
        config_manager = ConfigManager(Path("../config"))
        data_manager = UnifiedDataManager(config_manager)
        
        tool = ResearchTool(data_manager)
        
        # Test company analysis
        result = tool._run(
            action="analyze",
            symbol="AAPL",
            analysis_depth="comprehensive"
        )
        
        print("Company Analysis Result:")
        print(result)
    
    # Run test
    # asyncio.run(test_research_tool())