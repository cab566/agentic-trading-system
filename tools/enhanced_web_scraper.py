#!/usr/bin/env python3
"""
Enhanced Web Scraping Tool for Financial Data Discovery

This tool enables agents to discover and extract real financial data from various web sources
including SEC filings, earnings transcripts, analyst reports, and alternative data sources.
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import re
from urllib.parse import urljoin, urlparse
import time

# Web scraping imports
try:
    from bs4 import BeautifulSoup
    import requests
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logging.warning("Web scraping libraries not available. Install: pip install beautifulsoup4 selenium requests")

# CrewAI imports
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Local imports
from ..core.data_manager import UnifiedDataManager
from ..core.config_manager import ConfigManager
from ..core.session_manager import SessionManager, ResilientHttpClient


class ScrapingTarget(BaseModel):
    """Web scraping target configuration"""
    name: str = Field(description="Name of the scraping target")
    base_url: str = Field(description="Base URL for scraping")
    selectors: Dict[str, str] = Field(description="CSS selectors for data extraction")
    rate_limit: float = Field(default=1.0, description="Rate limit in seconds between requests")
    requires_js: bool = Field(default=False, description="Whether JavaScript rendering is required")
    headers: Dict[str, str] = Field(default={}, description="Custom headers for requests")


@dataclass
class ScrapedData:
    """Scraped data structure"""
    source: str
    url: str
    timestamp: datetime
    data: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]


class EnhancedWebScraperTool(BaseTool):
    """Enhanced web scraping tool for financial data discovery"""
    
    name: str = "enhanced_web_scraper"
    description: str = "Advanced web scraping tool for discovering real financial data from SEC filings, earnings transcripts, analyst reports, and alternative sources"
    
    def __init__(self, config_manager: ConfigManager, data_manager: UnifiedDataManager):
        super().__init__()
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Load scraping configuration
        self.config = self._load_config()
        
        # Initialize scraping targets
        self.targets = self._initialize_targets()
        
        # Setup HTTP clients with proper session management
        self.http_client = ResilientHttpClient(
            session_name="web_scraper",
            max_retries=self.config.get('max_retries', 3),
            timeout=self.config.get('timeout', 30)
        )
        
        # Setup synchronous session for non-async operations
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _load_config(self) -> Dict[str, Any]:
        """Load scraping configuration"""
        try:
            config = self.config_manager.get_config('web_scraper')
            return config
        except Exception as e:
            self.logger.warning(f"Could not load web scraper config: {e}")
            return {
                'rate_limit': 1.0,
                'timeout': 30,
                'max_retries': 3,
                'enable_selenium': True
            }
    
    def _initialize_targets(self) -> Dict[str, ScrapingTarget]:
        """Initialize predefined scraping targets"""
        targets = {}
        
        # SEC EDGAR filings
        targets['sec_edgar'] = ScrapingTarget(
            name="SEC EDGAR Filings",
            base_url="https://www.sec.gov/edgar/search/",
            selectors={
                'filing_table': 'table.table',
                'filing_link': 'a[href*="/Archives/edgar/data/"]',
                'filing_date': 'td:nth-child(4)',
                'filing_type': 'td:nth-child(1)'
            },
            rate_limit=1.0,
            requires_js=True
        )
        
        # Yahoo Finance earnings transcripts
        targets['yahoo_earnings'] = ScrapingTarget(
            name="Yahoo Finance Earnings",
            base_url="https://finance.yahoo.com/quote/{symbol}/analysis",
            selectors={
                'earnings_date': '[data-test="EARNINGS_DATE-value"]',
                'revenue_estimate': '[data-test="REVENUE_ESTIMATE-value"]',
                'eps_estimate': '[data-test="EPS_ESTIMATE-value"]',
                'analyst_recommendations': '[data-test="recommendation-rating"]'
            },
            rate_limit=2.0
        )
        
        # Seeking Alpha articles
        targets['seeking_alpha'] = ScrapingTarget(
            name="Seeking Alpha Analysis",
            base_url="https://seekingalpha.com/symbol/{symbol}/analysis",
            selectors={
                'article_title': 'h3[data-test-id="post-list-item-title"] a',
                'article_summary': '[data-test-id="post-list-item-summary"]',
                'author': '[data-test-id="post-list-author-name"]',
                'publish_date': 'time'
            },
            rate_limit=3.0
        )
        
        # Finviz screener
        targets['finviz_screener'] = ScrapingTarget(
            name="Finviz Stock Screener",
            base_url="https://finviz.com/screener.ashx",
            selectors={
                'stock_table': 'table.table-light',
                'ticker': 'td:nth-child(2) a',
                'price': 'td:nth-child(12)',
                'volume': 'td:nth-child(13)',
                'market_cap': 'td:nth-child(7)'
            },
            rate_limit=2.0
        )
        
        # Reddit WallStreetBets sentiment
        targets['reddit_wsb'] = ScrapingTarget(
            name="Reddit WallStreetBets",
            base_url="https://www.reddit.com/r/wallstreetbets/hot.json",
            selectors={},  # JSON API
            rate_limit=5.0
        )
        
        # Insider trading data
        targets['insider_trading'] = ScrapingTarget(
            name="Insider Trading Data",
            base_url="https://www.insidermonkey.com/insider-trading/purchases/",
            selectors={
                'insider_table': 'table.insider-trading-table',
                'company': 'td:nth-child(1)',
                'insider': 'td:nth-child(2)',
                'transaction': 'td:nth-child(3)',
                'date': 'td:nth-child(4)'
            },
            rate_limit=2.0
        )
        
        return targets
    
    def _run(self, target: str, symbol: str = None, parameters: Dict[str, Any] = None) -> str:
        """Synchronous scraping execution"""
        return asyncio.run(self._arun(target, symbol, parameters))
    
    async def _arun(self, target: str, symbol: str = None, parameters: Dict[str, Any] = None) -> str:
        """Asynchronous scraping execution"""
        try:
            if target not in self.targets:
                available_targets = list(self.targets.keys())
                return json.dumps({
                    'error': f'Unknown target: {target}',
                    'available_targets': available_targets
                })
            
            scraping_target = self.targets[target]
            
            # Execute scraping based on target type
            if target == 'sec_edgar':
                data = await self._scrape_sec_filings(symbol, parameters or {})
            elif target == 'yahoo_earnings':
                data = await self._scrape_yahoo_earnings(symbol)
            elif target == 'seeking_alpha':
                data = await self._scrape_seeking_alpha(symbol)
            elif target == 'finviz_screener':
                data = await self._scrape_finviz_screener(parameters or {})
            elif target == 'reddit_wsb':
                data = await self._scrape_reddit_sentiment(symbol)
            elif target == 'insider_trading':
                data = await self._scrape_insider_trading(symbol)
            else:
                data = await self._generic_scrape(scraping_target, symbol, parameters or {})
            
            return json.dumps({
                'success': True,
                'target': target,
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }, indent=2)
            
        except Exception as e:
            self.logger.error(f"Scraping failed for {target}: {e}")
            return json.dumps({
                'error': str(e),
                'target': target,
                'symbol': symbol
            })
    
    async def _scrape_sec_filings(self, symbol: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape SEC EDGAR filings for a symbol"""
        filings = []
        
        try:
            # Search for recent filings
            search_url = f"https://www.sec.gov/edgar/search/#/q={symbol}&dateRange=1y"
            
            if WEB_SCRAPING_AVAILABLE:
                # Use Selenium for JavaScript-heavy SEC site
                options = Options()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                driver = webdriver.Chrome(options=options)
                driver.get(search_url)
                
                # Wait for results to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'table'))
                )
                
                # Extract filing data
                rows = driver.find_elements(By.CSS_SELECTOR, 'tr')
                for row in rows[:10]:  # Limit to recent 10 filings
                    cells = row.find_elements(By.TAG_NAME, 'td')
                    if len(cells) >= 4:
                        filing = {
                            'form_type': cells[0].text.strip(),
                            'description': cells[1].text.strip(),
                            'filing_date': cells[3].text.strip(),
                            'link': cells[1].find_element(By.TAG_NAME, 'a').get_attribute('href') if cells[1].find_elements(By.TAG_NAME, 'a') else None
                        }
                        filings.append(filing)
                
                driver.quit()
            
        except Exception as e:
            self.logger.error(f"SEC filing scraping failed: {e}")
        
        return filings
    
    async def _scrape_yahoo_earnings(self, symbol: str) -> Dict[str, Any]:
        """Scrape Yahoo Finance earnings data"""
        earnings_data = {}
        
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/analysis"
            
            # Use managed session instead of creating new ClientSession
            async with self.http_client.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract earnings estimates
                earnings_table = soup.find('table', {'class': 'W(100%)'})
                if earnings_table:
                    rows = earnings_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            metric = cells[0].text.strip()
                            value = cells[1].text.strip()
                            earnings_data[metric] = value
            
            await asyncio.sleep(self.targets['yahoo_earnings'].rate_limit)
            
        except Exception as e:
            self.logger.error(f"Yahoo earnings scraping failed: {e}")
        
        return earnings_data
    
    async def _scrape_seeking_alpha(self, symbol: str) -> List[Dict[str, Any]]:
        """Scrape Seeking Alpha analysis articles"""
        articles = []
        
        try:
            url = f"https://seekingalpha.com/symbol/{symbol}/analysis"
            
            # Use managed session instead of creating new ClientSession
            async with self.http_client.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract article information
                article_elements = soup.find_all('article')[:5]  # Limit to 5 recent articles
                
                for article in article_elements:
                    title_elem = article.find('h3')
                    summary_elem = article.find('p')
                    
                    if title_elem and summary_elem:
                        articles.append({
                            'title': title_elem.text.strip(),
                            'summary': summary_elem.text.strip()[:200] + '...',
                            'url': urljoin(url, title_elem.find('a')['href']) if title_elem.find('a') else None
                        })
            
            await asyncio.sleep(self.targets['seeking_alpha'].rate_limit)
            
        except Exception as e:
            self.logger.error(f"Seeking Alpha scraping failed: {e}")
        
        return articles
    
    async def _scrape_finviz_screener(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape Finviz stock screener results"""
        stocks = []
        
        try:
            # Build screener URL with criteria
            base_url = "https://finviz.com/screener.ashx"
            params = []
            
            # Add screening criteria
            if 'market_cap' in criteria:
                params.append(f"f=cap_{criteria['market_cap']}")
            if 'volume' in criteria:
                params.append(f"f=sh_avgvol_{criteria['volume']}")
            if 'price' in criteria:
                params.append(f"f=sh_price_{criteria['price']}")
            
            url = f"{base_url}?{'&'.join(params)}" if params else base_url
            
            # Use managed session instead of creating new ClientSession
            async with self.http_client.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract stock data from screener table
                table = soup.find('table', {'class': 'table-light'})
                if table:
                    rows = table.find_all('tr')[1:]  # Skip header
                    
                    for row in rows[:20]:  # Limit to top 20 results
                        cells = row.find_all('td')
                        if len(cells) >= 12:
                            stock = {
                                'ticker': cells[1].text.strip(),
                                'company': cells[2].text.strip(),
                                'sector': cells[3].text.strip(),
                                'price': cells[8].text.strip(),
                                'volume': cells[9].text.strip(),
                                'market_cap': cells[6].text.strip()
                            }
                            stocks.append(stock)
            
            await asyncio.sleep(self.targets['finviz_screener'].rate_limit)
            
        except Exception as e:
            self.logger.error(f"Finviz screener scraping failed: {e}")
        
        return stocks
    
    async def _scrape_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Scrape Reddit WallStreetBets sentiment for symbol"""
        sentiment_data = {
            'mentions': 0,
            'sentiment_score': 0.0,
            'posts': []
        }
        
        try:
            # Search for symbol mentions in WSB
            url = f"https://www.reddit.com/r/wallstreetbets/search.json?q={symbol}&restrict_sr=1&sort=new&limit=25"
            
            # Use managed session instead of creating new ClientSession
            headers = {'User-Agent': 'TradingBot/1.0'}
            async with self.http_client.get(url, headers=headers) as response:
                data = await response.json()
                
                posts = data.get('data', {}).get('children', [])
                sentiment_scores = []
                
                for post in posts:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    
                    # Simple sentiment analysis (can be enhanced with NLP)
                    text = f"{title} {selftext}".lower()
                    positive_words = ['buy', 'bull', 'moon', 'rocket', 'calls', 'long']
                    negative_words = ['sell', 'bear', 'crash', 'puts', 'short', 'dump']
                    
                    pos_count = sum(1 for word in positive_words if word in text)
                    neg_count = sum(1 for word in negative_words if word in text)
                    
                    if pos_count + neg_count > 0:
                        sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                        sentiment_scores.append(sentiment)
                        
                        sentiment_data['posts'].append({
                            'title': title,
                            'score': post_data.get('score', 0),
                            'sentiment': sentiment,
                            'url': f"https://reddit.com{post_data.get('permalink', '')}"
                        })
                    
                    sentiment_data['mentions'] = len(posts)
                    sentiment_data['sentiment_score'] = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            await asyncio.sleep(self.targets['reddit_wsb'].rate_limit)
            
        except Exception as e:
            self.logger.error(f"Reddit sentiment scraping failed: {e}")
        
        return sentiment_data
    
    async def _scrape_insider_trading(self, symbol: str) -> List[Dict[str, Any]]:
        """Scrape insider trading data"""
        insider_trades = []
        
        try:
            url = f"https://www.insidermonkey.com/insider-trading/company/{symbol}/"
            
            # Use managed session instead of creating new ClientSession
            async with self.http_client.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract insider trading table
                table = soup.find('table', {'class': 'insider-trading-table'})
                if table:
                    rows = table.find_all('tr')[1:]  # Skip header
                    
                    for row in rows[:10]:  # Limit to recent 10 trades
                        cells = row.find_all('td')
                        if len(cells) >= 6:
                            trade = {
                                'insider': cells[0].text.strip(),
                                'position': cells[1].text.strip(),
                                'transaction': cells[2].text.strip(),
                                'date': cells[3].text.strip(),
                                'shares': cells[4].text.strip(),
                                'price': cells[5].text.strip()
                            }
                            trades.append(trade)
            
            await asyncio.sleep(self.targets['insider_trading'].rate_limit)
            
        except Exception as e:
            self.logger.error(f"Insider trading scraping failed: {e}")
        
        return insider_trades
    
    async def _generic_scrape(self, target: ScrapingTarget, symbol: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic scraping method for custom targets"""
        scraped_data = {}
        
        try:
            url = target.base_url.format(symbol=symbol) if symbol else target.base_url
            
            # Use managed session instead of creating new ClientSession
            async with self.http_client.get(url, headers=target.headers) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract data using configured selectors
                for key, selector in target.selectors.items():
                    elements = soup.select(selector)
                    scraped_data[key] = [elem.text.strip() for elem in elements]
            
            await asyncio.sleep(target.rate_limit)
            
        except Exception as e:
            self.logger.error(f"Generic scraping failed for {target.name}: {e}")
        
        return scraped_data


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from ..core.config_manager import ConfigManager
    from ..core.data_manager import UnifiedDataManager
    
    config_manager = ConfigManager(Path("../config"))
    data_manager = UnifiedDataManager(config_manager)
    
    scraper = EnhancedWebScraperTool(config_manager, data_manager)
    
    # Test scraping
    result = scraper._run('yahoo_earnings', 'AAPL')
    print("Yahoo Earnings Data:", result)
    
    result = scraper._run('reddit_wsb', 'TSLA')
    print("Reddit Sentiment:", result)