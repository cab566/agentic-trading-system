#!/usr/bin/env python3
"""
News Analysis Tool for CrewAI Trading System

Provides agents with news sentiment analysis, event detection,
and market impact assessment capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.data_manager import UnifiedDataManager, DataRequest


class SentimentScore(Enum):
    """Sentiment score categories."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class NewsCategory(Enum):
    """News categories for classification."""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    MANAGEMENT_CHANGE = "management_change"
    MARKET_OUTLOOK = "market_outlook"
    ECONOMIC_DATA = "economic_data"
    ANALYST_RATING = "analyst_rating"
    PARTNERSHIP = "partnership"
    LEGAL = "legal"
    OTHER = "other"


@dataclass
class NewsArticle:
    """News article data structure."""
    article_id: str
    title: str
    content: str
    source: str
    published_at: datetime
    symbols: List[str]
    url: Optional[str] = None
    author: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentScore] = None
    category: Optional[NewsCategory] = None
    relevance_score: Optional[float] = None
    market_impact_score: Optional[float] = None
    keywords: List[str] = None
    entities: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = []
        if self.metadata is None:
            self.metadata = {}


class NewsAnalysisInput(BaseModel):
    """Input schema for news analysis requests."""
    action: str = Field(
        ...,
        description="Action to perform: 'analyze', 'sentiment', 'events', 'impact', 'search', 'summary'"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Trading symbol to analyze news for (e.g., 'AAPL', 'MSFT')"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="List of symbols to analyze news for"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query for news articles"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Start date for news search (ISO format)"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="End date for news search (ISO format)"
    )
    limit: Optional[int] = Field(
        default=50,
        description="Maximum number of articles to analyze"
    )
    min_relevance: Optional[float] = Field(
        default=0.5,
        description="Minimum relevance score (0.0 to 1.0)"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter by news categories"
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Filter by news sources"
    )
    sentiment_filter: Optional[str] = Field(
        default=None,
        description="Filter by sentiment: 'positive', 'negative', 'neutral'"
    )
    include_content: Optional[bool] = Field(
        default=False,
        description="Include full article content in results"
    )
    language: Optional[str] = Field(
        default="en",
        description="Language filter for articles"
    )


class NewsAnalysisTool(BaseTool):
    """
    News Analysis Tool for CrewAI agents.
    
    Provides comprehensive news analysis including:
    - Sentiment analysis of news articles
    - Event detection and classification
    - Market impact assessment
    - News search and filtering
    - Trend analysis and summarization
    - Real-time news monitoring
    """
    
    name: str = "news_analysis_tool"
    description: str = (
        "Analyze news articles for sentiment, market impact, and trading signals. "
        "Provides news search, sentiment analysis, event detection, and market "
        "impact assessment for informed trading decisions."
    )
    args_schema: type[NewsAnalysisInput] = NewsAnalysisInput
    data_manager: UnifiedDataManager = Field(default=None, exclude=True)
    logger: Any = Field(default=None, exclude=True)
    news_cache: Dict = Field(default_factory=dict, exclude=True)
    sentiment_cache: Dict = Field(default_factory=dict, exclude=True)
    positive_keywords: set = Field(default_factory=set, exclude=True)
    negative_keywords: set = Field(default_factory=set, exclude=True)
    high_impact_keywords: set = Field(default_factory=set, exclude=True)
    
    def __init__(self, data_manager: UnifiedDataManager, **kwargs):
        """
        Initialize the news analysis tool.
        
        Args:
            data_manager: Unified data manager instance
        """
        super().__init__(**kwargs)
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # News cache (in production, this would be a database)
        self.news_cache: Dict[str, NewsArticle] = {}
        self.sentiment_cache: Dict[str, float] = {}
        
        # Sentiment keywords (simplified - in production, use ML models)
        self.positive_keywords = {
            'earnings beat', 'revenue growth', 'profit increase', 'strong performance',
            'bullish', 'upgrade', 'buy rating', 'outperform', 'positive outlook',
            'expansion', 'partnership', 'acquisition', 'innovation', 'breakthrough',
            'record high', 'exceeds expectations', 'strong demand', 'market leader'
        }
        
        self.negative_keywords = {
            'earnings miss', 'revenue decline', 'loss', 'weak performance',
            'bearish', 'downgrade', 'sell rating', 'underperform', 'negative outlook',
            'layoffs', 'bankruptcy', 'lawsuit', 'investigation', 'scandal',
            'record low', 'below expectations', 'weak demand', 'market share loss'
        }
        
        # Market impact keywords
        self.high_impact_keywords = {
            'earnings', 'merger', 'acquisition', 'fda approval', 'regulatory approval',
            'ceo change', 'dividend', 'stock split', 'guidance', 'forecast',
            'bankruptcy', 'delisting', 'investigation', 'lawsuit settlement'
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
            self.logger.error(f"Error in news analysis tool: {e}")
            return f"Error processing news analysis request: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution of news analysis."""
        try:
            # Parse input
            input_data = NewsAnalysisInput(**kwargs)
            
            # Route to appropriate handler
            if input_data.action == "analyze":
                return await self._analyze_news(input_data)
            elif input_data.action == "sentiment":
                return await self._analyze_sentiment(input_data)
            elif input_data.action == "events":
                return await self._detect_events(input_data)
            elif input_data.action == "impact":
                return await self._assess_market_impact(input_data)
            elif input_data.action == "search":
                return await self._search_news(input_data)
            elif input_data.action == "summary":
                return await self._generate_summary(input_data)
            else:
                return f"Error: Unknown action '{input_data.action}'"
                
        except Exception as e:
            self.logger.error(f"Error in async news analysis: {e}")
            return f"Error processing news analysis request: {str(e)}"
    
    async def _analyze_news(self, input_data: NewsAnalysisInput) -> str:
        """Comprehensive news analysis for a symbol or query."""
        try:
            # Get news articles
            articles = await self._fetch_news(input_data)
            
            if not articles:
                return "No news articles found for the specified criteria"
            
            # Analyze each article
            analyzed_articles = []
            for article in articles:
                analyzed_article = await self._analyze_article(article)
                if analyzed_article.relevance_score >= input_data.min_relevance:
                    analyzed_articles.append(analyzed_article)
            
            if not analyzed_articles:
                return f"No relevant articles found (minimum relevance: {input_data.min_relevance})"
            
            # Generate comprehensive analysis
            result = f"News Analysis Report\n"
            result += f"Analysis Period: {input_data.date_from or 'Last 7 days'} to {input_data.date_to or 'Now'}\n"
            result += f"Total Articles Analyzed: {len(analyzed_articles)}\n\n"
            
            # Sentiment distribution
            sentiment_counts = {}
            total_sentiment = 0
            for article in analyzed_articles:
                if article.sentiment_label:
                    label = article.sentiment_label.name
                    sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
                    total_sentiment += article.sentiment_score or 0
            
            avg_sentiment = total_sentiment / len(analyzed_articles) if analyzed_articles else 0
            
            result += "Sentiment Analysis:\n"
            result += f"  Average Sentiment Score: {avg_sentiment:.2f}\n"
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(analyzed_articles)) * 100
                result += f"  {sentiment}: {count} articles ({percentage:.1f}%)\n"
            result += "\n"
            
            # Category distribution
            category_counts = {}
            for article in analyzed_articles:
                if article.category:
                    cat = article.category.value
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            if category_counts:
                result += "News Categories:\n"
                for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(analyzed_articles)) * 100
                    result += f"  {category.replace('_', ' ').title()}: {count} articles ({percentage:.1f}%)\n"
                result += "\n"
            
            # Top articles by relevance
            top_articles = sorted(analyzed_articles, key=lambda x: x.relevance_score or 0, reverse=True)[:5]
            result += "Top 5 Most Relevant Articles:\n"
            for i, article in enumerate(top_articles, 1):
                result += f"{i}. {article.title}\n"
                result += f"   Source: {article.source} | Published: {article.published_at.strftime('%Y-%m-%d %H:%M')}\n"
                result += f"   Sentiment: {article.sentiment_score:.2f} | Relevance: {article.relevance_score:.2f}\n"
                if article.market_impact_score:
                    result += f"   Market Impact: {article.market_impact_score:.2f}\n"
                result += "\n"
            
            # Market impact assessment
            high_impact_articles = [a for a in analyzed_articles if (a.market_impact_score or 0) > 0.7]
            if high_impact_articles:
                result += f"High Impact Articles ({len(high_impact_articles)}):\n"
                for article in high_impact_articles[:3]:
                    result += f"• {article.title}\n"
                    result += f"  Impact Score: {article.market_impact_score:.2f} | Sentiment: {article.sentiment_score:.2f}\n"
                result += "\n"
            
            # Trading implications
            result += "Trading Implications:\n"
            if avg_sentiment > 0.5:
                result += "• Overall positive sentiment suggests potential upward price pressure\n"
            elif avg_sentiment < -0.5:
                result += "• Overall negative sentiment suggests potential downward price pressure\n"
            else:
                result += "• Mixed sentiment suggests sideways or volatile price action\n"
            
            if high_impact_articles:
                result += f"• {len(high_impact_articles)} high-impact events may cause significant price movements\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing news: {e}")
            return f"Error analyzing news: {str(e)}"
    
    async def _analyze_sentiment(self, input_data: NewsAnalysisInput) -> str:
        """Analyze sentiment of news articles."""
        try:
            articles = await self._fetch_news(input_data)
            
            if not articles:
                return "No news articles found for sentiment analysis"
            
            result = f"Sentiment Analysis Report\n"
            result += f"Articles Analyzed: {len(articles)}\n\n"
            
            sentiment_scores = []
            sentiment_distribution = {}
            
            for article in articles:
                sentiment_score = await self._calculate_sentiment(article)
                sentiment_label = self._get_sentiment_label(sentiment_score)
                
                sentiment_scores.append(sentiment_score)
                label_name = sentiment_label.name
                sentiment_distribution[label_name] = sentiment_distribution.get(label_name, 0) + 1
                
                # Cache sentiment
                self.sentiment_cache[article.article_id] = sentiment_score
            
            # Calculate statistics
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            
            result += f"Sentiment Statistics:\n"
            result += f"  Average Sentiment: {avg_sentiment:.3f}\n"
            result += f"  Standard Deviation: {sentiment_std:.3f}\n"
            result += f"  Range: {min(sentiment_scores):.3f} to {max(sentiment_scores):.3f}\n\n"
            
            result += "Sentiment Distribution:\n"
            for sentiment, count in sentiment_distribution.items():
                percentage = (count / len(articles)) * 100
                result += f"  {sentiment}: {count} articles ({percentage:.1f}%)\n"
            
            # Sentiment trend (if date range provided)
            if input_data.date_from and input_data.date_to:
                result += "\nSentiment Trend Analysis:\n"
                # Group articles by date and calculate daily sentiment
                daily_sentiment = {}
                for article in articles:
                    date_key = article.published_at.date()
                    if date_key not in daily_sentiment:
                        daily_sentiment[date_key] = []
                    daily_sentiment[date_key].append(self.sentiment_cache.get(article.article_id, 0))
                
                for date, scores in sorted(daily_sentiment.items()):
                    avg_daily = np.mean(scores)
                    result += f"  {date}: {avg_daily:.3f} ({len(scores)} articles)\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return f"Error analyzing sentiment: {str(e)}"
    
    async def _fetch_news(self, input_data: NewsAnalysisInput) -> List[NewsArticle]:
        """Fetch news articles based on input criteria."""
        try:
            articles = []
            
            # Determine symbols to search for
            symbols = []
            if input_data.symbol:
                symbols = [input_data.symbol.upper().replace('$', '')]
            elif input_data.symbols:
                symbols = [s.upper().replace('$', '') for s in input_data.symbols]
            
            # Set date range
            date_to = datetime.now()
            if input_data.date_to:
                date_to = datetime.fromisoformat(input_data.date_to.replace('Z', '+00:00'))
            
            date_from = date_to - timedelta(days=7)  # Default to last 7 days
            if input_data.date_from:
                date_from = datetime.fromisoformat(input_data.date_from.replace('Z', '+00:00'))
            
            # Fetch news from data manager
            for symbol in symbols or ['']:
                request = DataRequest(
                    symbol=symbol,
                    data_type="news",
                    start_date=date_from,
                    end_date=date_to,
                    limit=input_data.limit
                )
                
                response = await self.data_manager.get_data(request)
                
                if not response.error and response.data is not None:
                    # Convert response data to NewsArticle objects
                    for _, row in response.data.iterrows():
                        article = self._create_news_article(row, symbol)
                        if article:
                            articles.append(article)
            
            # Apply filters
            if input_data.sources:
                articles = [a for a in articles if a.source.lower() in [s.lower() for s in input_data.sources]]
            
            if input_data.query:
                query_lower = input_data.query.lower()
                articles = [a for a in articles if 
                           query_lower in a.title.lower() or 
                           query_lower in a.content.lower()]
            
            # Limit results
            articles = articles[:input_data.limit]
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
    
    def _create_news_article(self, row: pd.Series, symbol: str) -> Optional[NewsArticle]:
        """Create NewsArticle object from data row."""
        try:
            # Map common column names
            title = row.get('title', row.get('headline', ''))
            content = row.get('content', row.get('summary', row.get('description', '')))
            source = row.get('source', row.get('publisher', 'Unknown'))
            
            # Parse published date
            published_at = datetime.now()
            if 'published_at' in row:
                published_at = pd.to_datetime(row['published_at'])
            elif 'timestamp' in row:
                published_at = pd.to_datetime(row['timestamp'])
            elif 'date' in row:
                published_at = pd.to_datetime(row['date'])
            
            # Generate article ID
            article_id = f"{symbol}_{published_at.strftime('%Y%m%d_%H%M%S')}_{hash(title) % 10000}"
            
            article = NewsArticle(
                article_id=article_id,
                title=title,
                content=content,
                source=source,
                published_at=published_at,
                symbols=[symbol] if symbol else [],
                url=row.get('url', ''),
                author=row.get('author', '')
            )
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error creating news article: {e}")
            return None
    
    async def _analyze_article(self, article: NewsArticle) -> NewsArticle:
        """Analyze a single news article."""
        try:
            # Calculate sentiment
            article.sentiment_score = await self._calculate_sentiment(article)
            article.sentiment_label = self._get_sentiment_label(article.sentiment_score)
            
            # Classify category
            article.category = self._classify_category(article)
            
            # Calculate relevance score
            article.relevance_score = self._calculate_relevance(article)
            
            # Calculate market impact score
            article.market_impact_score = self._calculate_market_impact(article)
            
            # Extract keywords and entities
            article.keywords = self._extract_keywords(article)
            article.entities = self._extract_entities(article)
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error analyzing article {article.article_id}: {e}")
            return article
    
    async def _calculate_sentiment(self, article: NewsArticle) -> float:
        """Calculate sentiment score for an article."""
        try:
            # Check cache first
            if article.article_id in self.sentiment_cache:
                return self.sentiment_cache[article.article_id]
            
            # Simple keyword-based sentiment (in production, use ML models)
            text = f"{article.title} {article.content}".lower()
            
            positive_score = sum(1 for keyword in self.positive_keywords if keyword in text)
            negative_score = sum(1 for keyword in self.negative_keywords if keyword in text)
            
            # Normalize score between -1 and 1
            total_keywords = positive_score + negative_score
            if total_keywords == 0:
                sentiment_score = 0.0
            else:
                sentiment_score = (positive_score - negative_score) / total_keywords
            
            # Cache result
            self.sentiment_cache[article.article_id] = sentiment_score
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment: {e}")
            return 0.0
    
    def _get_sentiment_label(self, score: float) -> SentimentScore:
        """Convert sentiment score to label."""
        if score >= 0.6:
            return SentimentScore.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentScore.POSITIVE
        elif score <= -0.6:
            return SentimentScore.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentScore.NEGATIVE
        else:
            return SentimentScore.NEUTRAL
    
    def _classify_category(self, article: NewsArticle) -> NewsCategory:
        """Classify article category based on content."""
        try:
            text = f"{article.title} {article.content}".lower()
            
            # Category keywords mapping
            category_keywords = {
                NewsCategory.EARNINGS: ['earnings', 'quarterly results', 'revenue', 'profit', 'eps'],
                NewsCategory.MERGER_ACQUISITION: ['merger', 'acquisition', 'takeover', 'buyout'],
                NewsCategory.REGULATORY: ['fda', 'sec', 'regulatory', 'approval', 'investigation'],
                NewsCategory.PRODUCT_LAUNCH: ['launch', 'product', 'release', 'unveil'],
                NewsCategory.MANAGEMENT_CHANGE: ['ceo', 'cfo', 'management', 'executive', 'resign'],
                NewsCategory.ANALYST_RATING: ['upgrade', 'downgrade', 'rating', 'price target'],
                NewsCategory.PARTNERSHIP: ['partnership', 'collaboration', 'joint venture'],
                NewsCategory.LEGAL: ['lawsuit', 'legal', 'court', 'settlement']
            }
            
            # Find best matching category
            best_category = NewsCategory.OTHER
            best_score = 0
            
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > best_score:
                    best_score = score
                    best_category = category
            
            return best_category
            
        except Exception as e:
            self.logger.error(f"Error classifying category: {e}")
            return NewsCategory.OTHER
    
    def _calculate_relevance(self, article: NewsArticle) -> float:
        """Calculate relevance score for an article."""
        try:
            score = 0.5  # Base relevance
            
            # Boost for symbol mentions
            text = f"{article.title} {article.content}".lower()
            for symbol in article.symbols:
                if symbol.lower() in text:
                    score += 0.2
            
            # Boost for high-impact keywords
            impact_mentions = sum(1 for keyword in self.high_impact_keywords if keyword in text)
            score += min(impact_mentions * 0.1, 0.3)
            
            # Boost for recent articles
            hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
            if hours_old < 24:
                score += 0.1
            elif hours_old < 72:
                score += 0.05
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance: {e}")
            return 0.5
    
    def _calculate_market_impact(self, article: NewsArticle) -> float:
        """Calculate potential market impact score."""
        try:
            text = f"{article.title} {article.content}".lower()
            
            # Base impact from category
            category_impact = {
                NewsCategory.EARNINGS: 0.8,
                NewsCategory.MERGER_ACQUISITION: 0.9,
                NewsCategory.REGULATORY: 0.7,
                NewsCategory.MANAGEMENT_CHANGE: 0.6,
                NewsCategory.ANALYST_RATING: 0.5,
                NewsCategory.LEGAL: 0.6
            }
            
            impact_score = category_impact.get(article.category, 0.3)
            
            # Adjust for sentiment strength
            if article.sentiment_score:
                impact_score *= (1 + abs(article.sentiment_score) * 0.5)
            
            # Boost for high-impact keywords
            high_impact_count = sum(1 for keyword in self.high_impact_keywords if keyword in text)
            impact_score += min(high_impact_count * 0.1, 0.2)
            
            return min(impact_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {e}")
            return 0.3
    
    def _extract_keywords(self, article: NewsArticle) -> List[str]:
        """Extract key terms from article."""
        try:
            # Simple keyword extraction (in production, use NLP libraries)
            text = f"{article.title} {article.content}".lower()
            
            # Common financial keywords
            financial_keywords = {
                'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
                'merger', 'acquisition', 'ipo', 'dividend', 'split',
                'upgrade', 'downgrade', 'buy', 'sell', 'hold',
                'bullish', 'bearish', 'volatility', 'volume'
            }
            
            found_keywords = [kw for kw in financial_keywords if kw in text]
            return found_keywords[:10]  # Limit to top 10
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _extract_entities(self, article: NewsArticle) -> List[str]:
        """Extract named entities from article."""
        try:
            # Simple entity extraction (in production, use NER models)
            entities = []
            
            # Add symbols as entities
            entities.extend(article.symbols)
            
            # Add source as entity
            if article.source:
                entities.append(article.source)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    from core.config_manager import ConfigManager
    from core.data_manager import UnifiedDataManager
    
    async def test_news_analysis_tool():
        config_manager = ConfigManager(Path("../config"))
        data_manager = UnifiedDataManager(config_manager)
        
        tool = NewsAnalysisTool(data_manager)
        
        # Test news analysis
        result = tool._run(
            action="analyze",
            symbol="AAPL",
            limit=20
        )
        
        print("News Analysis Result:")
        print(result)
    
    # Run test
    # asyncio.run(test_news_analysis_tool())