"""Pytest configuration and fixtures for the trading system tests."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

import pandas as pd
import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set testing environment
os.environ['TESTING'] = 'true'
os.environ['LOG_LEVEL'] = 'WARNING'
os.environ['CACHE_DISABLED'] = 'true'
os.environ['RATE_LIMIT_DISABLED'] = 'true'
os.environ['MOCK_EXTERNAL_APIS'] = 'true'


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        'database': {
            'url': 'sqlite:///:memory:',
            'echo': False
        },
        'logging': {
            'level': 'WARNING',
            'format': 'simple'
        },
        'cache': {
            'enabled': False
        },
        'rate_limiting': {
            'enabled': False
        },
        'external_apis': {
            'mock': True
        },
        'trading': {
            'mode': 'paper',
            'initial_capital': 100000.0,
            'max_position_size': 0.1,
            'risk_free_rate': 0.02
        }
    }


@pytest.fixture
def mock_database():
    """Create a mock database session."""
    engine = create_engine('sqlite:///:memory:', echo=False)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def sample_stock_data():
    """Generate sample stock price data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic stock price data
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [100.0]  # Starting price
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    data.set_index('Date', inplace=True)
    return data


@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio data."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    positions = {
        'AAPL': {'quantity': 100, 'avg_price': 150.0, 'current_price': 155.0},
        'GOOGL': {'quantity': 50, 'avg_price': 2500.0, 'current_price': 2550.0},
        'MSFT': {'quantity': 75, 'avg_price': 300.0, 'current_price': 310.0},
        'TSLA': {'quantity': 25, 'avg_price': 800.0, 'current_price': 750.0},
        'AMZN': {'quantity': 30, 'avg_price': 3000.0, 'current_price': 3100.0}
    }
    
    return {
        'symbols': symbols,
        'positions': positions,
        'cash': 50000.0,
        'total_value': sum(
            pos['quantity'] * pos['current_price'] 
            for pos in positions.values()
        ) + 50000.0
    }


@pytest.fixture
def sample_news_data():
    """Generate sample news data."""
    return [
        {
            'title': 'Apple Reports Strong Q4 Earnings',
            'content': 'Apple Inc. reported better than expected earnings...',
            'source': 'Reuters',
            'published_at': datetime.now() - timedelta(hours=2),
            'symbols': ['AAPL'],
            'sentiment': 0.8,
            'relevance': 0.9
        },
        {
            'title': 'Federal Reserve Raises Interest Rates',
            'content': 'The Federal Reserve announced a 0.25% rate hike...',
            'source': 'Bloomberg',
            'published_at': datetime.now() - timedelta(hours=4),
            'symbols': [],
            'sentiment': -0.3,
            'relevance': 0.7
        },
        {
            'title': 'Tesla Announces New Model',
            'content': 'Tesla unveiled its latest electric vehicle model...',
            'source': 'TechCrunch',
            'published_at': datetime.now() - timedelta(hours=6),
            'symbols': ['TSLA'],
            'sentiment': 0.6,
            'relevance': 0.8
        }
    ]


@pytest.fixture
def mock_data_manager():
    """Create a mock UnifiedDataManager."""
    mock_manager = Mock()
    
    # Mock async methods
    mock_manager.get_historical_data = AsyncMock()
    mock_manager.get_real_time_data = AsyncMock()
    mock_manager.get_news = AsyncMock()
    mock_manager.get_fundamentals = AsyncMock()
    mock_manager.get_technical_indicators = AsyncMock()
    
    # Mock sync methods
    mock_manager.get_current_price = Mock(return_value=150.0)
    mock_manager.is_market_open = Mock(return_value=True)
    mock_manager.get_market_hours = Mock(return_value={
        'market_open': '09:30',
        'market_close': '16:00',
        'timezone': 'US/Eastern'
    })
    
    return mock_manager


@pytest.fixture
def mock_broker():
    """Create a mock broker interface."""
    mock_broker = Mock()
    
    # Mock order methods
    mock_broker.place_order = AsyncMock(return_value={
        'order_id': 'test_order_123',
        'status': 'submitted',
        'timestamp': datetime.now()
    })
    mock_broker.cancel_order = AsyncMock(return_value=True)
    mock_broker.get_order_status = AsyncMock(return_value='filled')
    mock_broker.get_positions = AsyncMock(return_value={})
    mock_broker.get_account_info = AsyncMock(return_value={
        'buying_power': 100000.0,
        'portfolio_value': 150000.0,
        'cash': 50000.0
    })
    
    return mock_broker


@pytest.fixture
def mock_crew_agent():
    """Create a mock CrewAI agent."""
    mock_agent = Mock()
    mock_agent.execute_task = AsyncMock()
    mock_agent.get_memory = Mock(return_value=[])
    mock_agent.add_memory = Mock()
    mock_agent.role = 'test_agent'
    mock_agent.goal = 'Test goal'
    mock_agent.backstory = 'Test backstory'
    
    return mock_agent


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock_llm = Mock()
    mock_llm.predict = Mock(return_value="Test LLM response")
    mock_llm.apredict = AsyncMock(return_value="Test async LLM response")
    
    return mock_llm


@pytest.fixture
def sample_technical_indicators():
    """Generate sample technical indicator data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    return pd.DataFrame({
        'SMA_20': np.random.uniform(145, 155, len(dates)),
        'SMA_50': np.random.uniform(140, 160, len(dates)),
        'EMA_12': np.random.uniform(148, 152, len(dates)),
        'EMA_26': np.random.uniform(146, 154, len(dates)),
        'RSI': np.random.uniform(30, 70, len(dates)),
        'MACD': np.random.uniform(-2, 2, len(dates)),
        'MACD_Signal': np.random.uniform(-1.5, 1.5, len(dates)),
        'BB_Upper': np.random.uniform(155, 165, len(dates)),
        'BB_Lower': np.random.uniform(135, 145, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)


@pytest.fixture
def sample_fundamental_data():
    """Generate sample fundamental data."""
    return {
        'market_cap': 2500000000000,  # $2.5T
        'pe_ratio': 25.5,
        'pb_ratio': 8.2,
        'debt_to_equity': 1.8,
        'roe': 0.28,
        'roa': 0.15,
        'revenue_growth': 0.08,
        'earnings_growth': 0.12,
        'dividend_yield': 0.005,
        'beta': 1.2,
        'shares_outstanding': 16000000000,
        'float_shares': 15800000000,
        'insider_ownership': 0.001,
        'institutional_ownership': 0.65
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    test_env_vars = {
        'TESTING': 'true',
        'LOG_LEVEL': 'WARNING',
        'DATABASE_URL': 'sqlite:///:memory:',
        'CACHE_DISABLED': 'true',
        'RATE_LIMIT_DISABLED': 'true',
        'MOCK_EXTERNAL_APIS': 'true',
        'OPENAI_API_KEY': 'test-key',
        'ANTHROPIC_API_KEY': 'test-key',
        'ALPACA_API_KEY': 'test-key',
        'ALPACA_SECRET_KEY': 'test-secret',
        'ALPHA_VANTAGE_API_KEY': 'test-key',
        'POLYGON_API_KEY': 'test-key',
        'FINNHUB_API_KEY': 'test-key'
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def mock_external_apis():
    """Mock external API calls."""
    with patch('yfinance.download') as mock_yf, \
         patch('requests.get') as mock_requests, \
         patch('websocket.WebSocketApp') as mock_ws:
        
        # Mock yfinance
        mock_yf.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        # Mock requests
        mock_response = Mock()
        mock_response.json.return_value = {'status': 'success', 'data': {}}
        mock_response.status_code = 200
        mock_requests.return_value = mock_response
        
        # Mock WebSocket
        mock_ws.return_value = Mock()
        
        yield {
            'yfinance': mock_yf,
            'requests': mock_requests,
            'websocket': mock_ws
        }


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.asyncio,  # Enable asyncio for all tests
]


# Custom assertions
def assert_valid_price(price: float):
    """Assert that a price is valid (positive number)."""
    assert isinstance(price, (int, float))
    assert price > 0
    assert not np.isnan(price)
    assert not np.isinf(price)


def assert_valid_dataframe(df: pd.DataFrame, required_columns: List[str] = None):
    """Assert that a DataFrame is valid."""
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert not df.isnull().all().all()
    
    if required_columns:
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"


def assert_valid_portfolio(portfolio: Dict[str, Any]):
    """Assert that a portfolio structure is valid."""
    required_keys = ['positions', 'cash', 'total_value']
    for key in required_keys:
        assert key in portfolio, f"Missing required key: {key}"
    
    assert isinstance(portfolio['positions'], dict)
    assert isinstance(portfolio['cash'], (int, float))
    assert isinstance(portfolio['total_value'], (int, float))
    assert portfolio['cash'] >= 0
    assert portfolio['total_value'] >= 0


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    # Cleanup code here if needed
    pass