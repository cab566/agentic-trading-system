#!/usr/bin/env python3
"""
Trade Storage System for Trading System v2

Provides database storage and retrieval for trade execution data.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .config_manager import ConfigManager

Base = declarative_base()

class TradeRecord(Base):
    """SQLAlchemy model for trade records."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    strategy = Column(String(50))
    venue = Column(String(50))
    order_id = Column(String(50))
    fill_id = Column(String(50))
    pnl = Column(Float, default=0.0)
    trade_metadata = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'value': self.value,
            'commission': self.commission,
            'fees': self.fees,
            'strategy': self.strategy,
            'venue': self.venue,
            'order_id': self.order_id,
            'fill_id': self.fill_id,
            'pnl': self.pnl,
            'metadata': self.trade_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TradeStorage:
    """Trade storage and retrieval system."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Get database configuration
        data_config = config_manager.get_data_management_config()
        db_config = data_config.get('database', {})
        
        # Initialize database
        self.engine = None
        self.SessionLocal = None
        self._init_database(db_config)
    
    def _init_database(self, db_config: Dict[str, Any]):
        """Initialize database connection and create tables."""
        db_url = None
        try:
            # Determine database URL
            if db_config.get('type') == 'duckdb':
                db_path = db_config.get('path', './data/trading_data.duckdb')
                # Ensure directory exists
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                db_url = f"duckdb:///{db_path}"
            elif db_config.get('type') == 'postgresql':
                host = db_config.get('host', 'localhost')
                port = db_config.get('port', 5432)
                database = db_config.get('database', 'trading_system')
                username = db_config.get('username', 'postgres')
                password = db_config.get('password', '')
                db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                # Default to SQLite
                db_path = db_config.get('path', './data/trading_data.db')
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                db_url = f"sqlite:///{db_path}"
            
            # Try to create engine
            self.engine = create_engine(db_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            self.logger.info(f"Trade storage initialized with database: {db_url}")
            
        except Exception as e:
            # If DuckDB fails, fallback to SQLite
            if db_config.get('type') == 'duckdb' and 'duckdb' in str(e).lower():
                self.logger.warning(f"DuckDB initialization failed: {e}. Falling back to SQLite")
                try:
                    db_path = './data/trading_data.db'
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                    db_url = f"sqlite:///{db_path}"
                    
                    self.engine = create_engine(db_url, echo=False)
                    self.SessionLocal = sessionmaker(bind=self.engine)
                    Base.metadata.create_all(self.engine)
                    
                    self.logger.info(f"Trade storage initialized with SQLite fallback: {db_url}")
                    return
                except Exception as fallback_error:
                    self.logger.error(f"SQLite fallback also failed: {fallback_error}")
                    raise fallback_error
            else:
                 self.logger.error(f"Failed to initialize trade storage database: {e}")
                 self.engine = None
                 self.SessionLocal = None
    
    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Save a trade record to the database."""
        if not self.SessionLocal:
            self.logger.warning("Database not initialized, cannot save trade")
            return False
        
        try:
            with self.SessionLocal() as session:
                trade_record = TradeRecord(
                    trade_id=trade_data.get('trade_id', f"trade_{datetime.now().timestamp()}"),
                    timestamp=trade_data.get('timestamp', datetime.now()),
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    quantity=float(trade_data['quantity']),
                    price=float(trade_data['price']),
                    value=float(trade_data.get('value', trade_data['quantity'] * trade_data['price'])),
                    commission=float(trade_data.get('commission', 0.0)),
                    fees=float(trade_data.get('fees', 0.0)),
                    strategy=trade_data.get('strategy'),
                    venue=trade_data.get('venue'),
                    order_id=trade_data.get('order_id'),
                    fill_id=trade_data.get('fill_id'),
                    pnl=float(trade_data.get('pnl', 0.0)),
                    trade_metadata=trade_data.get('metadata')
                )
                
                session.add(trade_record)
                session.commit()
                
                self.logger.info(f"Saved trade: {trade_data['symbol']} {trade_data['side']} {trade_data['quantity']}")
                return True
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error saving trade: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error saving trade: {e}")
            return False
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades from the database."""
        if not self.SessionLocal:
            self.logger.warning("Database not initialized, returning empty list")
            return []
        
        try:
            with self.SessionLocal() as session:
                trades = session.query(TradeRecord).order_by(
                    TradeRecord.timestamp.desc()
                ).limit(limit).all()
                
                return [trade.to_dict() for trade in trades]
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database error retrieving trades: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving trades: {e}")
            return []
    
    def get_trades_by_symbol(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trades for a specific symbol."""
        if not self.SessionLocal:
            return []
        
        try:
            with self.SessionLocal() as session:
                trades = session.query(TradeRecord).filter(
                    TradeRecord.symbol == symbol
                ).order_by(
                    TradeRecord.timestamp.desc()
                ).limit(limit).all()
                
                return [trade.to_dict() for trade in trades]
                
        except Exception as e:
            self.logger.error(f"Error retrieving trades for {symbol}: {e}")
            return []
    
    def get_trades_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get trades within a date range."""
        if not self.SessionLocal:
            return []
        
        try:
            with self.SessionLocal() as session:
                trades = session.query(TradeRecord).filter(
                    TradeRecord.timestamp >= start_date,
                    TradeRecord.timestamp <= end_date
                ).order_by(
                    TradeRecord.timestamp.desc()
                ).all()
                
                return [trade.to_dict() for trade in trades]
                
        except Exception as e:
            self.logger.error(f"Error retrieving trades for date range: {e}")
            return []
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get trade statistics."""
        if not self.SessionLocal:
            return {}
        
        try:
            with self.SessionLocal() as session:
                total_trades = session.query(TradeRecord).count()
                
                if total_trades == 0:
                    return {
                        'total_trades': 0,
                        'total_volume': 0.0,
                        'total_pnl': 0.0,
                        'avg_trade_size': 0.0,
                        'win_rate': 0.0
                    }
                
                # Calculate statistics
                from sqlalchemy import func
                stats = session.query(
                    func.count(TradeRecord.id).label('total_trades'),
                    func.sum(TradeRecord.value).label('total_volume'),
                    func.sum(TradeRecord.pnl).label('total_pnl'),
                    func.avg(TradeRecord.value).label('avg_trade_size')
                ).first()
                
                # Calculate win rate
                winning_trades = session.query(TradeRecord).filter(
                    TradeRecord.pnl > 0
                ).count()
                
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                return {
                    'total_trades': stats.total_trades or 0,
                    'total_volume': float(stats.total_volume or 0),
                    'total_pnl': float(stats.total_pnl or 0),
                    'avg_trade_size': float(stats.avg_trade_size or 0),
                    'win_rate': win_rate
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating trade statistics: {e}")
            return {}
    
    def is_database_connected(self) -> bool:
        """Check if database is connected and accessible."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.debug(f"Database connection test failed: {e}")
            return False
    
    def backup_data(self, backup_dir: str) -> bool:
        """Backup trade data to specified directory - DISABLED"""
        # DISABLED FOR PRODUCTION - NO BACKUPS
        return True
        # try:
        #     self.logger.info(f"Starting trade data backup to {backup_dir}")
        #     
        #     # Create backup directory
        #     backup_path = Path(backup_dir)
        #     backup_path.mkdir(parents=True, exist_ok=True)
        #     
        #     # Export to CSV
        #     trades_df = self.get_all_trades()
        #     csv_path = backup_path / f"trades_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        #     trades_df.to_csv(csv_path, index=False)
        #     self.logger.info(f"Trade data exported to {csv_path}")
        #     
        #     # If using SQLite, also backup the database file
        #     if hasattr(self, 'db_path') and self.db_path:
        #         db_file = Path(self.db_path)
        #         if db_file.exists():
        #             backup_db_path = backup_path / f"trades_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        #             shutil.copy2(db_file, backup_db_path)
        #             self.logger.info(f"Database file backed up to {backup_db_path}")
        #     
        #     return True
        #     
        # except Exception as e:
        #     self.logger.error(f"Trade data backup failed: {e}")
        #     return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get storage system status."""
        return {
            'database_connected': self.is_database_connected(),
            'engine_url': str(self.engine.url) if self.engine else None,
            'total_trades': self.get_trade_statistics().get('total_trades', 0)
        }