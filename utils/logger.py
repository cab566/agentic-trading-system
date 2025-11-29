"""
Centralized logging configuration for Trading System v2.

This module provides a unified logging interface with:
- Structured logging for better parsing
- Log levels for different environments
- File and console handlers
- Contextual information for debugging
"""

import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class TradingSystemFormatter(logging.Formatter):
    """Custom formatter for trading system logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with trading system context."""
        # Add timestamp if not present
        if not hasattr(record, 'created'):
            record.created = datetime.now().timestamp()
            
        # Standard format with module and function info
        formatted = super().format(record)
        return formatted


def setup_logger(name: str = 'trading_system', level: str = 'INFO') -> logging.Logger:
    """
    Set up a centralized logger with both file and console handlers.
    
    Args:
        name: The name of the logger
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    except Exception as e:
        print(f"Warning: Could not create log directory: {e}")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # File handler with rotation (if logs directory exists)
    file_handler = None
    if os.path.exists(log_dir):
        try:
            log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
        except Exception as e:
            print(f"Warning: Could not set up file handler: {e}")
    
    # Create formatters
    formatter = TradingSystemFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Add formatters to handlers
    console_handler.setFormatter(formatter)
    if file_handler:
        file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    if file_handler:
        logger.addHandler(file_handler)
    
    return logger


# Default logger instance
logger = setup_logger()


def get_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Get a logger instance with the specified name and level.
    
    Args:
        name: Logger name
        level: Log level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(name, level)