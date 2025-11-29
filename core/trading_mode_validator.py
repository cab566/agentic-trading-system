#!/usr/bin/env python3
"""
Trading Mode Validator

Provides safety checks and validation for trading mode configuration,
ensuring proper setup before executing live trades.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of trading mode validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    trading_mode: str
    
class TradingModeValidator:
    """Validates trading mode configuration and provides safety checks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_trading_mode(self) -> ValidationResult:
        """Comprehensive validation of trading mode configuration"""
        errors = []
        warnings = []
        
        # Get trading mode
        trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
        
        if trading_mode not in ['paper', 'live']:
            errors.append(f"Invalid TRADING_MODE: {trading_mode}. Must be 'paper' or 'live'")
            return ValidationResult(False, errors, warnings, trading_mode)
        
        # Validate based on trading mode
        if trading_mode == 'live':
            live_errors, live_warnings = self._validate_live_trading_config()
            errors.extend(live_errors)
            warnings.extend(live_warnings)
        else:
            paper_errors, paper_warnings = self._validate_paper_trading_config()
            errors.extend(paper_errors)
            warnings.extend(paper_warnings)
        
        # Common validations
        common_errors, common_warnings = self._validate_common_config()
        errors.extend(common_errors)
        warnings.extend(common_warnings)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, trading_mode)
    
    def _validate_live_trading_config(self) -> Tuple[List[str], List[str]]:
        """Validate live trading specific configuration"""
        errors = []
        warnings = []
        
        # Required environment variables for live trading
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY',
            'OPENAI_API_KEY'
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.startswith('demo-') or value.startswith('your-'):
                errors.append(f"Live trading requires valid {var}")
        
        # Check URLs are for live trading
        base_url = os.getenv('ALPACA_BASE_URL', '')
        if 'paper-api' in base_url:
            errors.append("ALPACA_BASE_URL still points to paper trading API")
        elif base_url != 'https://api.alpaca.markets':
            warnings.append(f"Unexpected ALPACA_BASE_URL: {base_url}")
        
        # Check demo mode is disabled
        demo_mode = os.getenv('DEMO_MODE', 'true').lower()
        if demo_mode == 'true':
            errors.append("DEMO_MODE must be 'false' for live trading")
        
        # Risk management checks
        self._validate_risk_settings(errors, warnings, is_live=True)
        
        # Safety features
        safety_checks = os.getenv('ENABLE_SAFETY_CHECKS', 'false').lower()
        if safety_checks != 'true':
            warnings.append("Consider enabling ENABLE_SAFETY_CHECKS for live trading")
        
        # Daily loss limit
        daily_loss_limit = os.getenv('DAILY_LOSS_LIMIT')
        if not daily_loss_limit:
            warnings.append("Consider setting DAILY_LOSS_LIMIT for live trading")
        
        return errors, warnings
    
    def _validate_paper_trading_config(self) -> Tuple[List[str], List[str]]:
        """Validate paper trading specific configuration"""
        errors = []
        warnings = []
        
        # Check URLs are for paper trading
        base_url = os.getenv('ALPACA_BASE_URL', '')
        if base_url and 'paper-api' not in base_url and 'api.alpaca.markets' in base_url:
            warnings.append("ALPACA_BASE_URL appears to be for live trading, but TRADING_MODE is paper")
        
        # Risk management checks (more lenient for paper trading)
        self._validate_risk_settings(errors, warnings, is_live=False)
        
        return errors, warnings
    
    def _validate_common_config(self) -> Tuple[List[str], List[str]]:
        """Validate common configuration requirements"""
        errors = []
        warnings = []
        
        # Check log level
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            warnings.append(f"Invalid LOG_LEVEL: {log_level}")
        
        # Check database URL
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            warnings.append("DATABASE_URL not set, using default")
        
        return errors, warnings
    
    def _validate_risk_settings(self, errors: List[str], warnings: List[str], is_live: bool):
        """Validate risk management settings"""
        try:
            # Maximum position size
            max_pos_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
            if is_live and max_pos_size > 0.1:  # 10% for live trading
                warnings.append(f"MAX_POSITION_SIZE ({max_pos_size}) is high for live trading")
            elif max_pos_size > 1.0:
                errors.append(f"MAX_POSITION_SIZE ({max_pos_size}) cannot exceed 1.0 (100%)")
        except ValueError:
            errors.append("MAX_POSITION_SIZE must be a valid number")
        
        try:
            # Risk per trade
            risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
            if is_live and risk_per_trade > 0.02:  # 2% for live trading
                warnings.append(f"RISK_PER_TRADE ({risk_per_trade}) is high for live trading")
            elif risk_per_trade > 0.1:
                errors.append(f"RISK_PER_TRADE ({risk_per_trade}) is dangerously high")
        except ValueError:
            errors.append("RISK_PER_TRADE must be a valid number")
        
        try:
            # Initial capital
            initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
            if is_live and initial_capital > 100000:
                warnings.append(f"INITIAL_CAPITAL ({initial_capital}) is high for live trading")
        except ValueError:
            errors.append("INITIAL_CAPITAL must be a valid number")
    
    def check_live_trading_readiness(self) -> Dict[str, any]:
        """Comprehensive readiness check for live trading"""
        validation = self.validate_trading_mode()
        
        readiness_checks = {
            'config_valid': validation.is_valid,
            'trading_mode': validation.trading_mode,
            'errors': validation.errors,
            'warnings': validation.warnings,
            'timestamp': datetime.now().isoformat(),
            'ready_for_live': False
        }
        
        if validation.trading_mode == 'live' and validation.is_valid:
            # Additional live trading readiness checks
            additional_checks = self._perform_live_readiness_checks()
            readiness_checks.update(additional_checks)
            readiness_checks['ready_for_live'] = all([
                validation.is_valid,
                additional_checks.get('api_connectivity', False),
                additional_checks.get('account_verified', False)
            ])
        
        return readiness_checks
    
    def _perform_live_readiness_checks(self) -> Dict[str, any]:
        """Perform additional checks for live trading readiness"""
        checks = {
            'api_connectivity': False,
            'account_verified': False,
            'sufficient_balance': False,
            'risk_limits_set': False
        }
        
        # These would be implemented with actual API calls
        # For now, we'll check configuration
        
        # Check if risk limits are properly configured
        daily_loss_limit = os.getenv('DAILY_LOSS_LIMIT')
        max_trades_per_day = os.getenv('MAX_TRADES_PER_DAY')
        min_account_balance = os.getenv('MIN_ACCOUNT_BALANCE')
        
        checks['risk_limits_set'] = all([daily_loss_limit, max_trades_per_day, min_account_balance])
        
        return checks
    
    def log_validation_results(self, validation: ValidationResult):
        """Log validation results with appropriate severity"""
        if validation.is_valid:
            self.logger.info(f"✅ Trading mode validation passed for {validation.trading_mode.upper()} trading")
        else:
            self.logger.error(f"❌ Trading mode validation failed for {validation.trading_mode.upper()} trading")
        
        for error in validation.errors:
            self.logger.error(f"ERROR: {error}")
        
        for warning in validation.warnings:
            self.logger.warning(f"WARNING: {warning}")
    
    def require_manual_confirmation(self, message: str) -> bool:
        """Require manual confirmation for critical operations"""
        if os.getenv('TRADING_MODE', 'paper').lower() == 'live':
            print(f"\n{'='*60}")
            print("🚨 LIVE TRADING CONFIRMATION REQUIRED 🚨")
            print(f"{'='*60}")
            print(f"{message}")
            print(f"{'='*60}")
            
            response = input("Type 'CONFIRM' to proceed with live trading: ")
            return response.strip().upper() == 'CONFIRM'
        
        return True  # No confirmation needed for paper trading

def validate_before_trading() -> Dict[str, any]:
    """Convenience function to validate configuration before trading"""
    validator = TradingModeValidator()
    validation = validator.validate_trading_mode()
    validator.log_validation_results(validation)
    
    # For live trading, require manual confirmation
    if validation.trading_mode == 'live':
        if not validator.require_manual_confirmation(
            "You are about to start LIVE TRADING with real money.\n"
            "Please ensure you have tested thoroughly in paper trading mode.\n"
            "Are you sure you want to proceed?"
        ):
            return {
                'valid': False,
                'errors': ['Live trading confirmation denied by user'],
                'warnings': validation.warnings,
                'mode': validation.trading_mode
            }
    
    return {
        'valid': validation.is_valid,
        'errors': validation.errors,
        'warnings': validation.warnings,
        'mode': validation.trading_mode
    }

if __name__ == "__main__":
    # Test the validator
    validator = TradingModeValidator()
    validation = validator.validate_trading_mode()
    validator.log_validation_results(validation)
    
    if validation.trading_mode == 'live':
        readiness = validator.check_live_trading_readiness()
        print("\nLive Trading Readiness Check:")
        for key, value in readiness.items():
            print(f"  {key}: {value}")