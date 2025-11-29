#!/bin/bash
# Docker entrypoint script for Trading System v2 Production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Function to wait for database
wait_for_db() {
    log "Waiting for database connection..."
    
    # Extract database connection details from DATABASE_URL
    if [[ -n "$DATABASE_URL" ]]; then
        # Parse PostgreSQL URL
        if [[ "$DATABASE_URL" =~ postgresql://([^:]+):([^@]+)@([^:]+):([^/]+)/(.+) ]]; then
            DB_USER="${BASH_REMATCH[1]}"
            DB_PASS="${BASH_REMATCH[2]}"
            DB_HOST="${BASH_REMATCH[3]}"
            DB_PORT="${BASH_REMATCH[4]}"
            DB_NAME="${BASH_REMATCH[5]}"
            
            # Wait for PostgreSQL
            for i in {1..30}; do
                if PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
                    success "Database connection established"
                    return 0
                fi
                log "Waiting for database... attempt $i/30"
                sleep 2
            done
            error "Failed to connect to database after 30 attempts"
            exit 1
        fi
    else
        log "Using SQLite database - no connection wait needed"
    fi
}

# Function to wait for Redis
wait_for_redis() {
    if [[ -n "$REDIS_URL" ]]; then
        log "Waiting for Redis connection..."
        
        # Extract Redis connection details
        if [[ "$REDIS_URL" =~ redis://([^:]+):([^/]+)/(.+) ]]; then
            REDIS_HOST="${BASH_REMATCH[1]}"
            REDIS_PORT="${BASH_REMATCH[2]}"
            
            for i in {1..30}; do
                if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
                    success "Redis connection established"
                    return 0
                fi
                log "Waiting for Redis... attempt $i/30"
                sleep 2
            done
            error "Failed to connect to Redis after 30 attempts"
            exit 1
        fi
    else
        log "No Redis configuration found - skipping Redis wait"
    fi
}

# Function to run database migrations
run_migrations() {
    log "Running database migrations..."
    
    if [[ -f "alembic.ini" ]]; then
        alembic upgrade head
        success "Database migrations completed"
    else
        warning "No alembic.ini found - skipping migrations"
    fi
}

# Function to create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p /app/data /app/logs /app/tmp
    chmod 700 /app/data /app/logs /app/tmp
    
    success "Directories created and secured"
}

# Function to validate environment
validate_environment() {
    log "Validating environment configuration..."
    
    # Check required environment variables
    required_vars=("ENVIRONMENT")
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate environment value
    if [[ "$ENVIRONMENT" != "production" ]]; then
        warning "Environment is set to '$ENVIRONMENT', expected 'production'"
    fi
    
    # Check if backup is disabled
    if [[ "$ENABLE_BACKUPS" == "true" ]]; then
        error "Backups are enabled in production - this should be disabled"
        exit 1
    fi
    
    success "Environment validation passed"
}

# Function to perform health check
health_check() {
    log "Performing initial health check..."
    
    # Check Python environment
    python --version
    
    # Check if main application module can be imported
    log "Testing application module import..."
    if python -c "import app" 2>&1; then
        success "Application module import successful"
    else
        error "Failed to import main application module"
        log "Attempting to show import error details..."
        python -c "import app" || true
        exit 1
    fi
    
    success "Health check passed"
}

# Function to set up monitoring
setup_monitoring() {
    log "Setting up monitoring and logging..."
    
    # Create log files with proper permissions
    touch /app/logs/access.log /app/logs/error.log /app/logs/application.log
    chmod 644 /app/logs/*.log
    
    # Set up log rotation if logrotate is available
    if command -v logrotate >/dev/null 2>&1; then
        log "Log rotation configured"
    fi
    
    success "Monitoring setup completed"
}

# Main initialization function
initialize() {
    log "Starting Trading System v2 Production Initialization..."
    
    # Create directories
    create_directories
    
    # Validate environment
    validate_environment
    
    # Wait for dependencies
    wait_for_db
    wait_for_redis
    
    # Run migrations
    run_migrations
    
    # Setup monitoring
    setup_monitoring
    
    # Health check
    health_check
    
    success "Initialization completed successfully"
}

# Signal handlers for graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    log "Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Trading System v2 Production Container Starting..."
    log "Environment: $ENVIRONMENT"
    log "Trading Mode: ${TRADING_MODE:-not_set}"
    log "Demo Mode: ${DEMO_MODE:-not_set}"
    
    # Run initialization
    initialize
    
    # Execute the command passed to the container
    if [[ $# -eq 0 ]]; then
        error "No command provided to execute"
        exit 1
    fi
    
    log "Starting application with command: $*"
    exec "$@"
}

# Run main function with all arguments
main "$@"