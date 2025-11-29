# Multi-stage Dockerfile for Trading System

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r trading && useradd -r -g trading trading

# Set working directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements files
COPY requirements-simple.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-simple.txt

# Stage 3: Development environment
FROM dependencies as development

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to trading user
RUN chown -R trading:trading /app

# Switch to non-root user
USER trading

# Expose ports
EXPOSE 8000 8080

# Default command for development
CMD ["python", "main.py", "run", "--mode", "paper"]

# Stage 4: Production environment
FROM dependencies as production

# Copy only necessary files for production
COPY trading_system_v2/ ./trading_system_v2/
COPY main.py setup.py ./
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p logs data test_reports

# Change ownership to trading user
RUN chown -R trading:trading /app

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python main.py status || exit 1

# Expose ports
EXPOSE 8000

# Production command
CMD ["python", "main.py", "run", "--mode", "live"]

# Stage 5: Testing environment
FROM development as testing

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini conftest.py run_tests.py ./

# Run tests by default
CMD ["python", "run_tests.py", "--type", "all", "--format", "all"]