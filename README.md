# Agentic Trading System (Full Stack)

An end-to-end agentic trading platform that orchestrates specialized AI agents for market research, technical and sentiment analysis, multi-strategy signal generation, smart execution, portfolio optimization, and continuous risk monitoring. Built for equities and forex (crypto disabled by default), with dashboards, APIs, and production-friendly deployment.

## Key Capabilities
- Agent orchestration: research, analysis, strategy, execution, risk, portfolio, monitoring
- Multi-source market data aggregation with caching
- Strategy modules: momentum, mean reversion, pairs, arbitrage
- Smart order routing and execution engine (Alpaca integration)
- Portfolio optimization and risk management (limits, sessions, alerts)
- Dashboards for monitoring and analytics (Streamlit)
- REST API endpoints for portfolio and performance metrics
- Deployment-ready: Dockerfiles, compose, and monitoring stack (Prometheus/Grafana)

## Safety and Secrets
- Do not commit `.env` files; use `.env.example` as a template.
- `.gitignore` excludes databases, logs, SSL, and virtual environments.
- Crypto tooling is disabled and commented out to avoid import issues.

## Quick Start (Python)
1. `python -m venv trading_env && source trading_env/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill values (Alpaca keys, FMP, etc.)
4. Run orchestrator: `python start_agent_orchestrator.py`
5. Launch dashboard: `streamlit run app.py --server.port 8501`

## Quick Start (Docker)
- `docker compose up -d`
- Visit `http://localhost:8501` for dashboards

## Configuration
- `config/agents.yaml`: agent definitions and capabilities
- `config/tools_config.yaml`: tool toggles and parameters
- `config/orchestrator_config.json`: orchestration behavior

## Architecture Overview
- `core/agent_orchestrator.py`: agent lifecycle, scheduling, sessions
- `core/market_data_aggregator.py`: unified feed, caching
- `core/execution_engine.py`: smart order routing
- `risk/`: risk limits and monitoring
- `dashboards/`: Streamlit components
- `monitoring/`: Prometheus/Grafana configs

## License and Attribution
This project is licensed under Apache-2.0. Attribution is required; please retain the `LICENSE` and `NOTICE` files in redistributions.

## Contributing
Issues and PRs are welcome. Please follow secure defaults and avoid committing secrets.
