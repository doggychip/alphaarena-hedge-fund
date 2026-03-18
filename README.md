# AlphaArena Hedge Fund

AI-powered multi-agent hedge fund supporting **equities** and **crypto** assets.

Built on [LangGraph](https://github.com/langchain-ai/langgraph) for multi-agent orchestration, with 19 AI analyst agents that generate trading signals using a mix of algorithmic analysis and LLM-based reasoning.

> Forked from [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) and extended with full cryptocurrency support via [CoinGecko API](https://www.coingecko.com/en/api).

## Features

- **19 AI Analyst Agents** — 12 persona-based (LLM), 5 specialist (algorithmic), 2 management
- **Dual Asset Class** — Seamlessly analyze equities (via Financial Datasets API) and crypto (via CoinGecko)
- **Multi-Agent Orchestration** — Parallel analyst execution → risk management → portfolio decisions
- **Backtesting Engine** — Full backtest support with Sharpe ratio, Sortino ratio, and max drawdown metrics
- **Multiple LLM Providers** — OpenAI, Anthropic, Google Gemini, Groq, Ollama (local)

## Quick Start

### 1. Install dependencies

```bash
cd alphaarena-hedge-fund
poetry install
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY (required for LLM agents)
# - FINANCIAL_DATASETS_API_KEY (required for equity data)
# - COINGECKO_API_KEY (optional — free tier works without key)
```

### 3. Run the hedge fund

```bash
# Equity analysis
poetry run python src/main.py --tickers AAPL,MSFT,GOOGL

# Crypto analysis
poetry run python src/main.py --tickers BTC,ETH,SOL

# Mixed portfolio
poetry run python src/main.py --tickers AAPL,BTC,ETH

# With specific analysts
poetry run python src/main.py --tickers BTC,ETH --analysts cathie_wood,stanley_druckenmiller,technical_analyst

# Show agent reasoning
poetry run python src/main.py --tickers BTC --show-reasoning
```

### 4. Run backtesting

```bash
# Backtest with crypto
poetry run python src/backtester.py --tickers BTC,ETH --start-date 2025-01-01 --end-date 2025-03-01

# Backtest with equities
poetry run python src/backtester.py --tickers AAPL,MSFT --start-date 2025-01-01 --end-date 2025-03-01
```

## Supported Crypto Tickers

| Symbol | CoinGecko ID    |
|--------|-----------------|
| BTC    | bitcoin         |
| ETH    | ethereum        |
| BNB    | binancecoin     |
| XRP    | ripple          |
| SOL    | solana          |
| ADA    | cardano         |
| DOGE   | dogecoin        |
| AVAX   | avalanche-2     |
| DOT    | polkadot        |
| LINK   | chainlink       |

Any equity ticker not in this list is routed to Financial Datasets API.

## Agent Roster

### Persona Agents (LLM-based)

| Agent | Style |
|-------|-------|
| Warren Buffett | Value investing, competitive moats, long-term ownership |
| Charlie Munger | Mental models, inversion thinking, quality businesses |
| Ben Graham | Margin of safety, intrinsic value, earnings stability |
| Peter Lynch | GARP — growth at a reasonable price |
| Phil Fisher | Scuttlebutt method, management quality, R&D investment |
| Cathie Wood | Disruptive innovation, exponential growth, crypto-native thesis |
| Michael Burry | Deep value, contrarian bets, downside protection |
| Stanley Druckenmiller | Macro trends, asymmetric risk-reward, crypto as macro trade |
| Bill Ackman | High-quality businesses, activist potential |
| Aswath Damodaran | Rigorous valuation — DCF, CAPM, intrinsic value |
| Rakesh Jhunjhunwala | Emerging market dynamics, growth at reasonable price |
| Mohnish Pabrai | Dhandho — heads I win, tails I don't lose much |

### Specialist Agents (Algorithmic)

| Agent | Analysis |
|-------|----------|
| Technical Analyst | Chart patterns, indicators, price action |
| Fundamentals Analyst | Financial statements, ratios, health metrics |
| Sentiment Analyst | Insider trades (equity) / momentum & volume (crypto) |
| Valuation Analyst | DCF, EV/EBITDA (equity) / NVT, ATH range (crypto) |
| Growth Analyst | Revenue growth (equity) / momentum & volume trends (crypto) |
| News Sentiment | News analysis and sentiment scoring |

### Management Agents

| Agent | Role |
|-------|------|
| Risk Manager | Position sizing, exposure limits, risk assessment |
| Portfolio Manager | Final trading decisions based on all signals |

## Architecture

```
                         ┌─────────────┐
                         │  Start Node │
                         └──────┬──────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────┴─────┐ ┌──┴──┐ ┌─────┴─────┐
              │  Persona   │ │Tech │ │  Growth   │  ... (17 analysts in parallel)
              │  Agents    │ │Anal.│ │  Analyst  │
              └─────┬─────┘ └──┬──┘ └─────┬─────┘
                    │          │           │
                    └──────────┼───────────┘
                               │
                     ┌─────────┴─────────┐
                     │  Risk Management  │
                     └─────────┬─────────┘
                               │
                     ┌─────────┴─────────┐
                     │ Portfolio Manager  │
                     └─────────┬─────────┘
                               │
                            ┌──┴──┐
                            │ END │
                            └─────┘
```

**Signal Format** (all agents):
```json
{
  "signal": "bullish | bearish | neutral",
  "confidence": 0-100,
  "reasoning": { ... }
}
```

**Crypto Routing**: Asset type is detected from the ticker symbol. Crypto tickers are routed to CoinGecko API; equity tickers to Financial Datasets API. All agents handle both asset types transparently.

## License

MIT

## Credits

- Original project: [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)
- Crypto data: [CoinGecko API](https://www.coingecko.com/en/api)
