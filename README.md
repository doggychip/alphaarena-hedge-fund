# AlphaArena Hedge Fund

An AI-powered hedge fund system that uses multiple LLM-based analyst agents to make trading decisions for **both equities and cryptocurrencies**.

Forked from [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) and extended with full crypto asset support via CoinGecko API.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLI / Main                        │
├─────────────────────────────────────────────────────┤
│              LangGraph StateGraph                    │
│                                                      │
│  start_node ──┬── analyst_1 ──┐                     │
│               ├── analyst_2 ──┤                     │
│               ├── ...        ──┤                     │
│               └── analyst_N ──┤                     │
│                               ▼                     │
│                    risk_management_agent              │
│                               ▼                     │
│                    portfolio_manager                  │
│                               ▼                     │
│                              END                     │
└─────────────────────────────────────────────────────┘
```

All analyst agents run **in parallel**, feeding into risk management, then portfolio management for final decisions.

## Agents (19 total)

### 12 Persona Agents (LLM-based)
Each channels the investment philosophy of a famous investor:

| Agent | Style |
|-------|-------|
| **Warren Buffett** | Value investing, competitive moats, margin of safety |
| **Charlie Munger** | Mental models, inversion thinking, quality businesses |
| **Ben Graham** | Deep value, margin of safety, earnings stability |
| **Peter Lynch** | GARP — growth at a reasonable price |
| **Phil Fisher** | Scuttlebutt method, management quality, R&D |
| **Cathie Wood** | Disruptive innovation, exponential growth |
| **Michael Burry** | Contrarian deep value, downside protection |
| **Stanley Druckenmiller** | Macro trends, asymmetric risk-reward |
| **Bill Ackman** | Activist investing, strong brands |
| **Aswath Damodaran** | DCF, CAPM, intrinsic value |
| **Rakesh Jhunjhunwala** | Emerging market growth |
| **Mohnish Pabrai** | Dhandho — heads I win, tails I don't lose much |

### 5 Specialist Agents (Algorithmic + LLM)
| Agent | Approach |
|-------|----------|
| **Technical Analyst** | Chart patterns, moving averages, RSI, MACD, Bollinger Bands |
| **Fundamentals Analyst** | Financial statements, profitability, solvency ratios |
| **Sentiment Analyst** | Insider trades + news sentiment (equity) / price momentum + volume (crypto) |
| **Valuation Analyst** | DCF, owner earnings, EV/EBITDA, residual income (equity) / NVT, ATH range (crypto) |
| **Growth Analyst** | Revenue/earnings growth, margins, insider activity (equity) / momentum, volume, rank (crypto) |
| **News Sentiment** | Company news analysis via LLM |

### 2 Management Agents
| Agent | Role |
|-------|------|
| **Risk Manager** | Position sizing, exposure limits, correlation checks |
| **Portfolio Manager** | Final buy/sell/hold decisions with quantities |

## Crypto Support

All agents automatically detect crypto tickers and route to crypto-specific analysis:

- **Specialist agents** use custom scoring algorithms based on CoinGecko market data (price momentum, volume, market cap rank, supply dynamics, NVT proxies)
- **Persona agents** use a shared LLM helper that provides crypto market context and lets each persona apply their philosophy to crypto assets

### Supported Crypto Tickers
`BTC`, `ETH`, `BNB`, `XRP`, `SOL`, `ADA`, `DOGE`, `AVAX`, `DOT`, `LINK`

### Data Sources
- **Equities**: Financial Datasets API (prices, financials, insider trades, news)
- **Crypto**: CoinGecko free tier API (market data, prices, volumes, supply)

## Setup

### Prerequisites
- Python 3.11+
- Poetry

### Installation

```bash
# Install dependencies
poetry install

# Copy and fill in your API keys
cp .env.example .env
```

### Required API Keys

Set in `.env`:
```
OPENAI_API_KEY=...           # or other LLM provider
FINANCIAL_DATASETS_API_KEY=... # for equity data
# CoinGecko free tier requires no API key
```

## Usage

### Run the hedge fund
```bash
# Equities
poetry run python -m src.main --tickers AAPL,MSFT,GOOGL

# Crypto
poetry run python -m src.main --tickers BTC,ETH,SOL

# Mixed portfolio
poetry run python -m src.main --tickers AAPL,BTC,ETH

# With specific analysts
poetry run python -m src.main --tickers BTC --analysts warren_buffett,cathie_wood

# All analysts, show reasoning
poetry run python -m src.main --tickers BTC --analysts-all --show-reasoning
```

### Run backtesting
```bash
poetry run python -m src.backtester --tickers AAPL,BTC --start-date 2025-01-01 --end-date 2025-03-01
```

### CLI Options
- `--tickers` — Comma-separated ticker symbols
- `--analysts` — Comma-separated analyst keys
- `--analysts-all` — Use all 17 analysts
- `--model` — LLM model name (default: gpt-4.1)
- `--ollama` — Use local Ollama inference
- `--start-date` / `--end-date` — Date range (YYYY-MM-DD)
- `--initial-cash` — Starting capital (default: $100,000)
- `--show-reasoning` — Display each agent's reasoning

## Signal Format

All agents produce signals in a unified format:
```json
{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": 0-100,
  "reasoning": "..."
}
```

## Project Structure

```
src/
├── agents/                  # All 19 agents
│   ├── crypto_persona_helper.py  # Shared crypto LLM helper for persona agents
│   ├── warren_buffett.py
│   ├── charlie_munger.py
│   ├── ... (10 more persona agents)
│   ├── technicals.py
│   ├── fundamentals.py
│   ├── sentiment.py
│   ├── valuation.py
│   ├── growth_agent.py
│   ├── news_sentiment.py
│   ├── risk_manager.py
│   └── portfolio_manager.py
├── backtesting/             # Backtesting engine
│   ├── engine.py
│   ├── benchmarks.py        # SPY (equity) / BTC (crypto) benchmarks
│   ├── portfolio.py
│   └── ...
├── cli/                     # CLI input handling
├── data/                    # Data models, crypto tickers, cache
│   ├── crypto_tickers.py    # Ticker → CoinGecko ID mapping
│   ├── models.py            # Pydantic models incl. CryptoMarketData
│   └── cache.py
├── graph/                   # LangGraph state definition
├── llm/                     # LLM model configs
├── tools/                   # API clients
│   ├── api.py               # Unified API (routes crypto to CoinGecko)
│   └── coingecko.py         # CoinGecko REST client
└── utils/                   # Display, analysts config, progress
```

## License

MIT
