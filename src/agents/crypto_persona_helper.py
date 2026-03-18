"""Shared helper for persona agents analyzing crypto assets.

Fetches crypto market data and formats it for LLM consumption, so each persona
agent can reason about crypto tickers using their unique investment philosophy.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal

from src.data.crypto_tickers import AssetType, detect_asset_type
from src.tools.api import get_crypto_metrics
from src.utils.llm import call_llm
from src.utils.progress import progress


def is_crypto(ticker: str) -> bool:
    return detect_asset_type(ticker) == AssetType.CRYPTO


def get_crypto_analysis_data(ticker: str, agent_id: str) -> dict | None:
    """Fetch crypto market data and format for LLM analysis.

    Returns a dict with crypto metrics formatted for inclusion in LLM prompts,
    or None if data is unavailable.
    """
    progress.update_status(agent_id, ticker, "Fetching crypto market data")
    crypto_data = get_crypto_metrics(ticker)
    if not crypto_data:
        progress.update_status(agent_id, ticker, "Failed: No crypto data available")
        return None

    return {
        "ticker": ticker,
        "asset_type": "cryptocurrency",
        "symbol": crypto_data.symbol,
        "current_price": crypto_data.current_price,
        "market_cap": crypto_data.market_cap,
        "market_cap_rank": crypto_data.market_cap_rank,
        "total_volume_24h": crypto_data.total_volume,
        "volume_to_market_cap": crypto_data.volume_to_market_cap,
        "price_change_24h_pct": crypto_data.price_change_24h,
        "price_change_7d_pct": crypto_data.price_change_7d,
        "price_change_30d_pct": crypto_data.price_change_30d,
        "all_time_high": crypto_data.ath,
        "ath_change_pct": crypto_data.ath_change_percentage,
        "circulating_supply": crypto_data.circulating_supply,
        "total_supply": crypto_data.total_supply,
        "supply_ratio": (crypto_data.circulating_supply / crypto_data.total_supply) if crypto_data.circulating_supply and crypto_data.total_supply and crypto_data.total_supply > 0 else None,
    }


CRYPTO_CONTEXT_BLOCK = """
NOTE: This is a CRYPTOCURRENCY asset, not a traditional equity. Traditional financial
metrics (P/E, debt-to-equity, revenue, earnings) do not apply. Instead, evaluate using:
- Market cap rank and dominance as a proxy for competitive position
- Volume/market cap ratio as a liquidity and adoption indicator
- Price trends (24h, 7d, 30d) for momentum assessment
- ATH distance for valuation context (how far from peak)
- Circulating/total supply ratio for dilution risk
- Network effects, adoption curves, and protocol utility (qualitative)

Apply your investment philosophy adapted to the crypto context. If this asset class
is outside your circle of competence, state that clearly and adjust confidence accordingly.
"""


def generate_crypto_persona_signal(
    ticker: str,
    crypto_data: dict,
    persona_system_prompt: str,
    signal_model: type[BaseModel],
    default_factory: callable,
    state: dict,
    agent_id: str,
) -> BaseModel:
    """Generate a persona-specific signal for a crypto asset via LLM."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                persona_system_prompt + "\n" + CRYPTO_CONTEXT_BLOCK,
            ),
            (
                "human",
                "Analyze this cryptocurrency asset:\n\n"
                "Ticker: {ticker}\n"
                "Market Data:\n{crypto_data}\n\n"
                "Return the trading signal in this JSON format:\n"
                "{{\n"
                '  "signal": "bullish" | "bearish" | "neutral",\n'
                '  "confidence": float between 0 and 100,\n'
                '  "reasoning": "string"\n'
                "}}",
            ),
        ]
    )

    prompt = template.invoke({
        "ticker": ticker,
        "crypto_data": json.dumps(crypto_data, indent=2),
    })

    return call_llm(
        prompt=prompt,
        pydantic_model=signal_model,
        agent_name=agent_id,
        state=state,
        default_factory=default_factory,
    )
