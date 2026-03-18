from __future__ import annotations

"""Growth Agent

Implements a growth-focused valuation methodology.
"""

import json
import statistics
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state
from src.data.crypto_tickers import AssetType, detect_asset_type
from src.tools.api import (
    get_financial_metrics,
    get_insider_trades,
    get_crypto_metrics,
)

def growth_analyst_agent(state: AgentState, agent_id: str = "growth_analyst_agent"):
    """Run growth analysis across tickers and write signals back to `state`."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    growth_analysis: dict[str, dict] = {}

    for ticker in tickers:
        if detect_asset_type(ticker) == AssetType.CRYPTO:
            result = _analyze_crypto_growth(ticker, agent_id)
            if result:
                growth_analysis[ticker] = result
            continue

        # Equity path: original implementation
        progress.update_status(agent_id, ticker, "Fetching financial data")

        # --- Historical financial metrics ---
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=12, # 3 years of ttm data
            api_key=api_key,
        )
        if not financial_metrics or len(financial_metrics) < 4:
            progress.update_status(agent_id, ticker, "Failed: Not enough financial metrics")
            continue

        most_recent_metrics = financial_metrics[0]

        # --- Insider Trades ---
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key
        )

        # ------------------------------------------------------------------
        # Tool Implementation
        # ------------------------------------------------------------------

        # 1. Historical Growth Analysis
        growth_trends = analyze_growth_trends(financial_metrics)

        # 2. Growth-Oriented Valuation
        valuation_metrics = analyze_valuation(most_recent_metrics)

        # 3. Margin Expansion Monitor
        margin_trends = analyze_margin_trends(financial_metrics)

        # 4. Insider Conviction Tracker
        insider_conviction = analyze_insider_conviction(insider_trades)

        # 5. Financial Health Check
        financial_health = check_financial_health(most_recent_metrics)

        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        scores = {
            "growth": growth_trends['score'],
            "valuation": valuation_metrics['score'],
            "margins": margin_trends['score'],
            "insider": insider_conviction['score'],
            "health": financial_health['score']
        }

        weights = {
            "growth": 0.40,
            "valuation": 0.25,
            "margins": 0.15,
            "insider": 0.10,
            "health": 0.10
        }

        weighted_score = sum(scores[key] * weights[key] for key in scores)

        if weighted_score > 0.6:
            signal = "bullish"
        elif weighted_score < 0.4:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = round(abs(weighted_score - 0.5) * 2 * 100)

        reasoning = {
            "historical_growth": growth_trends,
            "growth_valuation": valuation_metrics,
            "margin_expansion": margin_trends,
            "insider_conviction": insider_conviction,
            "financial_health": financial_health,
            "final_analysis": {
                "signal": signal,
                "confidence": confidence,
                "weighted_score": round(weighted_score, 2)
            }
        }

        growth_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # ---- Emit message (for LLM tool chain) ----
    msg = HumanMessage(content=json.dumps(growth_analysis), name=agent_id)
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(growth_analysis, "Growth Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = growth_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [msg], "data": data}


def _analyze_crypto_growth(ticker: str, agent_id: str) -> dict | None:
    """Analyze crypto growth using volume trends, price momentum, and market cap rank."""
    progress.update_status(agent_id, ticker, "Fetching crypto data for growth analysis")

    crypto_data = get_crypto_metrics(ticker)
    if not crypto_data:
        progress.update_status(agent_id, ticker, "Failed: No crypto market data")
        return None

    scores = {}

    # 1. Price momentum as growth proxy (weight: 0.40)
    change_24h = crypto_data.price_change_24h or 0
    change_7d = crypto_data.price_change_7d or 0
    change_30d = crypto_data.price_change_30d or 0

    momentum_score = 0.5  # Start neutral
    if change_30d > 30:
        momentum_score = 1.0
    elif change_30d > 10:
        momentum_score = 0.75
    elif change_30d > 0:
        momentum_score = 0.6
    elif change_30d > -10:
        momentum_score = 0.4
    elif change_30d > -30:
        momentum_score = 0.25
    else:
        momentum_score = 0.0

    # Bonus for accelerating momentum (7d > 30d annualized pace)
    if change_7d > 0 and change_30d > 0 and change_7d * 4.3 > change_30d:
        momentum_score = min(momentum_score + 0.1, 1.0)

    scores["momentum"] = momentum_score
    momentum_details = {
        "score": momentum_score,
        "price_change_24h": change_24h,
        "price_change_7d": change_7d,
        "price_change_30d": change_30d,
    }

    # 2. Volume health as adoption growth proxy (weight: 0.25)
    vol_mcap = crypto_data.volume_to_market_cap
    volume_score = 0.5
    if vol_mcap is not None:
        if vol_mcap > 0.20:
            volume_score = 1.0
        elif vol_mcap > 0.10:
            volume_score = 0.75
        elif vol_mcap > 0.05:
            volume_score = 0.5
        else:
            volume_score = 0.25
    scores["volume"] = volume_score
    volume_details = {
        "score": volume_score,
        "volume_to_market_cap": vol_mcap,
    }

    # 3. Market cap rank as market position (weight: 0.20)
    rank = crypto_data.market_cap_rank
    rank_score = 0.5
    if rank is not None:
        if rank <= 5:
            rank_score = 0.6  # Established, moderate growth potential
        elif rank <= 15:
            rank_score = 0.8  # Good position with growth room
        elif rank <= 30:
            rank_score = 0.7
        elif rank <= 50:
            rank_score = 0.5
        else:
            rank_score = 0.3  # Higher risk
    scores["rank"] = rank_score
    rank_details = {
        "score": rank_score,
        "market_cap_rank": rank,
    }

    # 4. Supply dynamics (weight: 0.15) — less supply inflation = better growth retention
    circ = crypto_data.circulating_supply
    total = crypto_data.total_supply
    supply_score = 0.5
    supply_ratio = None
    if circ is not None and total is not None and total > 0:
        supply_ratio = circ / total
        if supply_ratio > 0.90:
            supply_score = 0.8  # Minimal dilution
        elif supply_ratio > 0.70:
            supply_score = 0.6
        elif supply_ratio > 0.50:
            supply_score = 0.4
        else:
            supply_score = 0.2  # Heavy future dilution
    scores["supply"] = supply_score
    supply_details = {
        "score": supply_score,
        "circulating_to_total": supply_ratio,
    }

    # Weighted aggregate
    weights = {
        "momentum": 0.40,
        "volume": 0.25,
        "rank": 0.20,
        "supply": 0.15,
    }
    weighted_score = sum(scores[k] * weights[k] for k in scores)

    if weighted_score > 0.6:
        signal = "bullish"
    elif weighted_score < 0.4:
        signal = "bearish"
    else:
        signal = "neutral"

    confidence = round(abs(weighted_score - 0.5) * 2 * 100)

    reasoning = {
        "price_momentum": momentum_details,
        "volume_growth": volume_details,
        "market_position": rank_details,
        "supply_dynamics": supply_details,
        "final_analysis": {
            "signal": signal,
            "confidence": confidence,
            "weighted_score": round(weighted_score, 2),
        },
    }

    progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    return {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }


#############################
# Helper Functions
#############################

def _calculate_trend(data: list[float | None]) -> float:
    """Calculates the slope of the trend line for the given data."""
    clean_data = [d for d in data if d is not None]
    if len(clean_data) < 2:
        return 0.0

    y = clean_data
    x = list(range(len(y)))

    try:
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(i * j for i, j in zip(x, y))
        sum_x2 = sum(i**2 for i in x)
        n = len(y)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        return slope
    except ZeroDivisionError:
        return 0.0

def analyze_growth_trends(metrics: list) -> dict:
    """Analyzes historical growth trends."""

    rev_growth = [m.revenue_growth for m in metrics]
    eps_growth = [m.earnings_per_share_growth for m in metrics]
    fcf_growth = [m.free_cash_flow_growth for m in metrics]

    rev_trend = _calculate_trend(rev_growth)
    eps_trend = _calculate_trend(eps_growth)
    fcf_trend = _calculate_trend(fcf_growth)

    # Score based on recent growth and trend
    score = 0

    # Revenue
    if rev_growth[0] is not None:
        if rev_growth[0] > 0.20:
            score += 0.4
        elif rev_growth[0] > 0.10:
            score += 0.2
        if rev_trend > 0:
            score += 0.1 # Accelerating

    # EPS
    if eps_growth[0] is not None:
        if eps_growth[0] > 0.20:
            score += 0.25
        elif eps_growth[0] > 0.10:
            score += 0.1
        if eps_trend > 0:
            score += 0.05

    # FCF
    if fcf_growth[0] is not None:
        if fcf_growth[0] > 0.15:
            score += 0.1

    score = min(score, 1.0)

    return {
        "score": score,
        "revenue_growth": rev_growth[0],
        "revenue_trend": rev_trend,
        "eps_growth": eps_growth[0],
        "eps_trend": eps_trend,
        "fcf_growth": fcf_growth[0],
        "fcf_trend": fcf_trend
    }

def analyze_valuation(metrics) -> dict:
    """Analyzes valuation from a growth perspective."""

    peg_ratio = metrics.peg_ratio
    ps_ratio = metrics.price_to_sales_ratio

    score = 0

    # PEG Ratio
    if peg_ratio is not None:
        if peg_ratio < 1.0:
            score += 0.5
        elif peg_ratio < 2.0:
            score += 0.25

    # Price to Sales Ratio
    if ps_ratio is not None:
        if ps_ratio < 2.0:
            score += 0.5
        elif ps_ratio < 5.0:
            score += 0.25

    score = min(score, 1.0)

    return {
        "score": score,
        "peg_ratio": peg_ratio,
        "price_to_sales_ratio": ps_ratio
    }

def analyze_margin_trends(metrics: list) -> dict:
    """Analyzes historical margin trends."""

    gross_margins = [m.gross_margin for m in metrics]
    operating_margins = [m.operating_margin for m in metrics]
    net_margins = [m.net_margin for m in metrics]

    gm_trend = _calculate_trend(gross_margins)
    om_trend = _calculate_trend(operating_margins)
    nm_trend = _calculate_trend(net_margins)

    score = 0

    # Gross Margin
    if gross_margins[0] is not None:
        if gross_margins[0] > 0.5: # Healthy margin
            score += 0.2
        if gm_trend > 0: # Expanding
            score += 0.2

    # Operating Margin
    if operating_margins[0] is not None:
        if operating_margins[0] > 0.15: # Healthy margin
            score += 0.2
        if om_trend > 0: # Expanding
            score += 0.2

    # Net Margin Trend
    if nm_trend > 0:
        score += 0.2

    score = min(score, 1.0)

    return {
        "score": score,
        "gross_margin": gross_margins[0],
        "gross_margin_trend": gm_trend,
        "operating_margin": operating_margins[0],
        "operating_margin_trend": om_trend,
        "net_margin": net_margins[0],
        "net_margin_trend": nm_trend
    }

def analyze_insider_conviction(trades: list) -> dict:
    """Analyzes insider trading activity."""

    buys = sum(t.transaction_value for t in trades if t.transaction_value and t.transaction_shares > 0)
    sells = sum(abs(t.transaction_value) for t in trades if t.transaction_value and t.transaction_shares < 0)

    if (buys + sells) == 0:
        net_flow_ratio = 0
    else:
        net_flow_ratio = (buys - sells) / (buys + sells)

    score = 0
    if net_flow_ratio > 0.5:
        score = 1.0
    elif net_flow_ratio > 0.1:
        score = 0.7
    elif net_flow_ratio > -0.1:
        score = 0.5 # Neutral
    else:
        score = 0.2

    return {
        "score": score,
        "net_flow_ratio": net_flow_ratio,
        "buys": buys,
        "sells": sells
    }

def check_financial_health(metrics) -> dict:
    """Checks the company's financial health."""

    debt_to_equity = metrics.debt_to_equity
    current_ratio = metrics.current_ratio

    score = 1.0

    # Debt to Equity
    if debt_to_equity is not None:
        if debt_to_equity > 1.5:
            score -= 0.5
        elif debt_to_equity > 0.8:
            score -= 0.2

    # Current Ratio
    if current_ratio is not None:
        if current_ratio < 1.0:
            score -= 0.5
        elif current_ratio < 1.5:
            score -= 0.2

    score = max(score, 0.0)

    return {
        "score": score,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio
    }
