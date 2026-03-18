from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress
from src.data.crypto_tickers import AssetType, detect_asset_type
from src.tools.api import get_financial_metrics, get_crypto_metrics
import json


##### Fundamental Agent #####
def fundamentals_analyst_agent(state: AgentState, agent_id: str = "fundamentals_analyst_agent"):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    fundamental_analysis = {}

    for ticker in tickers:
        if detect_asset_type(ticker) == AssetType.CRYPTO:
            # Crypto path: use market data instead of financial statements
            result = _analyze_crypto_fundamentals(ticker, agent_id)
            if result:
                fundamental_analysis[ticker] = result
            continue

        # Equity path: original implementation
        progress.update_status(agent_id, ticker, "Fetching financial metrics")

        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        if not financial_metrics:
            progress.update_status(agent_id, ticker, "Failed: No financial metrics found")
            continue

        metrics = financial_metrics[0]
        signals = []
        reasoning = {}

        progress.update_status(agent_id, ticker, "Analyzing profitability")
        # 1. Profitability Analysis
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        thresholds = [
            (return_on_equity, 0.15),
            (net_margin, 0.20),
            (operating_margin, 0.15),
        ]
        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing growth")
        # 2. Growth Analysis
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        thresholds = [
            (revenue_growth, 0.10),
            (earnings_growth, 0.10),
            (book_value_growth, 0.10),
        ]
        growth_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing financial health")
        # 3. Financial Health
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_score = 0
        if current_ratio and current_ratio > 1.5:
            health_score += 1
        if debt_to_equity and debt_to_equity < 0.5:
            health_score += 1
        if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * 0.8:
            health_score += 1

        signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        thresholds = [
            (pe_ratio, 25),
            (pb_ratio, 3),
            (ps_ratio, 5),
        ]
        price_ratio_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bearish" if price_ratio_score >= 2 else "bullish" if price_ratio_score == 0 else "neutral")
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        progress.update_status(agent_id, ticker, "Calculating final signal")
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        total_signals = len(signals)
        confidence = round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    state["data"]["analyst_signals"][agent_id] = fundamental_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }


def _analyze_crypto_fundamentals(ticker: str, agent_id: str) -> dict | None:
    """Analyze crypto asset fundamentals using market data."""
    progress.update_status(agent_id, ticker, "Fetching crypto market data")

    crypto_data = get_crypto_metrics(ticker)
    if not crypto_data:
        progress.update_status(agent_id, ticker, "Failed: No crypto market data")
        return None

    signals = []
    reasoning = {}

    # 1. Volume/Market Cap Health (healthy > 0.05)
    vol_mcap = crypto_data.volume_to_market_cap
    if vol_mcap is not None:
        if vol_mcap > 0.10:
            signals.append("bullish")
        elif vol_mcap > 0.05:
            signals.append("neutral")
        else:
            signals.append("bearish")
    else:
        signals.append("neutral")
    reasoning["volume_health"] = {
        "signal": signals[-1],
        "details": f"Volume/MCap: {vol_mcap:.4f}" if vol_mcap else "Volume/MCap: N/A",
    }

    # 2. Price Change Trends
    change_24h = crypto_data.price_change_24h or 0
    change_7d = crypto_data.price_change_7d or 0
    change_30d = crypto_data.price_change_30d or 0
    trend_score = sum([change_24h > 0, change_7d > 0, change_30d > 0])

    if trend_score >= 2:
        signals.append("bullish")
    elif trend_score == 0:
        signals.append("bearish")
    else:
        signals.append("neutral")
    reasoning["price_trends"] = {
        "signal": signals[-1],
        "details": f"24h: {change_24h:.1f}%, 7d: {change_7d:.1f}%, 30d: {change_30d:.1f}%",
    }

    # 3. Market Cap Rank Stability
    rank = crypto_data.market_cap_rank
    if rank is not None:
        if rank <= 10:
            signals.append("bullish")
        elif rank <= 50:
            signals.append("neutral")
        else:
            signals.append("bearish")
    else:
        signals.append("neutral")
    reasoning["market_position"] = {
        "signal": signals[-1],
        "details": f"Market Cap Rank: #{rank}" if rank else "Rank: N/A",
    }

    # 4. ATH Distance (further from ATH = more potential upside)
    ath_change = crypto_data.ath_change_percentage
    if ath_change is not None:
        if ath_change > -20:
            signals.append("bearish")  # Near ATH, less upside
        elif ath_change > -50:
            signals.append("neutral")
        else:
            signals.append("bullish")  # Far from ATH, more upside potential
    else:
        signals.append("neutral")
    reasoning["ath_distance"] = {
        "signal": signals[-1],
        "details": f"ATH Change: {ath_change:.1f}%" if ath_change else "ATH Change: N/A",
    }

    bullish_count = signals.count("bullish")
    bearish_count = signals.count("bearish")

    if bullish_count > bearish_count:
        overall_signal = "bullish"
    elif bearish_count > bullish_count:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    total = len(signals)
    confidence = round(max(bullish_count, bearish_count) / total, 2) * 100

    progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    return {
        "signal": overall_signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }
