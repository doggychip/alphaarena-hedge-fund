from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.data.crypto_tickers import AssetType, detect_asset_type
from src.tools.api import get_insider_trades, get_company_news, get_crypto_metrics
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        if detect_asset_type(ticker) == AssetType.CRYPTO:
            result = _analyze_crypto_sentiment(ticker, agent_id)
            if result:
                sentiment_analysis[ticker] = result
            continue

        # Equity path: original implementation
        progress.update_status(agent_id, ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status(agent_id, ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", np.where(sentiment == "positive", "bullish", "neutral")).tolist()

        progress.update_status(agent_id, ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7

        # Calculate weighted signal counts
        bullish_signals = insider_signals.count("bullish") * insider_weight + news_signals.count("bullish") * news_weight
        bearish_signals = insider_signals.count("bearish") * insider_weight + news_signals.count("bearish") * news_weight

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_weighted_signals) * 100, 2)

        # Create structured reasoning similar to technical analysis
        reasoning = {
            "insider_trading": {
                "signal": "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral",
                "confidence": round((max(insider_signals.count("bullish"), insider_signals.count("bearish")) / max(len(insider_signals), 1)) * 100),
                "metrics": {
                    "total_trades": len(insider_signals),
                    "bullish_trades": insider_signals.count("bullish"),
                    "bearish_trades": insider_signals.count("bearish"),
                    "weight": insider_weight,
                    "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 1),
                    "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 1),
                },
            },
            "news_sentiment": {
                "signal": "bullish" if news_signals.count("bullish") > news_signals.count("bearish") else "bearish" if news_signals.count("bearish") > news_signals.count("bullish") else "neutral",
                "confidence": round((max(news_signals.count("bullish"), news_signals.count("bearish")) / max(len(news_signals), 1)) * 100),
                "metrics": {
                    "total_articles": len(news_signals),
                    "bullish_articles": news_signals.count("bullish"),
                    "bearish_articles": news_signals.count("bearish"),
                    "neutral_articles": news_signals.count("neutral"),
                    "weight": news_weight,
                    "weighted_bullish": round(news_signals.count("bullish") * news_weight, 1),
                    "weighted_bearish": round(news_signals.count("bearish") * news_weight, 1),
                },
            },
            "combined_analysis": {
                "total_weighted_bullish": round(bullish_signals, 1),
                "total_weighted_bearish": round(bearish_signals, 1),
                "signal_determination": f"{'Bullish' if bullish_signals > bearish_signals else 'Bearish' if bearish_signals > bullish_signals else 'Neutral'} based on weighted signal comparison",
            },
        }

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }


def _analyze_crypto_sentiment(ticker: str, agent_id: str) -> dict | None:
    """Analyze crypto sentiment using market data (no insider trades for crypto)."""
    progress.update_status(agent_id, ticker, "Fetching crypto market data for sentiment")

    crypto_data = get_crypto_metrics(ticker)
    if not crypto_data:
        progress.update_status(agent_id, ticker, "Failed: No crypto market data")
        return None

    signals = []
    reasoning = {}

    # 1. Price momentum (short-term sentiment proxy)
    change_24h = crypto_data.price_change_24h or 0
    change_7d = crypto_data.price_change_7d or 0

    momentum_score = 0
    if change_24h > 5:
        momentum_score += 2
    elif change_24h > 0:
        momentum_score += 1
    elif change_24h < -5:
        momentum_score -= 2
    else:
        momentum_score -= 1

    if change_7d > 10:
        momentum_score += 2
    elif change_7d > 0:
        momentum_score += 1
    elif change_7d < -10:
        momentum_score -= 2
    else:
        momentum_score -= 1

    if momentum_score >= 2:
        signals.append("bullish")
    elif momentum_score <= -2:
        signals.append("bearish")
    else:
        signals.append("neutral")

    reasoning["price_momentum"] = {
        "signal": signals[-1],
        "details": f"24h: {change_24h:.1f}%, 7d: {change_7d:.1f}%, score: {momentum_score}",
    }

    # 2. Volume health (high volume = strong sentiment)
    vol_mcap = crypto_data.volume_to_market_cap
    if vol_mcap is not None:
        if vol_mcap > 0.15:
            signals.append("bullish")
        elif vol_mcap > 0.05:
            signals.append("neutral")
        else:
            signals.append("bearish")
    else:
        signals.append("neutral")
    reasoning["volume_sentiment"] = {
        "signal": signals[-1],
        "details": f"Volume/MCap: {vol_mcap:.4f}" if vol_mcap else "Volume/MCap: N/A",
    }

    # 3. ATH proximity (near ATH = euphoria/FOMO, far = fear/depression)
    ath_change = crypto_data.ath_change_percentage
    if ath_change is not None:
        if ath_change > -10:
            signals.append("bullish")  # Near ATH, strong positive sentiment
        elif ath_change > -40:
            signals.append("neutral")
        else:
            signals.append("bearish")  # Far from ATH, negative sentiment
    else:
        signals.append("neutral")
    reasoning["ath_sentiment"] = {
        "signal": signals[-1],
        "details": f"ATH Change: {ath_change:.1f}%" if ath_change else "ATH Change: N/A",
    }

    # 4. 30-day trend (longer-term sentiment)
    change_30d = crypto_data.price_change_30d or 0
    if change_30d > 15:
        signals.append("bullish")
    elif change_30d > -15:
        signals.append("neutral")
    else:
        signals.append("bearish")
    reasoning["monthly_trend"] = {
        "signal": signals[-1],
        "details": f"30d: {change_30d:.1f}%",
    }

    # Aggregate
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
