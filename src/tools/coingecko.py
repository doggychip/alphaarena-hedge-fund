"""CoinGecko API client for crypto price and market data."""

import os
import time
from datetime import datetime

import requests

from src.data.models import CompanyNews, Price

# Rate limiting for CoinGecko free tier (10-30 calls/min)
_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 2.5  # seconds between requests


def _rate_limit():
    """Enforce rate limiting for CoinGecko free tier."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _get_headers() -> dict:
    """Get request headers, including API key if available."""
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("COINGECKO_API_KEY")
    if api_key and api_key != "your-coingecko-api-key":
        headers["x-cg-demo-api-key"] = api_key
    return headers


BASE_URL = "https://api.coingecko.com/api/v3"


def get_crypto_prices(coin_id: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch historical price data from CoinGecko.

    Args:
        coin_id: CoinGecko coin ID (e.g. "bitcoin", "ethereum")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of Price objects with daily OHLCV data
    """
    try:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86400  # Include end date

        _rate_limit()
        resp = requests.get(
            f"{BASE_URL}/coins/{coin_id}/market_chart/range",
            params={"vs_currency": "usd", "from": start_ts, "to": end_ts},
            headers=_get_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        prices_raw = data.get("prices", [])
        volumes_raw = data.get("total_volumes", [])

        # Build a volume lookup by date
        volume_by_date = {}
        for ts_ms, vol in volumes_raw:
            date_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            volume_by_date[date_str] = vol

        # Group prices by date and build Price objects
        daily_prices: dict[str, list[float]] = {}
        for ts_ms, price in prices_raw:
            date_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            if date_str not in daily_prices:
                daily_prices[date_str] = []
            daily_prices[date_str].append(price)

        result = []
        for date_str in sorted(daily_prices.keys()):
            day_prices = daily_prices[date_str]
            open_price = day_prices[0]
            close_price = day_prices[-1]
            high_price = max(day_prices)
            low_price = min(day_prices)
            volume = volume_by_date.get(date_str, 0)

            result.append(
                Price(
                    open=open_price,
                    close=close_price,
                    high=high_price,
                    low=low_price,
                    volume=int(volume) if volume else 0,
                    time=date_str,
                    ticker=coin_id,
                )
            )

        return result
    except Exception as e:
        print(f"CoinGecko price fetch failed for {coin_id}: {e}")
        return []


def get_crypto_market_data(coin_id: str) -> dict:
    """Fetch current market data for a crypto asset.

    Returns dict with market_cap, total_volume, price changes, supply info, etc.
    """
    try:
        _rate_limit()
        resp = requests.get(
            f"{BASE_URL}/coins/{coin_id}",
            params={"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"},
            headers=_get_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        market = data.get("market_data", {})
        return {
            "coin_id": coin_id,
            "symbol": data.get("symbol", ""),
            "market_cap": market.get("market_cap", {}).get("usd"),
            "total_volume": market.get("total_volume", {}).get("usd"),
            "price_change_24h": market.get("price_change_percentage_24h"),
            "price_change_7d": market.get("price_change_percentage_7d"),
            "price_change_30d": market.get("price_change_percentage_30d"),
            "circulating_supply": market.get("circulating_supply"),
            "total_supply": market.get("total_supply"),
            "ath": market.get("ath", {}).get("usd"),
            "ath_change_percentage": market.get("ath_change_percentage", {}).get("usd"),
            "market_cap_rank": data.get("market_cap_rank"),
            "current_price": market.get("current_price", {}).get("usd"),
        }
    except Exception as e:
        print(f"CoinGecko market data fetch failed for {coin_id}: {e}")
        return {}


def get_crypto_news_search(coin_id: str) -> list[CompanyNews]:
    """Attempt to get crypto-related news. Returns empty list if unavailable.

    CoinGecko free tier doesn't have a dedicated news endpoint,
    so this returns an empty list. The news_sentiment agent handles this gracefully.
    """
    return []
