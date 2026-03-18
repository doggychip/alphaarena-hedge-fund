from __future__ import annotations

import pandas as pd

from src.tools.api import get_price_data, get_crypto_metrics
from src.data.crypto_tickers import AssetType, detect_asset_type


class BenchmarkCalculator:
    """Compute benchmark returns for both equities (SPY) and crypto (BTC)."""

    def get_benchmark_ticker(self, tickers: list[str]) -> str:
        """Choose benchmark: BTC if all tickers are crypto, else SPY."""
        if all(detect_asset_type(t) == AssetType.CRYPTO for t in tickers):
            return "BTC"
        return "SPY"

    def get_return_pct(self, ticker: str, start_date: str, end_date: str) -> float | None:
        """Compute simple buy-and-hold return % for ticker from start_date to end_date."""
        if detect_asset_type(ticker) == AssetType.CRYPTO:
            return self._get_crypto_return_pct(ticker, start_date, end_date)
        return self._get_equity_return_pct(ticker, start_date, end_date)

    def _get_equity_return_pct(self, ticker: str, start_date: str, end_date: str) -> float | None:
        try:
            df = get_price_data(ticker, start_date, end_date)
            if df.empty:
                return None
            first_close = df.iloc[0]["close"]
            last_close = df.iloc[-1]["close"]
            if first_close is None or pd.isna(first_close):
                return None
            if last_close is None or pd.isna(last_close):
                last_valid = df["close"].dropna()
                if last_valid.empty:
                    return None
                last_close = float(last_valid.iloc[-1])
            return (float(last_close) / float(first_close) - 1.0) * 100.0
        except Exception:
            return None

    def _get_crypto_return_pct(self, ticker: str, start_date: str, end_date: str) -> float | None:
        try:
            # First try using price data (same path as equities, routed via CoinGecko)
            df = get_price_data(ticker, start_date, end_date)
            if not df.empty:
                first_close = df.iloc[0]["close"]
                last_close = df.iloc[-1]["close"]
                if first_close and not pd.isna(first_close) and first_close > 0:
                    if last_close and not pd.isna(last_close):
                        return (float(last_close) / float(first_close) - 1.0) * 100.0
            # Fallback to 30d change from market data
            metrics = get_crypto_metrics(ticker)
            if metrics and metrics.price_change_30d is not None:
                return float(metrics.price_change_30d)
            return None
        except Exception:
            return None
