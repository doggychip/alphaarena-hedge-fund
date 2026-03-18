from __future__ import annotations

from typing import Sequence

from .types import PerformanceMetrics, PortfolioValuePoint


class PerformanceMetricsCalculator:
    def __init__(self, *, annual_trading_days: int = 252, annual_rf_rate: float = 0.0434) -> None:
        self.annual_trading_days = annual_trading_days
        self.annual_rf_rate = annual_rf_rate

    def compute_metrics(self, values: Sequence[PortfolioValuePoint]) -> PerformanceMetrics:
        import pandas as pd
        import numpy as np

        if not values:
            return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}

        df = pd.DataFrame(values)
        if df.empty or "Portfolio Value" not in df:
            return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}

        df = df.set_index("Date")
        df["Daily Return"] = df["Portfolio Value"].pct_change()
        clean_returns = df["Daily Return"].dropna()
        if len(clean_returns) < 2:
            return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}

        daily_rf = self.annual_rf_rate / self.annual_trading_days
        excess = clean_returns - daily_rf
        mean_excess = excess.mean()
        std_excess = excess.std()

        sharpe = float(np.sqrt(self.annual_trading_days) * (mean_excess / std_excess)) if std_excess > 1e-12 else 0.0

        downside_diff = np.minimum(excess, 0)
        downside_dev = float(np.sqrt(np.mean(downside_diff**2)))
        if downside_dev > 1e-12:
            sortino = float(np.sqrt(self.annual_trading_days) * (mean_excess / downside_dev))
        else:
            sortino = float("inf") if mean_excess > 0 else 0.0

        rolling_max = df["Portfolio Value"].cummax()
        drawdown = (df["Portfolio Value"] - rolling_max) / rolling_max
        if len(drawdown) > 0:
            min_dd = float(drawdown.min())
            max_drawdown = float(min_dd * 100.0)
            max_drawdown_date = drawdown.idxmin().strftime("%Y-%m-%d") if min_dd < 0 else None
        else:
            max_drawdown = 0.0
            max_drawdown_date = None

        return {"sharpe_ratio": sharpe, "sortino_ratio": sortino, "max_drawdown": max_drawdown, "max_drawdown_date": max_drawdown_date}
