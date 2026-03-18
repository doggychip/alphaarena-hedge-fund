from __future__ import annotations

from datetime import datetime
from typing import Sequence, Dict

import pandas as pd
from dateutil.relativedelta import relativedelta

from .controller import AgentController
from .trader import TradeExecutor
from .metrics import PerformanceMetricsCalculator
from .portfolio import Portfolio
from .types import PerformanceMetrics, PortfolioValuePoint
from .valuation import calculate_portfolio_value, compute_exposures
from .output import OutputBuilder
from .benchmarks import BenchmarkCalculator

from src.data.crypto_tickers import AssetType, detect_asset_type
from src.tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
    get_crypto_metrics,
)


class BacktestEngine:
    """Coordinates the backtest loop. Supports both equities and crypto tickers."""

    def __init__(
        self,
        *,
        agent,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        initial_margin_requirement: float,
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._initial_capital = float(initial_capital)
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts

        self._portfolio = Portfolio(
            tickers=tickers,
            initial_cash=initial_capital,
            margin_requirement=initial_margin_requirement,
        )
        self._executor = TradeExecutor()
        self._agent_controller = AgentController()
        self._perf = PerformanceMetricsCalculator()
        self._results = OutputBuilder(initial_capital=self._initial_capital)

        self._benchmark = BenchmarkCalculator()
        self._benchmark_ticker = self._benchmark.get_benchmark_ticker(tickers)

        self._portfolio_values: list[PortfolioValuePoint] = []
        self._table_rows: list[list] = []
        self._performance_metrics: PerformanceMetrics = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "long_short_ratio": None,
            "gross_exposure": None,
            "net_exposure": None,
        }

    def _prefetch_data(self) -> None:
        end_date_dt = datetime.strptime(self._end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        for ticker in self._tickers:
            if detect_asset_type(ticker) == AssetType.CRYPTO:
                # Crypto data is fetched on-demand via CoinGecko; just warm the cache
                get_crypto_metrics(ticker)
            else:
                get_prices(ticker, start_date_str, self._end_date)
                get_financial_metrics(ticker, self._end_date, limit=10)
                get_insider_trades(ticker, self._end_date, start_date=self._start_date, limit=1000)
                get_company_news(ticker, self._end_date, start_date=self._start_date, limit=1000)

        # Preload benchmark data
        if detect_asset_type(self._benchmark_ticker) != AssetType.CRYPTO:
            get_prices(self._benchmark_ticker, self._start_date, self._end_date)

    def _get_current_price(self, ticker: str, previous_date_str: str, current_date_str: str) -> float | None:
        """Get current price for a ticker, handling both equities and crypto."""
        if detect_asset_type(ticker) == AssetType.CRYPTO:
            metrics = get_crypto_metrics(ticker)
            if metrics and metrics.current_price:
                return float(metrics.current_price)
            return None
        else:
            try:
                price_data = get_price_data(ticker, previous_date_str, current_date_str)
                if price_data.empty:
                    return None
                return float(price_data.iloc[-1]["close"])
            except Exception:
                return None

    def run_backtest(self) -> PerformanceMetrics:
        self._prefetch_data()

        dates = pd.date_range(self._start_date, self._end_date, freq="B")
        if len(dates) > 0:
            self._portfolio_values = [
                {"Date": dates[0], "Portfolio Value": self._initial_capital}
            ]
        else:
            self._portfolio_values = []

        for current_date in dates:
            lookback_start = (current_date - relativedelta(months=1)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - relativedelta(days=1)).strftime("%Y-%m-%d")
            if lookback_start == current_date_str:
                continue

            try:
                current_prices: Dict[str, float] = {}
                missing_data = False
                for ticker in self._tickers:
                    price = self._get_current_price(ticker, previous_date_str, current_date_str)
                    if price is None:
                        missing_data = True
                        break
                    current_prices[ticker] = price
                if missing_data:
                    continue
            except Exception:
                continue

            agent_output = self._agent_controller.run_agent(
                self._agent,
                tickers=self._tickers,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self._portfolio,
                model_name=self._model_name,
                model_provider=self._model_provider,
                selected_analysts=self._selected_analysts,
            )
            decisions = agent_output["decisions"]

            executed_trades: Dict[str, int] = {}
            for ticker in self._tickers:
                d = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action = d.get("action", "hold")
                qty = d.get("quantity", 0)
                executed_qty = self._executor.execute_trade(ticker, action, qty, current_prices[ticker], self._portfolio)
                executed_trades[ticker] = executed_qty

            total_value = calculate_portfolio_value(self._portfolio, current_prices)
            exposures = compute_exposures(self._portfolio, current_prices)

            point: PortfolioValuePoint = {
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": exposures["Long Exposure"],
                "Short Exposure": exposures["Short Exposure"],
                "Gross Exposure": exposures["Gross Exposure"],
                "Net Exposure": exposures["Net Exposure"],
                "Long/Short Ratio": exposures["Long/Short Ratio"],
            }
            self._portfolio_values.append(point)

            rows = self._results.build_day_rows(
                date_str=current_date_str,
                tickers=self._tickers,
                agent_output=agent_output,
                executed_trades=executed_trades,
                current_prices=current_prices,
                portfolio=self._portfolio,
                performance_metrics=self._performance_metrics,
                total_value=total_value,
                benchmark_return_pct=self._benchmark.get_return_pct(
                    self._benchmark_ticker, self._start_date, current_date_str
                ),
            )
            self._table_rows = rows + self._table_rows
            self._results.print_rows(self._table_rows)

            if len(self._portfolio_values) > 3:
                computed = self._perf.compute_metrics(self._portfolio_values)
                if computed:
                    self._performance_metrics.update(computed)

        return self._performance_metrics

    def get_portfolio_values(self) -> Sequence[PortfolioValuePoint]:
        return list(self._portfolio_values)
