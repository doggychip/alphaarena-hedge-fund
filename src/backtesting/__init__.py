"""Backtesting package for AlphaArena hedge fund."""

from .types import (
    ActionLiteral,
    AgentDecision,
    AgentDecisions,
    AgentOutput,
    AgentSignals,
    PerformanceMetrics,
    PortfolioSnapshot,
    PortfolioValuePoint,
    PositionState,
    PriceDataFrame,
    TickerRealizedGains,
)

from .portfolio import Portfolio
from .trader import TradeExecutor
from .metrics import PerformanceMetricsCalculator
from .controller import AgentController
from .engine import BacktestEngine
from .valuation import calculate_portfolio_value, compute_exposures
from .output import OutputBuilder

__all__ = [
    "ActionLiteral",
    "AgentDecision",
    "AgentDecisions",
    "AgentOutput",
    "AgentSignals",
    "PerformanceMetrics",
    "PortfolioSnapshot",
    "PortfolioValuePoint",
    "PositionState",
    "PriceDataFrame",
    "TickerRealizedGains",
    "Portfolio",
    "TradeExecutor",
    "PerformanceMetricsCalculator",
    "AgentController",
    "BacktestEngine",
    "calculate_portfolio_value",
    "compute_exposures",
    "OutputBuilder",
]
