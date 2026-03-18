from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, TypedDict, Literal
from enum import Enum

import pandas as pd


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"
    HOLD = "hold"

# Backward-compatible alias
ActionLiteral = Literal["buy", "sell", "short", "cover", "hold"]


class PositionState(TypedDict):
    long: int
    short: int
    long_cost_basis: float
    short_cost_basis: float
    short_margin_used: float


class TickerRealizedGains(TypedDict):
    long: float
    short: float


class PortfolioSnapshot(TypedDict):
    cash: float
    margin_used: float
    margin_requirement: float
    positions: Dict[str, PositionState]
    realized_gains: Dict[str, TickerRealizedGains]


PriceDataFrame = pd.DataFrame


class AgentDecision(TypedDict):
    action: ActionLiteral
    quantity: float


AgentDecisions = Dict[str, AgentDecision]

AnalystSignal = Dict[str, Any]
AgentSignals = Dict[str, Dict[str, AnalystSignal]]


class AgentOutput(TypedDict):
    decisions: AgentDecisions
    analyst_signals: AgentSignals


PortfolioValuePoint = TypedDict(
    "PortfolioValuePoint",
    {
        "Date": datetime,
        "Portfolio Value": float,
        "Long Exposure": float,
        "Short Exposure": float,
        "Gross Exposure": float,
        "Net Exposure": float,
        "Long/Short Ratio": float,
    },
    total=False,
)


class PerformanceMetrics(TypedDict, total=False):
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown: Optional[float]
    max_drawdown_date: Optional[str]
    long_short_ratio: Optional[float]
    gross_exposure: Optional[float]
    net_exposure: Optional[float]
