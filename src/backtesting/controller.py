from __future__ import annotations

from typing import Callable, Sequence, Dict, Any

from .types import AgentOutput, AgentDecisions, PortfolioSnapshot, ActionLiteral, Action
from .portfolio import Portfolio


class AgentController:
    def run_agent(
        self,
        agent: Callable[..., AgentOutput],
        *,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        portfolio: Portfolio | PortfolioSnapshot,
        model_name: str,
        model_provider: str,
        selected_analysts: Sequence[str] | None,
    ) -> AgentOutput:
        if isinstance(portfolio, Portfolio):
            portfolio_payload: PortfolioSnapshot = portfolio.get_snapshot()
        else:
            portfolio_payload = portfolio

        output = agent(
            tickers=list(tickers),
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_payload,
            model_name=model_name,
            model_provider=model_provider,
            selected_analysts=list(selected_analysts) if selected_analysts is not None else None,
        )

        decisions_in: Dict[str, Any] = dict(output.get("decisions", {})) if isinstance(output, dict) else {}
        analyst_signals_in: Dict[str, Any] = dict(output.get("analyst_signals", {})) if isinstance(output, dict) else {}

        normalized_decisions: AgentDecisions = {}
        for ticker in tickers:
            d = decisions_in.get(ticker, {})
            action = d.get("action", "hold")
            qty = d.get("quantity", 0)
            try:
                qty_val = float(qty)
            except Exception:
                qty_val = 0.0
            try:
                action = Action(action).value
            except Exception:
                action = Action.HOLD.value
            normalized_decisions[ticker] = {"action": action, "quantity": qty_val}

        return {"decisions": normalized_decisions, "analyst_signals": analyst_signals_in}
