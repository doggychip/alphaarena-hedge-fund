"""FastAPI signal server for AlphaArena hedge fund engine."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Import hedge fund runner (after dotenv is loaded)
from src.main import run_hedge_fund  # noqa: E402

app = FastAPI(title="AlphaArena Signal Server", version="1.0.0")

# In-memory cache for latest analysis results
signal_cache: dict = {}

# Configuration
TICKERS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
MODEL_NAME = "deepseek-chat"
MODEL_PROVIDER = "DeepSeek"
CYCLE_SECONDS = 7200  # 2 hours


def _build_empty_portfolio(tickers: list[str]) -> dict:
    """Build an empty portfolio structure for the given tickers."""
    return {
        "cash": 100_000.0,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {"long": 0.0, "short": 0.0}
            for ticker in tickers
        },
    }


def _count_signals() -> int:
    """Count the total number of individual ticker signals across all analysts."""
    analyst_signals = signal_cache.get("analyst_signals", {})
    count = 0
    for ticker_signals in analyst_signals.values():
        count += len(ticker_signals)
    return count


def _run_analysis(run_id: str) -> None:
    """Execute one full analysis cycle and store results in signal_cache."""
    logger.info(f"[run_id={run_id}] Starting analysis cycle for tickers: {TICKERS}")
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - relativedelta(months=3)

        result = run_hedge_fund(
            tickers=TICKERS,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            portfolio=_build_empty_portfolio(TICKERS),
            show_reasoning=True,
            selected_analysts=[],  # empty → all analysts
            model_name=MODEL_NAME,
            model_provider=MODEL_PROVIDER,
        )

        signal_cache.update(
            {
                "run_id": run_id,
                "run_at": datetime.now(timezone.utc).isoformat(),
                "tickers": TICKERS,
                "analyst_signals": result.get("analyst_signals", {}),
                "decisions": result.get("decisions") or {},
            }
        )
        logger.info(f"[run_id={run_id}] Analysis complete. Signals stored: {_count_signals()}")
    except Exception as exc:
        logger.error(f"[run_id={run_id}] Analysis failed: {exc}", exc_info=True)
        # Keep previous cache intact on failure


async def _background_loop() -> None:
    """Background coroutine: run analysis on startup and repeat every CYCLE_SECONDS."""
    while True:
        run_id = str(uuid.uuid4())
        # Run in a thread pool so the blocking LangGraph call doesn't block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_analysis, run_id)
        logger.info(f"Next analysis cycle in {CYCLE_SECONDS} seconds.")
        await asyncio.sleep(CYCLE_SECONDS)


@app.on_event("startup")
async def startup_event() -> None:
    """Schedule the background analysis loop on server startup."""
    logger.info("Server startup: scheduling background analysis loop.")
    asyncio.create_task(_background_loop())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "last_run": signal_cache.get("run_at"),
        "signal_count": _count_signals(),
    }


@app.get("/signals/latest")
async def signals_latest():
    """Return the latest analysis results."""
    if not signal_cache:
        return {"status": "warming_up", "analyst_signals": {}}
    return signal_cache


@app.post("/run")
async def trigger_run():
    """Trigger an immediate analysis cycle in the background."""
    run_id = str(uuid.uuid4())
    logger.info(f"Manual /run triggered. run_id={run_id}")
    loop = asyncio.get_event_loop()
    asyncio.create_task(
        loop.run_in_executor(None, _run_analysis, run_id)
    )
    return {"status": "started", "run_id": run_id}
