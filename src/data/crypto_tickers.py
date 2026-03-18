from enum import Enum


class AssetType(Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"


# Top 10 crypto assets by market cap mapped to CoinGecko IDs
CRYPTO_SYMBOL_TO_COINGECKO = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "SOL": "solana",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
}

COINGECKO_TO_SYMBOL = {v: k for k, v in CRYPTO_SYMBOL_TO_COINGECKO.items()}


def detect_asset_type(ticker: str) -> AssetType:
    """Detect whether a ticker is an equity or crypto asset."""
    if ticker.upper() in CRYPTO_SYMBOL_TO_COINGECKO:
        return AssetType.CRYPTO
    return AssetType.EQUITY


def get_coingecko_id(ticker: str) -> str | None:
    """Get the CoinGecko ID for a crypto ticker symbol."""
    return CRYPTO_SYMBOL_TO_COINGECKO.get(ticker.upper())
