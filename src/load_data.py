"""Загрузка сырых данных для проекта.

Источники:
  - Binance: дневные свечи BTCUSDT и ETHUSDT (с августа 2017);
  - alternative.me: Crypto Fear & Greed Index (с февраля 2018);
  - Yahoo Finance: S&P 500 (^GSPC) и индекс доллара (DX-Y.NYB).

Файлы сохраняются в data/raw/ без какой-либо очистки: обработка пропусков
и ошибок выполняется и обосновывается в notebooks/EDA.ipynb.

Запуск из корня репозитория: python src/load_data.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
FNG_URL = "https://api.alternative.me/fng/"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
START_DATE = "2017-08-17"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
PRICE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote",
]


def fetch_binance_daily(symbol: str, start: str = START_DATE) -> pd.DataFrame:
    """Скачивает все завершённые дневные свечи пары постранично."""
    now_ms = int(time.time() * 1000)
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list[list] = []
    while cursor < now_ms:
        response = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": "1d", "startTime": cursor, "limit": 1000},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 86_400_000
        time.sleep(0.2)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS).drop(columns="ignore")
    # Текущая свеча ещё формируется: её close — не цена закрытия дня.
    df = df[df["close_time"] < now_ms].copy()
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df[["date", *PRICE_COLUMNS, "trades"]]


def fetch_fear_greed() -> pd.DataFrame:
    """Скачивает полную историю Fear & Greed Index."""
    response = requests.get(FNG_URL, params={"limit": 0, "format": "json"}, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["fng_value"] = df["value"].astype(int)
    df = df.rename(columns={"value_classification": "fng_class"})
    return df[["date", "fng_value", "fng_class"]].sort_values("date").reset_index(drop=True)


def fetch_yahoo_close(ticker: str, column: str, start: str = START_DATE) -> pd.DataFrame:
    """Скачивает дневные цены закрытия инструмента с Yahoo Finance."""
    response = requests.get(
        YAHOO_URL.format(ticker=ticker),
        params={
            "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
            "period2": int(time.time()),
            "interval": "1d",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    # Метка времени Yahoo — открытие сессии биржи, дата берётся в часовом поясе биржи.
    timezone = result["meta"]["exchangeTimezoneName"]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(timezone)
    df = pd.DataFrame({
        "date": dates.tz_localize(None).normalize(),
        column: result["indicators"]["quote"][0]["close"],
    })
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        "btcusdt_1d": fetch_binance_daily("BTCUSDT"),
        "ethusdt_1d": fetch_binance_daily("ETHUSDT"),
        "fear_greed": fetch_fear_greed(),
        "sp500": fetch_yahoo_close("^GSPC", "sp500_close"),
        "dxy": fetch_yahoo_close("DX-Y.NYB", "dxy_close"),
    }
    for name, frame in frames.items():
        path = RAW_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"{name:12s} {len(frame):5d} строк  "
              f"{frame['date'].min():%Y-%m-%d} — {frame['date'].max():%Y-%m-%d}  -> {path.name}")


if __name__ == "__main__":
    main()
