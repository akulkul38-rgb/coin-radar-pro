import os
import time
import math
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import requests
import pandas as pd
import numpy as np

BINANCE_TR_BASE = os.getenv("BINANCE_TR_BASE", "https://api.binance.me")
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
MIN_TRY_VOLUME = float(os.getenv("MIN_TRY_VOLUME", "10000000"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "78"))
MAX_24H_CHANGE = float(os.getenv("MAX_24H_CHANGE", "12"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "180"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

session = requests.Session()
session.headers.update({"User-Agent": "CoinRadarV1/1.0"})


@dataclass
class Signal:
    symbol: str
    price: float
    score: float
    entry_low: float
    entry_high: float
    stop: float
    target: float
    reasons: List[str]


def get_json(url: str, params: Optional[dict] = None, timeout: int = 15):
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code in (418, 429):
        retry_after = int(r.headers.get("Retry-After", "60"))
        logging.warning("Rate limit. %s saniye bekleniyor.", retry_after)
        time.sleep(retry_after)
        raise RuntimeError("Rate limited")
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and "code" in payload and payload.get("code") not in (0, None):
        raise RuntimeError(payload)
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def get_symbols() -> List[str]:
    # Binance TR official endpoint
    data = get_json("https://www.binance.tr/open/v1/common/symbols")
    symbols = []
    for item in data.get("list", data if isinstance(data, list) else []):
        symbol = item.get("s") or item.get("symbol")
        status = item.get("st") or item.get("status")
        if not symbol:
            continue
        normalized = symbol.replace("_", "")
        if normalized.endswith("TRY") and status in (None, 1, "TRADING"):
            symbols.append(normalized)
    return sorted(set(symbols))


def get_24h_tickers() -> Dict[str, dict]:
    # Binance-compatible public market endpoint
    data = get_json(f"{BINANCE_TR_BASE}/api/v3/ticker/24hr")
    if isinstance(data, dict):
        data = [data]
    return {x["symbol"]: x for x in data if x.get("symbol", "").endswith("TRY")}


def get_klines(symbol: str, interval: str = "15m", limit: int = 100) -> pd.DataFrame:
    data = get_json(
        f"{BINANCE_TR_BASE}/api/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    rows = data if isinstance(data, list) else []
    if len(rows) < 30:
        raise RuntimeError(f"{symbol}: yetersiz mum verisi")
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore"
        ][:len(rows[0])]
    )
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close", "volume"])


def indicators(df: pd.DataFrame) -> dict:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    bandwidth = (upper - lower) / mid.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    vol_avg20 = volume.rolling(20).mean()
    rvol = volume.iloc[-1] / max(vol_avg20.iloc[-1], 1e-12)

    return {
        "close": float(close.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "mid": float(mid.iloc[-1]),
        "upper": float(upper.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "bandwidth": float(bandwidth.iloc[-1]),
        "bandwidth_prev": float(bandwidth.iloc[-6]),
        "rsi": float(rsi.iloc[-1]),
        "atr": float(atr.iloc[-1]),
        "rvol": float(rvol),
        "recent_high": float(high.iloc[-20:].max()),
        "recent_low": float(low.iloc[-20:].min()),
        "last_volume": float(volume.iloc[-1]),
    }


def score_signal(symbol: str, ticker: dict, ind15: dict, ind1h: dict) -> Optional[Signal]:
    price = ind15["close"]
    change24 = float(ticker.get("priceChangePercent", 0))
    quote_volume = float(ticker.get("quoteVolume", 0))

    if quote_volume < MIN_TRY_VOLUME:
        return None
    if change24 > MAX_24H_CHANGE or change24 < -5:
        return None

    score = 0.0
    reasons = []

    # Trend
    if ind15["ema20"] > ind15["ema50"]:
        score += 14
        reasons.append("15 dk trend yukarı")
    if ind1h["ema20"] > ind1h["ema50"]:
        score += 16
        reasons.append("1 saat trend yukarı")

    # Momentum: aşırı alım değil, pozitif bölge
    if 52 <= ind15["rsi"] <= 68:
        score += 14
        reasons.append(f"RSI dengeli ({ind15['rsi']:.0f})")
    elif 45 <= ind15["rsi"] < 52:
        score += 7

    # Hacim
    if 1.4 <= ind15["rvol"] <= 4.0:
        score += 18
        reasons.append(f"hacim artışı {ind15['rvol']:.1f}x")
    elif ind15["rvol"] > 4.0:
        score += 8
        reasons.append("hacim çok sert; takip gerekli")

    # Bollinger sıkışmasından kontrollü açılma
    if ind15["bandwidth"] > ind15["bandwidth_prev"] and price > ind15["mid"]:
        score += 12
        reasons.append("Bollinger açılıyor")
    if ind15["bandwidth_prev"] < 0.08:
        score += 8
        reasons.append("öncesinde sıkışma vardı")

    # Fiyatı zirvede kovalamama
    distance_to_high = (ind15["recent_high"] - price) / max(price, 1e-12)
    if 0.01 <= distance_to_high <= 0.05:
        score += 10
        reasons.append("zirveye yakın ama aşırı uzamamış")
    elif distance_to_high < 0.005:
        score -= 10

    # 24h hareketi kontrollü olsun
    if 1.0 <= change24 <= 7.0:
        score += 8
        reasons.append(f"24s hareket kontrollü %{change24:.1f}")
    elif 7.0 < change24 <= MAX_24H_CHANGE:
        score += 2

    # Entry/stop/target: ATR ve yakın destek
    atr = max(ind15["atr"], price * 0.003)
    entry_low = max(ind15["ema20"], price - 0.35 * atr)
    entry_high = price + 0.15 * atr
    stop = min(entry_low - 1.15 * atr, ind15["mid"] - 0.45 * atr)
    risk = max(entry_high - stop, price * 0.005)
    target = entry_high + 1.8 * risk

    risk_pct = risk / entry_high
    target_pct = (target - entry_high) / entry_high

    if risk_pct <= 0.035:
        score += 8
        reasons.append(f"stop mesafesi %{risk_pct*100:.1f}")
    else:
        score -= 12

    if target_pct < 0.025 or target_pct > 0.09:
        score -= 6

    if score < MIN_SCORE:
        return None

    return Signal(
        symbol=symbol,
        price=price,
        score=round(score, 1),
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        target=target,
        reasons=reasons[:5],
    )


def fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def send_telegram(signal: Signal):
    text = (
        f"🚨 COIN RADAR\n\n"
        f"Coin: {signal.symbol.replace('TRY', '/TRY')}\n"
        f"Puan: {signal.score}/100\n"
        f"Fiyat: {fmt_price(signal.price)}\n\n"
        f"Giriş: {fmt_price(signal.entry_low)} - {fmt_price(signal.entry_high)}\n"
        f"Stop: {fmt_price(signal.stop)}\n"
        f"Kâr alma: {fmt_price(signal.target)}\n"
        f"Maksimum bekleme: 6-8 saat\n\n"
        f"Neden: " + ", ".join(signal.reasons) +
        "\n\n⚠️ Bu otomatik ön elemedir; işlem açmadan önce son kontrol yap."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    r.raise_for_status()


def load_state() -> dict:
    try:
        with open("state.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict):
    with open("state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)


def run_scan():
    symbols = get_symbols()
    tickers = get_24h_tickers()
    state = load_state()
    now = time.time()

    candidates: List[Signal] = []
    for symbol in symbols:
        try:
            ticker = tickers.get(symbol)
            if not ticker:
                continue
            df15 = get_klines(symbol, "15m", 100)
            df1h = get_klines(symbol, "1h", 100)
            signal = score_signal(symbol, ticker, indicators(df15), indicators(df1h))
            if signal:
                candidates.append(signal)
            time.sleep(0.08)
        except Exception as exc:
            logging.debug("%s atlandı: %s", symbol, exc)

    candidates.sort(key=lambda x: x.score, reverse=True)

    for signal in candidates[:3]:
        last_sent = float(state.get(signal.symbol, 0))
        if now - last_sent < COOLDOWN_MINUTES * 60:
            continue
        send_telegram(signal)
        state[signal.symbol] = now
        save_state(state)
        logging.info("Sinyal gönderildi: %s %.1f", signal.symbol, signal.score)
        break

    if not candidates:
        logging.info("Uygun sinyal yok.")


def main():
    logging.info("Coin Radar başladı.")
    while True:
        started = time.time()
        try:
            run_scan()
        except Exception:
            logging.exception("Tarama hatası")
        elapsed = time.time() - started
        time.sleep(max(5, SCAN_INTERVAL_SECONDS - elapsed))


if __name__ == "__main__":
    main()
