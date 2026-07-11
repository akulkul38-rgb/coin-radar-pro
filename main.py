import os
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

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

# 0 = taramadan geçen bütün sinyalleri gönder.
MAX_SIGNALS_PER_SCAN = int(os.getenv("MAX_SIGNALS_PER_SCAN", "0"))

SIGNAL_MAX_AGE_HOURS = float(os.getenv("SIGNAL_MAX_AGE_HOURS", "8"))
ORDERBOOK_LIMIT = int(os.getenv("ORDERBOOK_LIMIT", "100"))
STATUS_UPDATE_MINUTES = int(os.getenv("STATUS_UPDATE_MINUTES", "5"))

STATE_FILE = os.getenv("STATE_FILE", "state.json")
HISTORY_FILE = os.getenv("HISTORY_FILE", "signal_history.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

session = requests.Session()
session.headers.update({"User-Agent": "CoinRadarPro/2.0"})


@dataclass
class Signal:
    symbol: str
    price: float
    score: float
    grade: str
    entry_low: float
    entry_high: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    reasons: List[str]
    rvol: float
    orderbook_ratio: float
    btc_alignment: str
    created_at: float = 0.0


def get_json(url: str, params: Optional[dict] = None, timeout: int = 15):
    response = session.get(url, params=params, timeout=timeout)

    if response.status_code in (418, 429):
        retry_after = int(response.headers.get("Retry-After", "60"))
        logging.warning("Rate limit. %s saniye bekleniyor.", retry_after)
        time.sleep(retry_after)
        raise RuntimeError("Rate limited")

    response.raise_for_status()
    payload = response.json()

    if (
        isinstance(payload, dict)
        and "code" in payload
        and payload.get("code") not in (0, None)
    ):
        raise RuntimeError(payload)

    return payload.get("data", payload) if isinstance(payload, dict) else payload


def get_symbols() -> List[str]:
    data = get_json("https://www.binance.tr/open/v1/common/symbols")
    items = data.get("list", data if isinstance(data, list) else [])

    symbols: List[str] = []
    for item in items:
        symbol = item.get("s") or item.get("symbol")
        status = item.get("st") or item.get("status")

        if not symbol:
            continue

        normalized = symbol.replace("_", "")
        if normalized.endswith("TRY") and status in (None, 1, "TRADING"):
            symbols.append(normalized)

    return sorted(set(symbols))


def get_24h_tickers() -> Dict[str, dict]:
    data = get_json(f"{BINANCE_TR_BASE}/api/v3/ticker/24hr")
    if isinstance(data, dict):
        data = [data]

    return {
        item["symbol"]: item
        for item in data
        if item.get("symbol", "").endswith("TRY")
    }


def get_klines(
    symbol: str,
    interval: str = "15m",
    limit: int = 100
) -> pd.DataFrame:
    data = get_json(
        f"{BINANCE_TR_BASE}/api/v1/klines",
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )

    rows = data if isinstance(data, list) else []
    if len(rows) < 30:
        raise RuntimeError(f"{symbol}: yetersiz mum verisi")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore",
    ][:len(rows[0])]

    df = pd.DataFrame(rows, columns=columns)

    for column in [
        "open", "high", "low", "close", "volume", "quote_volume"
    ]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

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

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean()

    vol_avg20 = volume.rolling(20).mean()
    rvol = volume.iloc[-1] / max(vol_avg20.iloc[-1], 1e-12)

    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd = macd_fast - macd_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    return {
        "close": float(close.iloc[-1]),
        "previous_close": float(close.iloc[-2]),
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
        "macd_hist": float(macd_hist.iloc[-1]),
        "macd_hist_prev": float(macd_hist.iloc[-2]),
    }


def get_orderbook_ratio(symbol: str) -> float:
    """
    İlk ORDERBOOK_LIMIT kademe içindeki toplam alış tutarını
    toplam satış tutarına böler.

    1.00 = dengeli
    >1.10 = alış tarafı güçlü
    <0.90 = satış tarafı güçlü
    """
    data = get_json(
        f"{BINANCE_TR_BASE}/api/v3/depth",
        params={"symbol": symbol, "limit": ORDERBOOK_LIMIT},
    )

    bids = data.get("bids", [])
    asks = data.get("asks", [])

    bid_value = sum(float(price) * float(amount) for price, amount in bids)
    ask_value = sum(float(price) * float(amount) for price, amount in asks)

    return bid_value / max(ask_value, 1e-12)


def get_btc_context(
    tickers: Dict[str, dict]
) -> Tuple[str, float]:
    """
    BTC/TRY mevcutsa 15 dk + 1 saat eğilimini değerlendirir.
    Sonuç: (metin, puan etkisi)
    """
    symbol = "BTCTRY"
    if symbol not in tickers:
        return "Nötr (BTC/TRY bulunamadı)", 0.0

    try:
        ind15 = indicators(get_klines(symbol, "15m", 100))
        ind1h = indicators(get_klines(symbol, "1h", 100))

        bullish_15 = ind15["ema20"] > ind15["ema50"]
        bullish_1h = ind1h["ema20"] > ind1h["ema50"]
        price_above_mid = ind15["close"] > ind15["mid"]

        if bullish_15 and bullish_1h and price_above_mid:
            return "Olumlu", 6.0

        if not bullish_15 and not bullish_1h and not price_above_mid:
            return "Olumsuz", -10.0

        return "Karışık", -2.0
    except Exception as exc:
        logging.warning("BTC bağlamı alınamadı: %s", exc)
        return "Nötr", 0.0


def grade_for_score(score: float) -> str:
    if score >= 97:
        return "S"
    if score >= 93:
        return "A+"
    if score >= 88:
        return "A"
    if score >= 83:
        return "B"
    return "C"


def score_signal_base(
    symbol: str,
    ticker: dict,
    ind15: dict,
    ind1h: dict,
    btc_alignment: str,
    btc_score: float,
) -> Optional[Signal]:
    price = ind15["close"]
    change24 = float(ticker.get("priceChangePercent", 0))
    quote_volume = float(ticker.get("quoteVolume", 0))

    if quote_volume < MIN_TRY_VOLUME:
        return None

    if change24 > MAX_24H_CHANGE or change24 < -5:
        return None

    score = btc_score
    reasons: List[str] = []

    if btc_score > 0:
        reasons.append("BTC yönü uyumlu")
    elif btc_score < 0:
        reasons.append("BTC yönü tam uyumlu değil")

    # Trend
    if ind15["ema20"] > ind15["ema50"]:
        score += 14
        reasons.append("15 dk trend yukarı")

    if ind1h["ema20"] > ind1h["ema50"]:
        score += 16
        reasons.append("1 saat trend yukarı")

    # Momentum
    if 52 <= ind15["rsi"] <= 68:
        score += 14
        reasons.append(f"RSI dengeli ({ind15['rsi']:.0f})")
    elif 45 <= ind15["rsi"] < 52:
        score += 7
    elif ind15["rsi"] > 72:
        score -= 10
        reasons.append("RSI aşırı alıma yakın")

    # RVOL
    if 1.4 <= ind15["rvol"] <= 4.0:
        score += 18
        reasons.append(f"RVOL {ind15['rvol']:.1f}x")
    elif ind15["rvol"] > 4.0:
        score += 8
        reasons.append("Hacim çok sert; FOMO riski")
    elif ind15["rvol"] < 0.8:
        score -= 6

    # Bollinger
    if (
        ind15["bandwidth"] > ind15["bandwidth_prev"]
        and price > ind15["mid"]
    ):
        score += 12
        reasons.append("Bollinger açılıyor")

    if ind15["bandwidth_prev"] < 0.08:
        score += 8
        reasons.append("Öncesinde sıkışma vardı")

    # Zirveyi kovalamama
    distance_to_high = (
        ind15["recent_high"] - price
    ) / max(price, 1e-12)

    if 0.01 <= distance_to_high <= 0.05:
        score += 10
        reasons.append("Zirveye yakın ama uzamamış")
    elif distance_to_high < 0.005:
        score -= 10
        reasons.append("Yerel zirveye fazla yakın")

    # FOMO cezası: EMA20'den fazla uzaklaşma
    extension_from_ema20 = (
        price - ind15["ema20"]
    ) / max(ind15["ema20"], 1e-12)

    if extension_from_ema20 > 0.045:
        score -= 12
        reasons.append("EMA20 üstünde fazla uzamış")
    elif extension_from_ema20 > 0.03:
        score -= 6

    # MACD momentumu
    if ind15["macd_hist"] > 0 and ind15["macd_hist"] >= ind15["macd_hist_prev"]:
        score += 6
        reasons.append("MACD momentumu güçleniyor")
    elif ind15["macd_hist"] < ind15["macd_hist_prev"]:
        score -= 5

    # 24 saatlik hareket
    if 1.0 <= change24 <= 7.0:
        score += 8
        reasons.append(f"24s hareket kontrollü %{change24:.1f}")
    elif 7.0 < change24 <= MAX_24H_CHANGE:
        score += 2

    # Entry / stop / hedefler
    atr = max(ind15["atr"], price * 0.003)
    entry_low = max(ind15["ema20"], price - 0.35 * atr)
    entry_high = price + 0.15 * atr
    stop = min(
        entry_low - 1.15 * atr,
        ind15["mid"] - 0.45 * atr,
    )

    risk = max(entry_high - stop, price * 0.005)
    risk_pct = risk / max(entry_high, 1e-12)

    tp1 = entry_high + 1.00 * risk
    tp2 = entry_high + 1.50 * risk
    tp3 = entry_high + 2.00 * risk

    if risk_pct <= 0.035:
        score += 8
        reasons.append(f"Stop mesafesi %{risk_pct * 100:.1f}")
    else:
        score -= 12

    target_pct = (tp3 - entry_high) / max(entry_high, 1e-12)
    if target_pct < 0.025 or target_pct > 0.12:
        score -= 6

    return Signal(
        symbol=symbol,
        price=price,
        score=round(score, 1),
        grade=grade_for_score(score),
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        reasons=reasons,
        rvol=ind15["rvol"],
        orderbook_ratio=1.0,
        btc_alignment=btc_alignment,
        created_at=time.time(),
    )


def enrich_with_orderbook(signal: Signal) -> Signal:
    try:
        ratio = get_orderbook_ratio(signal.symbol)
        signal.orderbook_ratio = ratio

        if ratio >= 1.30:
            signal.score += 8
            signal.reasons.append(f"Emir defteri alış güçlü ({ratio:.2f}x)")
        elif ratio >= 1.10:
            signal.score += 4
            signal.reasons.append(f"Alış baskısı olumlu ({ratio:.2f}x)")
        elif ratio <= 0.70:
            signal.score -= 10
            signal.reasons.append(f"Satış baskısı yüksek ({ratio:.2f}x)")
        elif ratio <= 0.90:
            signal.score -= 5
            signal.reasons.append(f"Satış tarafı ağır ({ratio:.2f}x)")

        signal.score = round(signal.score, 1)
        signal.grade = grade_for_score(signal.score)
    except Exception as exc:
        logging.debug(
            "%s emir defteri alınamadı: %s",
            signal.symbol,
            exc,
        )

    return signal


def fmt_price(value: float) -> str:
    if value >= 100:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def send_telegram_text(text: str):
    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )
    response = session.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
        },
        timeout=15,
    )
    response.raise_for_status()


def send_signal(signal: Signal):
    text = (
        "🚨 COIN RADAR PRO\n\n"
        f"Coin: {signal.symbol.replace('TRY', '/TRY')}\n"
        f"Puan: {signal.score}/100 — Sınıf {signal.grade}\n"
        f"Fiyat: {fmt_price(signal.price)}\n\n"
        f"Giriş: {fmt_price(signal.entry_low)} - "
        f"{fmt_price(signal.entry_high)}\n"
        f"Stop: {fmt_price(signal.stop)}\n"
        f"TP1 (%30): {fmt_price(signal.tp1)}\n"
        f"TP2 (%30): {fmt_price(signal.tp2)}\n"
        f"TP3 (%40): {fmt_price(signal.tp3)}\n"
        f"Maksimum bekleme: "
        f"{SIGNAL_MAX_AGE_HOURS:.0f} saat\n\n"
        f"📊 RVOL: {signal.rvol:.2f}x\n"
        f"📚 Alış/Satış baskısı: "
        f"{signal.orderbook_ratio:.2f}x\n"
        f"₿ BTC uyumu: {signal.btc_alignment}\n"
        f"🟢 Durum: AKTİF\n\n"
        "Neden: "
        + ", ".join(signal.reasons[:8])
        + "\n\n⚠️ Otomatik ön elemedir; emir vermeden önce son kontrol yap."
    )
    send_telegram_text(text)


def send_status_update(
    symbol: str,
    status: str,
    price: float,
    reason: str,
):
    emoji = {
        "ACTIVE": "🟢",
        "WAIT": "🟡",
        "CANCELLED": "🔴",
        "TP1": "✅",
        "TP2": "✅",
        "TP3": "🏁",
        "STOP": "🛑",
        "EXPIRED": "⌛",
    }.get(status, "ℹ️")

    text = (
        f"{emoji} SİNYAL GÜNCELLEMESİ\n\n"
        f"Coin: {symbol.replace('TRY', '/TRY')}\n"
        f"Durum: {status}\n"
        f"Fiyat: {fmt_price(price)}\n"
        f"Neden: {reason}"
    )
    send_telegram_text(text)


def default_state() -> dict:
    return {
        "last_sent": {},
        "active_signals": {},
    }


def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            raw = json.load(file)

        # Eski sürümde state doğrudan {"COINTRY": timestamp} idi.
        if (
            isinstance(raw, dict)
            and "last_sent" not in raw
            and "active_signals" not in raw
        ):
            return {
                "last_sent": raw,
                "active_signals": {},
            }

        state = default_state()
        state.update(raw)
        state.setdefault("last_sent", {})
        state.setdefault("active_signals", {})
        return state
    except Exception:
        return default_state()


def save_state(state: dict):
    temp_path = f"{STATE_FILE}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, STATE_FILE)


def append_history(record: dict):
    with open(HISTORY_FILE, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def signal_status(
    signal_data: dict,
    ind15: dict,
    now: float,
) -> Tuple[str, str]:
    price = ind15["close"]
    created_at = float(signal_data["created_at"])
    age_hours = (now - created_at) / 3600

    stop = float(signal_data["stop"])
    tp1 = float(signal_data["tp1"])
    tp2 = float(signal_data["tp2"])
    tp3 = float(signal_data["tp3"])
    entry_low = float(signal_data["entry_low"])
    entry_high = float(signal_data["entry_high"])

    if price <= stop:
        return "STOP", "Stop seviyesi görüldü"

    if price >= tp3:
        return "TP3", "Üçüncü hedef görüldü"

    if price >= tp2:
        return "TP2", "İkinci hedef görüldü"

    if price >= tp1:
        return "TP1", "İlk hedef görüldü"

    if age_hours >= SIGNAL_MAX_AGE_HOURS:
        return "EXPIRED", "Maksimum bekleme süresi doldu"

    trend_broken = (
        ind15["ema20"] < ind15["ema50"]
        and price < ind15["mid"]
    )
    momentum_broken = (
        ind15["macd_hist"] < 0
        and ind15["macd_hist"] < ind15["macd_hist_prev"]
        and ind15["rsi"] < 46
    )

    if trend_broken or momentum_broken:
        return "CANCELLED", "Trend veya momentum yapısı bozuldu"

    if entry_low <= price <= entry_high:
        return "ACTIVE", "Fiyat giriş bölgesinde"

    if stop < price < entry_low:
        return "WAIT", "Fiyat giriş bölgesinin altında; tepki teyidi bekleniyor"

    if entry_high < price < tp1:
        return "WAIT", "Fiyat giriş bölgesinin üstünde; kovalamamak daha güvenli"

    return "WAIT", "Yeni teyit bekleniyor"


def monitor_active_signals(state: dict, now: float):
    active_signals = state.get("active_signals", {})
    to_remove: List[str] = []

    for symbol, signal_data in list(active_signals.items()):
        try:
            ind15 = indicators(
                get_klines(symbol, "15m", 100)
            )
            status, reason = signal_status(
                signal_data,
                ind15,
                now,
            )

            previous_status = signal_data.get(
                "status",
                "ACTIVE",
            )
            last_update = float(
                signal_data.get("last_status_update", 0)
            )

            terminal = status in {
                "STOP",
                "TP3",
                "CANCELLED",
                "EXPIRED",
            }

            meaningful_change = status != previous_status
            update_due = (
                now - last_update
                >= STATUS_UPDATE_MINUTES * 60
            )

            # Durum değiştiğinde bildir. Aynı WAIT/ACTIVE durumunu
            # sürekli mesaj yağmuruna çevirmemek için tekrar gönderme.
            if meaningful_change and update_due:
                send_status_update(
                    symbol,
                    status,
                    ind15["close"],
                    reason,
                )
                signal_data["status"] = status
                signal_data["last_status_update"] = now

            # TP1 ve TP2 görüldüğünde sinyal açık kalır.
            if status in {"TP1", "TP2"}:
                highest = signal_data.get(
                    "highest_target",
                    "",
                )
                if highest != status:
                    send_status_update(
                        symbol,
                        status,
                        ind15["close"],
                        reason,
                    )
                    signal_data["highest_target"] = status
                    signal_data["last_status_update"] = now

            if terminal:
                append_history(
                    {
                        "symbol": symbol,
                        "created_at": signal_data["created_at"],
                        "closed_at": now,
                        "result": status,
                        "close_price": ind15["close"],
                        "signal": signal_data,
                    }
                )
                to_remove.append(symbol)

            time.sleep(0.05)

        except Exception as exc:
            logging.debug(
                "%s aktif sinyal izleme hatası: %s",
                symbol,
                exc,
            )

    for symbol in to_remove:
        active_signals.pop(symbol, None)

    state["active_signals"] = active_signals


def run_scan():
    symbols = get_symbols()
    tickers = get_24h_tickers()
    state = load_state()
    now = time.time()

    monitor_active_signals(state, now)

    btc_alignment, btc_score = get_btc_context(
        tickers
    )

    base_candidates: List[Signal] = []

    for symbol in symbols:
        try:
            ticker = tickers.get(symbol)
            if not ticker:
                continue

            df15 = get_klines(symbol, "15m", 100)
            df1h = get_klines(symbol, "1h", 100)

            signal = score_signal_base(
                symbol=symbol,
                ticker=ticker,
                ind15=indicators(df15),
                ind1h=indicators(df1h),
                btc_alignment=btc_alignment,
                btc_score=btc_score,
            )

            # Emir defteri çağrısını sadece temel puanı yakın olan
            # coinlerde yaparak API yükünü azalt.
            if signal and signal.score >= MIN_SCORE - 10:
                base_candidates.append(signal)

            time.sleep(0.08)

        except Exception as exc:
            logging.debug(
                "%s atlandı: %s",
                symbol,
                exc,
            )

    candidates: List[Signal] = []

    for signal in base_candidates:
        signal = enrich_with_orderbook(signal)
        if signal.score >= MIN_SCORE:
            candidates.append(signal)
        time.sleep(0.08)

    candidates.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    if MAX_SIGNALS_PER_SCAN > 0:
        candidates = candidates[:MAX_SIGNALS_PER_SCAN]

    sent_count = 0

    for signal in candidates:
        last_sent = float(
            state["last_sent"].get(
                signal.symbol,
                0,
            )
        )

        if (
            now - last_sent
            < COOLDOWN_MINUTES * 60
        ):
            continue

        # Aynı coin hâlihazırda aktif izleniyorsa yeni sinyal gönderme.
        if signal.symbol in state["active_signals"]:
            continue

        send_signal(signal)

        state["last_sent"][signal.symbol] = now

        signal.created_at = now
        signal_data = asdict(signal)
        signal_data["status"] = "ACTIVE"
        signal_data["last_status_update"] = now
        signal_data["highest_target"] = ""

        state["active_signals"][
            signal.symbol
        ] = signal_data

        save_state(state)

        sent_count += 1
        logging.info(
            "Sinyal gönderildi: %s %.1f",
            signal.symbol,
            signal.score,
        )

    save_state(state)

    if not candidates:
        logging.info("Uygun sinyal yok.")
    else:
        logging.info(
            "%s aday bulundu, %s yeni sinyal gönderildi.",
            len(candidates),
            sent_count,
        )


def main():
    logging.info("Coin Radar Pro v2 başladı.")

    while True:
        started = time.time()

        try:
            run_scan()
        except Exception:
            logging.exception("Tarama hatası")

        elapsed = time.time() - started
        time.sleep(
            max(
                5,
                SCAN_INTERVAL_SECONDS - elapsed,
            )
        )


if __name__ == "__main__":
    main()
