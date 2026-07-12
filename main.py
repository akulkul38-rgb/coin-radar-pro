"""
Coin Radar Stable v1
--------------------
Binance TR TRY paritelerini tarar, Telegram'a aday gönderir, kâğıt işlem
sonuçlarını kaydeder ve geçmiş performansa göre sınırlı/temkinli öğrenme yapar.

ÖNEMLİ:
- Otomatik alım-satım yapmaz.
- Borsa API anahtarı istemez.
- Günlük sinyal sayısı sınırı yoktur.
- Aynı kurulumun tekrar tekrar gönderilmesini cooldown ile engeller.
- Öğrenme katmanı temel risk kurallarını değiştiremez.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "evet"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    telegram_chat_id: str

    scan_interval_seconds: int
    request_timeout_seconds: int
    request_pause_seconds: float
    max_prefilter_symbols: int

    min_score: float
    min_24h_try_volume: float
    min_24h_change: float
    max_24h_change: float
    max_spread_percent: float
    max_stop_percent: float
    min_risk_reward: float
    preferred_target_percent: float
    min_target_percent: float
    max_target_percent: float
    max_hold_hours: int

    cooldown_minutes: int
    resignal_score_improvement: float

    btc_filter_enabled: bool
    send_startup_message: bool
    heartbeat_minutes: int
    performance_report_hours: int

    learning_enabled: bool
    learning_db_path: str
    learning_min_samples: int
    learning_max_bonus: float
    learning_review_minutes: int

    dry_run: bool
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        dry_run = env_bool("DRY_RUN", False)
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

        if not dry_run and (not token or not chat_id):
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID Railway Variables "
                "bölümünde tanımlanmalıdır."
            )

        return cls(
            telegram_token=token,
            telegram_chat_id=chat_id,
            scan_interval_seconds=max(60, env_int("SCAN_INTERVAL_SECONDS", 300)),
            request_timeout_seconds=max(5, env_int("REQUEST_TIMEOUT_SECONDS", 12)),
            request_pause_seconds=max(0.03, env_float("REQUEST_PAUSE_SECONDS", 0.08)),
            max_prefilter_symbols=max(10, env_int("MAX_PREFILTER_SYMBOLS", 50)),

            min_score=env_float("MIN_SCORE", 80),
            min_24h_try_volume=env_float("MIN_24H_TRY_VOLUME", 15_000_000),
            min_24h_change=env_float("MIN_24H_CHANGE", -3),
            max_24h_change=env_float("MAX_24H_CHANGE", 14),
            max_spread_percent=env_float("MAX_SPREAD_PERCENT", 0.70),
            max_stop_percent=env_float("MAX_STOP_PERCENT", 3.20),
            min_risk_reward=env_float("MIN_RISK_REWARD", 1.55),
            preferred_target_percent=env_float("PREFERRED_TARGET_PERCENT", 4.0),
            min_target_percent=env_float("MIN_TARGET_PERCENT", 2.2),
            max_target_percent=env_float("MAX_TARGET_PERCENT", 7.0),
            max_hold_hours=max(1, env_int("MAX_HOLD_HOURS", 8)),

            cooldown_minutes=max(15, env_int("COOLDOWN_MINUTES", 120)),
            resignal_score_improvement=env_float("RESIGNAL_SCORE_IMPROVEMENT", 5),

            btc_filter_enabled=env_bool("BTC_FILTER_ENABLED", True),
            send_startup_message=env_bool("SEND_STARTUP_MESSAGE", True),
            heartbeat_minutes=max(5, env_int("HEARTBEAT_MINUTES", 30)),
            performance_report_hours=max(1, env_int("PERFORMANCE_REPORT_HOURS", 24)),

            learning_enabled=env_bool("LEARNING_ENABLED", True),
            learning_db_path=os.getenv("LEARNING_DB_PATH", "/data/radar_learning.db"),
            learning_min_samples=max(5, env_int("LEARNING_MIN_SAMPLES", 10)),
            learning_max_bonus=env_float("LEARNING_MAX_BONUS", 10.0),
            learning_review_minutes=max(5, env_int("LEARNING_REVIEW_MINUTES", 15)),

            dry_run=dry_run,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


# ============================================================
# MODELS
# ============================================================

@dataclass(frozen=True)
class Metrics:
    price: float
    previous_close: float
    ema20: float
    ema50: float
    ema200: float
    rsi14: float
    macd_hist: float
    macd_hist_previous: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    bb_width: float
    bb_width_previous: float
    atr14: float
    atr_percent: float
    rvol: float
    quote_rvol: float
    volume_trend: float
    recent_high_20: float
    recent_low_20: float
    prior_high_20: float
    upper_wick_ratio: float
    lower_wick_ratio: float


@dataclass(frozen=True)
class MarketContext:
    btc_symbol: str
    btc_price: float
    btc_change_24h: float
    btc_15m_ok: bool
    btc_1h_ok: bool
    risk_off: bool


@dataclass
class Signal:
    symbol: str
    score: float
    raw_score: float
    learning_bonus: float
    price: float
    entry_low: float
    entry_high: float
    stop: float
    target: float
    stop_percent: float
    target_percent: float
    risk_reward: float
    change_24h: float
    quote_volume_24h: float
    spread_percent: float
    max_hold_hours: int
    reasons: List[str]
    warnings: List[str]
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ============================================================
# HTTP / EXCHANGE
# ============================================================

class HttpClient:
    def __init__(self, timeout: int):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CoinRadarStable/1.0",
            "Accept": "application/json",
            "Connection": "keep-alive",
        })

    def get_json(
        self,
        url: str,
        params: Optional[dict] = None,
        attempts: int = 3,
    ) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {418, 429}:
                    retry_after = int(response.headers.get("Retry-After", "20"))
                    raise RuntimeError(f"Rate limit; retry_after={retry_after}")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("code") not in (None, 0, "0"):
                    raise RuntimeError(str(payload))
                return payload.get("data", payload) if isinstance(payload, dict) else payload
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    delay = min(15.0, (2 ** (attempt - 1)) + random.random())
                    logging.warning(
                        "HTTP %s/%s başarısız: %s; %.1fs bekleniyor",
                        attempt, attempts, exc, delay
                    )
                    time.sleep(delay)
        raise RuntimeError(f"GET başarısız: {url}: {last_error}")

    def post_json(self, url: str, payload: dict, attempts: int = 3) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("ok") is False:
                    raise RuntimeError(data.get("description", str(data)))
                return data
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(10, 2 ** attempt))
        raise RuntimeError(f"POST başarısız: {last_error}")


class BinanceTRClient:
    """
    Public market endpoints only.

    Symbol discovery uses market-wide ticker data. This avoids the cloud-region
    HTTP 451 problem that can affect the common/symbols endpoint.
    """

    MARKET_BASES = (
        "https://api.binance.me",
        "https://cloudme-tr.2meta.app",
    )

    def __init__(self, http: HttpClient):
        self.http = http
        self._base_cache: Dict[str, str] = {}

    def _get_from_bases(
        self,
        cache_key: str,
        paths: Iterable[str],
        params: Optional[dict] = None,
    ) -> Any:
        bases: List[str] = []
        cached = self._base_cache.get(cache_key)
        if cached:
            bases.append(cached)
        bases.extend(x for x in self.MARKET_BASES if x not in bases)

        errors: List[str] = []
        for base in bases:
            for path in paths:
                try:
                    result = self.http.get_json(f"{base}{path}", params=params)
                    self._base_cache[cache_key] = base
                    return result
                except Exception as exc:
                    errors.append(f"{base}{path}: {exc}")
        raise RuntimeError(" | ".join(errors[-4:]))

    def tickers_24h(self) -> Dict[str, dict]:
        payload = self._get_from_bases(
            "ticker24",
            ("/api/v3/ticker/24hr", "/api/v1/ticker/24hr"),
        )
        rows = payload if isinstance(payload, list) else [payload]
        result: Dict[str, dict] = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).replace("_", "").upper()
            if symbol.endswith("TRY"):
                result[symbol] = row
        if not result:
            raise RuntimeError("24 saatlik ticker yanıtında TRY paritesi bulunamadı.")
        return result

    def book_tickers(self) -> Dict[str, dict]:
        payload = self._get_from_bases(
            "book",
            ("/api/v3/ticker/bookTicker", "/api/v1/ticker/bookTicker"),
        )
        rows = payload if isinstance(payload, list) else [payload]
        result: Dict[str, dict] = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).replace("_", "").upper()
            if symbol.endswith("TRY"):
                result[symbol] = row
        return result

    def klines(self, symbol: str, interval: str, limit: int = 220) -> pd.DataFrame:
        payload = self._get_from_bases(
            f"klines:{symbol}",
            ("/api/v1/klines", "/api/v3/klines"),
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        if not isinstance(payload, list) or len(payload) < 55:
            raise RuntimeError(f"{symbol} {interval}: yetersiz mum verisi")

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore",
        ][: len(payload[0])]
        df = pd.DataFrame(payload, columns=columns)
        for col in (
            "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_base", "taker_quote",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close", "volume"])


# ============================================================
# INDICATORS
# ============================================================

def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except Exception:
        return fallback


def calculate_metrics(df: pd.DataFrame) -> Metrics:
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    quote_volume = (
        df["quote_volume"].astype(float)
        if "quote_volume" in df.columns
        else close * volume
    )

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    rvol = volume / volume.shift(1).rolling(20).mean().replace(0, np.nan)
    quote_rvol = quote_volume / quote_volume.shift(1).rolling(20).mean().replace(
        0, np.nan
    )
    volume_trend = volume.iloc[-5:].mean() / max(
        volume.iloc[-10:-5].mean(), 1e-12
    )

    candle_range = max(float(high.iloc[-1] - low.iloc[-1]), 1e-12)
    upper_wick = float(high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1]))
    lower_wick = float(min(open_.iloc[-1], close.iloc[-1]) - low.iloc[-1])

    price = float(close.iloc[-1])
    return Metrics(
        price=price,
        previous_close=float(close.iloc[-2]),
        ema20=safe_float(ema20.iloc[-1], price),
        ema50=safe_float(ema50.iloc[-1], price),
        ema200=safe_float(ema200.iloc[-1], price),
        rsi14=safe_float(rsi.iloc[-1], 50),
        macd_hist=safe_float(macd_hist.iloc[-1]),
        macd_hist_previous=safe_float(macd_hist.iloc[-2]),
        bb_mid=safe_float(bb_mid.iloc[-1], price),
        bb_upper=safe_float(bb_upper.iloc[-1], price),
        bb_lower=safe_float(bb_lower.iloc[-1], price),
        bb_width=safe_float(bb_width.iloc[-1]),
        bb_width_previous=safe_float(bb_width.iloc[-6]),
        atr14=safe_float(atr.iloc[-1], price * 0.01),
        atr_percent=safe_float(atr.iloc[-1] / max(price, 1e-12) * 100),
        rvol=safe_float(rvol.iloc[-1]),
        quote_rvol=safe_float(quote_rvol.iloc[-1]),
        volume_trend=safe_float(volume_trend, 1),
        recent_high_20=float(high.iloc[-20:].max()),
        recent_low_20=float(low.iloc[-20:].min()),
        prior_high_20=float(high.iloc[-21:-1].max()),
        upper_wick_ratio=upper_wick / candle_range,
        lower_wick_ratio=lower_wick / candle_range,
    )


# ============================================================
# LEARNING DATABASE
# ============================================================

class LearningEngine:
    """
    Bounded online learning.

    - Records every sent signal as a paper trade.
    - Reviews target/stop/max-hold outcome.
    - Learns feature win rates with Beta priors.
    - Adds only a bounded score bonus/penalty.
    - Never changes stop limits or minimum risk/reward.
    """

    def __init__(
        self,
        db_path: str,
        enabled: bool,
        min_samples: int,
        max_bonus: float,
    ):
        self.enabled = enabled
        self.min_samples = min_samples
        self.max_bonus = max_bonus
        self.db_path = Path(db_path)
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    entry REAL NOT NULL,
                    stop REAL NOT NULL,
                    target REAL NOT NULL,
                    max_hold_hours INTEGER NOT NULL,
                    score REAL NOT NULL,
                    features_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    outcome TEXT,
                    closed_at REAL,
                    realized_return REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_stats (
                    feature TEXT PRIMARY KEY,
                    wins REAL NOT NULL DEFAULT 3.0,
                    losses REAL NOT NULL DEFAULT 2.0,
                    samples INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def normalize_feature(reason: str) -> str:
        text = reason.lower().strip()
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"%?-?\d+(?:[.,]\d+)?x?", "#", text)
        text = re.sub(r"\s+", " ", text)
        return text[:120]

    def tags(self, reasons: List[str]) -> List[str]:
        return sorted(set(
            self.normalize_feature(x)
            for x in reasons
            if self.normalize_feature(x)
        ))

    def adaptive_bonus(self, reasons: List[str]) -> float:
        if not self.enabled:
            return 0.0

        bonus = 0.0
        with self._connect() as conn:
            for tag in self.tags(reasons):
                row = conn.execute(
                    "SELECT wins, losses, samples FROM feature_stats WHERE feature=?",
                    (tag,),
                ).fetchone()
                if not row:
                    continue
                wins, losses, samples = row
                if samples < self.min_samples:
                    continue
                posterior = wins / max(wins + losses, 1e-12)
                feature_bonus = max(-2.5, min(2.5, (posterior - 0.60) * 12.0))
                bonus += feature_bonus
        return max(-self.max_bonus, min(self.max_bonus, bonus))

    def record_signal(self, signal: Signal) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals
                (symbol, created_at, entry, stop, target, max_hold_hours,
                 score, features_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """,
                (
                    signal.symbol,
                    time.time(),
                    signal.entry_high,
                    signal.stop,
                    signal.target,
                    signal.max_hold_hours,
                    signal.score,
                    json.dumps(self.tags(signal.reasons), ensure_ascii=False),
                ),
            )
            conn.commit()

    def open_signals(self) -> List[tuple]:
        if not self.enabled:
            return []
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT id, symbol, created_at, entry, stop, target,
                       max_hold_hours, features_json
                FROM signals WHERE status='OPEN'
                ORDER BY created_at ASC
                """
            ).fetchall()

    def close_signal(
        self,
        signal_id: int,
        outcome: str,
        realized_return: float,
        features_json: str,
    ) -> None:
        if not self.enabled:
            return

        now = time.time()
        features = json.loads(features_json)
        if outcome == "WIN":
            win_credit, loss_credit = 1.0, 0.0
        elif outcome == "LOSS":
            win_credit, loss_credit = 0.0, 1.0
        else:
            win_credit, loss_credit = 0.5, 0.5

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE signals
                SET status='CLOSED', outcome=?, closed_at=?, realized_return=?
                WHERE id=?
                """,
                (outcome, now, realized_return, signal_id),
            )

            for feature in features:
                existing = conn.execute(
                    "SELECT wins, losses, samples FROM feature_stats WHERE feature=?",
                    (feature,),
                ).fetchone()

                if existing:
                    conn.execute(
                        """
                        UPDATE feature_stats
                        SET wins=?, losses=?, samples=?, updated_at=?
                        WHERE feature=?
                        """,
                        (
                            existing[0] + win_credit,
                            existing[1] + loss_credit,
                            existing[2] + 1,
                            now,
                            feature,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO feature_stats
                        (feature, wins, losses, samples, updated_at)
                        VALUES (?, ?, ?, 1, ?)
                        """,
                        (
                            feature,
                            3.0 + win_credit,
                            2.0 + loss_credit,
                            now,
                        ),
                    )
            conn.commit()

    def review_open_signals(self, exchange: BinanceTRClient) -> int:
        if not self.enabled:
            return 0

        closed = 0
        now = time.time()

        for row in self.open_signals():
            (
                signal_id, symbol, created_at, entry, stop, target,
                max_hold_hours, features_json
            ) = row

            try:
                df = exchange.klines(symbol, "5m", 500)
                if "open_time" not in df.columns:
                    continue

                bars = df[df["open_time"] >= int(created_at * 1000)]
                if bars.empty:
                    continue

                outcome = None
                realized_return = 0.0

                for _, candle in bars.iterrows():
                    low = float(candle["low"])
                    high = float(candle["high"])
                    hit_stop = low <= stop
                    hit_target = high >= target

                    if hit_stop and hit_target:
                        outcome = "LOSS"
                        realized_return = (stop - entry) / entry * 100
                        break
                    if hit_stop:
                        outcome = "LOSS"
                        realized_return = (stop - entry) / entry * 100
                        break
                    if hit_target:
                        outcome = "WIN"
                        realized_return = (target - entry) / entry * 100
                        break

                if outcome is None and now - created_at >= max_hold_hours * 3600:
                    last_close = float(bars.iloc[-1]["close"])
                    realized_return = (last_close - entry) / entry * 100
                    if realized_return >= 0.8:
                        outcome = "WIN"
                    elif realized_return <= -0.8:
                        outcome = "LOSS"
                    else:
                        outcome = "NEUTRAL"

                if outcome:
                    self.close_signal(
                        signal_id,
                        outcome,
                        realized_return,
                        features_json,
                    )
                    logging.info(
                        "Öğrenme sonucu: %s %s %.2f%%",
                        symbol, outcome, realized_return
                    )
                    closed += 1

            except Exception as exc:
                logging.warning(
                    "Açık sinyal değerlendirilemedi %s: %s",
                    symbol, exc
                )

        return closed

    def summary(self) -> Dict[str, float]:
        if not self.enabled:
            return {
                "closed": 0,
                "wins": 0,
                "losses": 0,
                "neutral": 0,
                "win_rate": 0.0,
                "average_return": 0.0,
            }

        with self._connect() as conn:
            closed = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE status='CLOSED'"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome='WIN'"
            ).fetchone()[0]
            losses = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome='LOSS'"
            ).fetchone()[0]
            neutral = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome='NEUTRAL'"
            ).fetchone()[0]
            avg = conn.execute(
                "SELECT AVG(realized_return) FROM signals WHERE status='CLOSED'"
            ).fetchone()[0]

        return {
            "closed": closed,
            "wins": wins,
            "losses": losses,
            "neutral": neutral,
            "win_rate": (wins / closed * 100) if closed else 0.0,
            "average_return": float(avg or 0.0),
        }


# ============================================================
# STRATEGY
# ============================================================

def ticker_float(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return safe_float(row[key], default)
    return default


def build_market_context(
    client: BinanceTRClient,
    tickers: Dict[str, dict],
) -> MarketContext:
    symbol = "BTCTRY"
    try:
        m15 = calculate_metrics(client.klines(symbol, "15m"))
        m1h = calculate_metrics(client.klines(symbol, "1h"))
        change = ticker_float(tickers.get(symbol, {}), "priceChangePercent")

        ok15 = m15.ema20 >= m15.ema50 and m15.rsi14 >= 43
        ok1h = m1h.ema20 >= m1h.ema50 and m1h.rsi14 >= 42

        risk_off = (
            not ok15
            and not ok1h
            and m15.macd_hist < 0
            and m1h.macd_hist < 0
            and change < -1.5
        )

        return MarketContext(
            btc_symbol=symbol,
            btc_price=m15.price,
            btc_change_24h=change,
            btc_15m_ok=ok15,
            btc_1h_ok=ok1h,
            risk_off=risk_off,
        )

    except Exception as exc:
        logging.warning("BTC bağlamı alınamadı, nötr kullanılıyor: %s", exc)
        return MarketContext("UNKNOWN", 0, 0, True, True, False)


def prefilter_score(
    ticker: dict,
    book: dict,
    cfg: Settings,
) -> Optional[float]:
    change = ticker_float(ticker, "priceChangePercent")
    quote_volume = ticker_float(ticker, "quoteVolume", "quoteQty")
    bid = ticker_float(book, "bidPrice", "bid")
    ask = ticker_float(book, "askPrice", "ask")

    if quote_volume < cfg.min_24h_try_volume:
        return None
    if not (cfg.min_24h_change <= change <= cfg.max_24h_change):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    spread = (ask - bid) / max((ask + bid) / 2, 1e-12) * 100
    if spread > cfg.max_spread_percent:
        return None

    liquidity_component = min(25.0, math.log10(max(quote_volume, 1)) * 3)
    movement_component = 15 - min(15, abs(change - 3.5))
    spread_component = max(0, 10 - spread * 12)

    return liquidity_component + movement_component + spread_component


def evaluate_signal(
    symbol: str,
    ticker: dict,
    book: dict,
    m5: Metrics,
    m15: Metrics,
    m1h: Metrics,
    m4h: Metrics,
    market: MarketContext,
    cfg: Settings,
    learner: LearningEngine,
) -> Optional[Signal]:

    price = m15.price
    change = ticker_float(ticker, "priceChangePercent")
    quote_volume = ticker_float(ticker, "quoteVolume", "quoteQty")
    bid = ticker_float(book, "bidPrice", "bid")
    ask = ticker_float(book, "askPrice", "ask")
    spread = (ask - bid) / max((ask + bid) / 2, 1e-12) * 100

    if cfg.btc_filter_enabled and market.risk_off:
        return None
    if m15.atr_percent <= 0 or m15.atr_percent > 6:
        return None

    score = 0.0
    reasons: List[str] = []
    warnings: List[str] = []

    if m15.ema20 > m15.ema50:
        score += 11
        reasons.append("15 dk trend yukarı")
    else:
        score -= 12

    if m1h.ema20 > m1h.ema50:
        score += 15
        reasons.append("1 saat trend yukarı")
    else:
        score -= 15

    if m4h.price > m4h.ema20:
        score += 7
        reasons.append("4 saat ana trend destekli")
    else:
        score -= 8

    if m1h.price > m1h.ema200:
        score += 4

    effective_rvol = max(m15.rvol, m15.quote_rvol)
    if 1.30 <= effective_rvol <= 3.8:
        score += 18
        reasons.append(f"göreceli hacim {effective_rvol:.1f}x")
    elif 1.05 <= effective_rvol < 1.30:
        score += 7
    elif effective_rvol > 5:
        score -= 11
        warnings.append("hacim aşırı patlamış")
    else:
        score -= 7

    if 1.05 <= m15.volume_trend <= 2.8:
        score += 5

    if 51 <= m15.rsi14 <= 66:
        score += 13
        reasons.append(f"RSI sağlıklı ({m15.rsi14:.0f})")
    elif 46 <= m15.rsi14 < 51:
        score += 5
    elif m15.rsi14 > 72:
        score -= 14
        warnings.append("RSI aşırı ısınmış")
    elif m15.rsi14 < 41:
        score -= 10

    if m15.macd_hist > 0 and m15.macd_hist >= m15.macd_hist_previous:
        score += 10
        reasons.append("MACD ivmesi pozitif")
    elif m15.macd_hist < m15.macd_hist_previous:
        score -= 5

    if m15.bb_width_previous < 0.07 and m15.bb_width > m15.bb_width_previous:
        score += 12
        reasons.append("Bollinger sıkışması açılıyor")
    elif m15.bb_width > m15.bb_width_previous:
        score += 4

    distance_above_upper = (price - m15.bb_upper) / max(price, 1e-12) * 100
    if distance_above_upper > 1.5:
        score -= 13
        warnings.append("üst bandın fazla üzerinde")

    breakout_distance = (
        price - m15.prior_high_20
    ) / max(m15.prior_high_20, 1e-12) * 100

    if -1.2 <= breakout_distance <= 1.5:
        score += 9
        reasons.append("dirence yakın kontrollü kurulum")
    elif breakout_distance > 3:
        score -= 11
        warnings.append("kırılım fazla uzamış")

    if m15.upper_wick_ratio > 0.55:
        score -= 8
        warnings.append("üst fitil satış baskısı")

    if 0.5 <= change <= 6.5:
        score += 9
        reasons.append(f"24s hareket kontrollü %{change:.1f}")
    elif 6.5 < change <= 10:
        score += 1
        warnings.append("günlük hareket ilerlemiş")
    elif change > 10:
        score -= 8

    if cfg.btc_filter_enabled:
        if market.btc_15m_ok and market.btc_1h_ok:
            score += 8
            reasons.append("BTC yönü destekli")
        elif not market.btc_1h_ok:
            score -= 8
            warnings.append("BTC 1 saat zayıf")

    atr = max(m15.atr14, price * 0.0035)
    support = max(
        m15.ema20,
        m15.bb_mid,
        m15.recent_low_20
        + 0.35 * (m15.recent_high_20 - m15.recent_low_20),
    )

    entry_high = min(ask, price + 0.10 * atr)
    entry_low = max(support, price - 0.35 * atr)
    if entry_low > entry_high:
        entry_low = price - 0.15 * atr

    structural_stop = min(
        entry_low - 1.05 * atr,
        m15.bb_mid - 0.35 * atr,
    )
    stop_floor = entry_high * (1 - cfg.max_stop_percent / 100)
    stop = max(structural_stop, stop_floor)

    risk = entry_high - stop
    if risk <= 0:
        return None

    stop_percent = risk / entry_high * 100
    desired_target_percent = min(
        cfg.max_target_percent,
        max(cfg.min_target_percent, cfg.preferred_target_percent),
    )

    target_by_percent = entry_high * (1 + desired_target_percent / 100)
    target_by_rr = entry_high + risk * cfg.min_risk_reward
    target = min(
        max(target_by_percent, target_by_rr),
        entry_high * (1 + cfg.max_target_percent / 100),
    )

    target_percent = (target - entry_high) / entry_high * 100
    risk_reward = (target - entry_high) / risk

    if stop_percent > cfg.max_stop_percent:
        return None
    if not (cfg.min_target_percent <= target_percent <= cfg.max_target_percent):
        return None
    if risk_reward < cfg.min_risk_reward:
        return None

    if stop_percent <= 2.2:
        score += 8
        reasons.append(f"sıkı stop %{stop_percent:.1f}")
    else:
        score += 3

    if risk_reward >= 1.9:
        score += 8
        reasons.append(f"risk/ödül {risk_reward:.1f}")
    else:
        score += 4

    raw_score = score
    learning_bonus = learner.adaptive_bonus(reasons)
    score += learning_bonus

    if learning_bonus:
        reasons.append(f"öğrenme etkisi {learning_bonus:+.1f} puan")

    if score < cfg.min_score:
        return None

    return Signal(
        symbol=symbol,
        score=round(min(score, 100), 1),
        raw_score=round(raw_score, 1),
        learning_bonus=round(learning_bonus, 1),
        price=price,
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        target=target,
        stop_percent=stop_percent,
        target_percent=target_percent,
        risk_reward=risk_reward,
        change_24h=change,
        quote_volume_24h=quote_volume,
        spread_percent=spread,
        max_hold_hours=cfg.max_hold_hours,
        reasons=reasons[:9],
        warnings=warnings[:4],
    )


# ============================================================
# DUPLICATE CONTROL
# ============================================================

class DuplicateStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

    def should_send(
        self,
        signal: Signal,
        cooldown_minutes: int,
        required_improvement: float,
    ) -> bool:
        if not self.db_path.exists():
            return True

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT created_at, score
                FROM signals
                WHERE symbol=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (signal.symbol,),
            ).fetchone()

        if not row:
            return True

        elapsed = time.time() - float(row[0])
        old_score = float(row[1])

        if elapsed >= cooldown_minutes * 60:
            return True

        return signal.score >= old_score + required_improvement


# ============================================================
# TELEGRAM
# ============================================================

def format_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    if value >= 10:
        return f"{value:.3f}"
    if value >= 1:
        return f"{value:.4f}"
    if value >= 0.01:
        return f"{value:.6f}"
    return f"{value:.8f}"


class TelegramNotifier:
    def __init__(self, http: HttpClient, cfg: Settings):
        self.http = http
        self.cfg = cfg

    def send(self, text: str) -> None:
        if self.cfg.dry_run:
            logging.info("DRY RUN Telegram:\n%s", text)
            return

        self.http.post_json(
            f"https://api.telegram.org/bot{self.cfg.telegram_token}/sendMessage",
            {
                "chat_id": self.cfg.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )

    def startup(self) -> None:
        self.send(
            "✅ <b>Coin Radar Stable çalışıyor</b>\n\n"
            "Günlük sinyal sınırı: <b>Yok</b>\n"
            f"Minimum puan: <b>{self.cfg.min_score:.0f}/100</b>\n"
            f"Kendi kendine öğrenme: <b>{'Açık' if self.cfg.learning_enabled else 'Kapalı'}</b>\n"
            "Mod: <b>kâğıt işlem + uyarı</b>\n"
            "Otomatik alım-satım: <b>Kapalı</b>"
        )

    def heartbeat(
        self,
        scanned: int,
        candidates: int,
        elapsed: float,
        learning: Dict[str, float],
    ) -> None:
        self.send(
            "💓 <b>Coin Radar sağlık kontrolü</b>\n\n"
            f"Son turda incelenen: <b>{scanned}</b>\n"
            f"Filtreyi geçen: <b>{candidates}</b>\n"
            f"Tarama süresi: <b>{elapsed:.1f} sn</b>\n"
            f"Sonuçlanan kâğıt işlem: <b>{int(learning['closed'])}</b>\n"
            f"Başarı oranı: <b>%{learning['win_rate']:.1f}</b>\n"
            f"Ortalama sonuç: <b>%{learning['average_return']:.2f}</b>\n"
            "Sistem çalışıyor."
        )

    def performance_report(self, learning: Dict[str, float]) -> None:
        self.send(
            "📊 <b>Coin Radar performans raporu</b>\n\n"
            f"Sonuçlanan sinyal: <b>{int(learning['closed'])}</b>\n"
            f"Kazanan: <b>{int(learning['wins'])}</b>\n"
            f"Kaybeden: <b>{int(learning['losses'])}</b>\n"
            f"Nötr: <b>{int(learning['neutral'])}</b>\n"
            f"Başarı oranı: <b>%{learning['win_rate']:.1f}</b>\n"
            f"Ortalama sonuç: <b>%{learning['average_return']:.2f}</b>"
        )

    def signal(self, signal: Signal, market: MarketContext) -> None:
        reasons = "\n".join(f"• {x}" for x in signal.reasons)
        warnings = (
            "\n\n⚠️ <b>Uyarılar</b>\n"
            + "\n".join(f"• {x}" for x in signal.warnings)
            if signal.warnings
            else ""
        )

        self.send(
            "🚨 <b>COIN RADAR — ADAY</b>\n\n"
            f"<b>{signal.symbol.replace('TRY', '/TRY')}</b>\n"
            f"Puan: <b>{signal.score:.0f}/100</b>\n"
            f"Ham puan: <b>{signal.raw_score:.1f}</b>\n"
            f"Öğrenme etkisi: <b>{signal.learning_bonus:+.1f}</b>\n"
            f"Anlık: <b>{format_price(signal.price)}</b>\n"
            f"24s: <b>%{signal.change_24h:.2f}</b>\n\n"
            f"🟢 Giriş: <b>{format_price(signal.entry_low)} – "
            f"{format_price(signal.entry_high)}</b>\n"
            f"🔴 Stop: <b>{format_price(signal.stop)}</b> "
            f"(%{signal.stop_percent:.2f})\n"
            f"🎯 Tam satış: <b>{format_price(signal.target)}</b> "
            f"(%{signal.target_percent:.2f})\n"
            f"⚖️ Risk/ödül: <b>{signal.risk_reward:.2f}</b>\n"
            f"⏱ Maksimum bekleme: <b>{signal.max_hold_hours} saat</b>\n"
            f"↔️ Spread: <b>%{signal.spread_percent:.3f}</b>\n\n"
            f"<b>Neden?</b>\n{reasons}"
            f"{warnings}\n\n"
            f"BTC: {market.btc_symbol} | 24s %{market.btc_change_24h:.2f}\n\n"
            "Bu otomatik ön elemedir; işlem emri değildir."
        )


# ============================================================
# APPLICATION
# ============================================================

class RadarApp:
    def __init__(self, cfg: Settings):
        self.cfg = cfg

        logging.basicConfig(
            level=getattr(logging, cfg.log_level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        self.http = HttpClient(cfg.request_timeout_seconds)
        self.exchange = BinanceTRClient(self.http)
        self.telegram = TelegramNotifier(self.http, cfg)
        self.learner = LearningEngine(
            db_path=cfg.learning_db_path,
            enabled=cfg.learning_enabled,
            min_samples=cfg.learning_min_samples,
            max_bonus=cfg.learning_max_bonus,
        )
        self.duplicate_store = DuplicateStore(cfg.learning_db_path)

        self.last_heartbeat_at = 0.0
        self.last_learning_review_at = 0.0
        self.last_performance_report_at = 0.0

    def scan_once(self) -> Tuple[int, int]:
        scan_started = time.time()
        logging.info("Tarama turu başladı.")

        now = time.time()

        if (
            self.cfg.learning_enabled
            and now - self.last_learning_review_at
            >= self.cfg.learning_review_minutes * 60
        ):
            reviewed = self.learner.review_open_signals(self.exchange)
            if reviewed:
                logging.info("%s açık sinyal sonuçlandırıldı.", reviewed)
            self.last_learning_review_at = now

        tickers = self.exchange.tickers_24h()
        books = self.exchange.book_tickers()

        logging.info(
            "Ticker verisinden %s TRY paritesi keşfedildi.",
            len(tickers)
        )

        market = build_market_context(self.exchange, tickers)

        ranked: List[Tuple[float, str]] = []
        for symbol, ticker in tickers.items():
            if symbol == "BTCTRY":
                continue

            score = prefilter_score(
                ticker,
                books.get(symbol, {}),
                self.cfg,
            )

            if score is not None:
                ranked.append((score, symbol))

        ranked.sort(reverse=True)
        selected = [
            symbol
            for _, symbol in ranked[: self.cfg.max_prefilter_symbols]
        ]

        logging.info(
            "Ön filtreden %s coin geçti; %s coin detaylı incelenecek.",
            len(ranked),
            len(selected),
        )

        candidates: List[Signal] = []
        scanned = 0

        for index, symbol in enumerate(selected, start=1):
            try:
                ticker = tickers[symbol]
                book = books[symbol]

                m5 = calculate_metrics(self.exchange.klines(symbol, "5m"))
                m15 = calculate_metrics(self.exchange.klines(symbol, "15m"))
                m1h = calculate_metrics(self.exchange.klines(symbol, "1h"))
                m4h = calculate_metrics(self.exchange.klines(symbol, "4h"))

                signal = evaluate_signal(
                    symbol=symbol,
                    ticker=ticker,
                    book=book,
                    m5=m5,
                    m15=m15,
                    m1h=m1h,
                    m4h=m4h,
                    market=market,
                    cfg=self.cfg,
                    learner=self.learner,
                )

                scanned += 1

                if signal:
                    candidates.append(signal)
                    logging.info(
                        "Aday: %s puan=%.1f",
                        symbol,
                        signal.score,
                    )

                if index % 10 == 0:
                    logging.info(
                        "Tarama ilerlemesi: %s/%s",
                        index,
                        len(selected),
                    )

                time.sleep(self.cfg.request_pause_seconds)

            except Exception as exc:
                logging.warning("%s incelenemedi: %s", symbol, exc)

        candidates.sort(
            key=lambda item: (
                item.score,
                item.risk_reward,
                item.quote_volume_24h,
            ),
            reverse=True,
        )

        sent_count = 0

        for signal in candidates:
            if self.duplicate_store.should_send(
                signal,
                cooldown_minutes=self.cfg.cooldown_minutes,
                required_improvement=self.cfg.resignal_score_improvement,
            ):
                self.telegram.signal(signal, market)
                self.learner.record_signal(signal)
                sent_count += 1

                logging.info(
                    "Telegram sinyali gönderildi: %s puan=%.1f",
                    signal.symbol,
                    signal.score,
                )

        elapsed = time.time() - scan_started

        logging.info(
            "Tarama tamamlandı: incelenen=%s aday=%s gönderilen=%s süre=%.1fs",
            scanned,
            len(candidates),
            sent_count,
            elapsed,
        )

        learning_summary = self.learner.summary()

        if (
            now - self.last_heartbeat_at
            >= self.cfg.heartbeat_minutes * 60
        ):
            self.telegram.heartbeat(
                scanned=scanned,
                candidates=len(candidates),
                elapsed=elapsed,
                learning=learning_summary,
            )
            self.last_heartbeat_at = now

        if (
            now - self.last_performance_report_at
            >= self.cfg.performance_report_hours * 3600
        ):
            self.telegram.performance_report(learning_summary)
            self.last_performance_report_at = now

        return scanned, len(candidates)

    def run(self) -> None:
        logging.info("Coin Radar Stable başlatılıyor.")

        if self.cfg.send_startup_message:
            self.telegram.startup()

        logging.info("Telegram başlangıç mesajı gönderildi.")

        consecutive_errors = 0

        while True:
            started = time.time()

            try:
                self.scan_once()
                consecutive_errors = 0

            except KeyboardInterrupt:
                logging.info("Kapatılıyor.")
                return

            except Exception:
                consecutive_errors += 1
                logging.exception(
                    "Tarama turu hata verdi; ardışık hata=%s",
                    consecutive_errors,
                )

            elapsed = time.time() - started
            wait = max(5.0, self.cfg.scan_interval_seconds - elapsed)

            logging.info("Sonraki tur %.0f saniye sonra.", wait)
            time.sleep(wait)


if __name__ == "__main__":
    RadarApp(Settings.from_env()).run()
