"""
Coin Radar AI — Stable Adaptive Edition

Binance TR TRY paritelerini tarar, Telegram'a yüksek kaliteli adaylar gönderir,
açık kâğıt işlemleri yönetir ve geçmiş sonuçlardan kontrollü biçimde öğrenir.

Temel ilkeler:
- Sabit %4 kâr hedefi YOKTUR.
- Stop sıkıdır ve öğrenme sistemi stop güvenlik sınırlarını gevşetemez.
- Hedef; ATR, dirençler, trend gücü, hacim, BTC bağlamı ve öğrenilen MFE
  (işlem sırasında görülen en yüksek olumlu hareket) verisiyle dinamik oluşur.
- Güçlü harekette ilk hedef nihai satış değildir; takip eden stop devreye girer.
- Otomatik emir vermez. Yalnızca radar, kâğıt işlem yönetimi ve bildirim yapar.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


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
    min_stop_percent: float
    min_risk_reward: float
    min_initial_target_percent: float
    max_initial_target_percent: float
    max_hold_hours: int
    cooldown_minutes: int
    resignal_score_improvement: float
    btc_filter_enabled: bool
    learning_enabled: bool
    learning_db_path: str
    learning_min_samples: int
    learning_max_score_bonus: float
    learning_max_target_adjustment: float
    learning_review_minutes: int
    heartbeat_minutes: int
    performance_report_hours: int
    send_startup_message: bool
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
            scan_interval_seconds=max(60, env_int("SCAN_INTERVAL_SECONDS", 180)),
            request_timeout_seconds=max(5, env_int("REQUEST_TIMEOUT_SECONDS", 12)),
            request_pause_seconds=max(0.03, env_float("REQUEST_PAUSE_SECONDS", 0.08)),
            max_prefilter_symbols=max(15, env_int("MAX_PREFILTER_SYMBOLS", 55)),
            min_score=env_float("MIN_SCORE", 80.0),
            min_24h_try_volume=env_float("MIN_24H_TRY_VOLUME", 15_000_000),
            min_24h_change=env_float("MIN_24H_CHANGE", -3.0),
            max_24h_change=env_float("MAX_24H_CHANGE", 16.0),
            max_spread_percent=env_float("MAX_SPREAD_PERCENT", 0.70),
            max_stop_percent=env_float("MAX_STOP_PERCENT", 3.20),
            min_stop_percent=env_float("MIN_STOP_PERCENT", 0.70),
            min_risk_reward=env_float("MIN_RISK_REWARD", 1.60),
            min_initial_target_percent=env_float("MIN_INITIAL_TARGET_PERCENT", 2.20),
            max_initial_target_percent=env_float("MAX_INITIAL_TARGET_PERCENT", 18.0),
            max_hold_hours=max(2, env_int("MAX_HOLD_HOURS", 12)),
            cooldown_minutes=max(15, env_int("COOLDOWN_MINUTES", 120)),
            resignal_score_improvement=env_float("RESIGNAL_SCORE_IMPROVEMENT", 5.0),
            btc_filter_enabled=env_bool("BTC_FILTER_ENABLED", True),
            learning_enabled=env_bool("LEARNING_ENABLED", True),
            learning_db_path=os.getenv("LEARNING_DB_PATH", "/data/radar_learning.db"),
            learning_min_samples=max(10, env_int("LEARNING_MIN_SAMPLES", 20)),
            learning_max_score_bonus=env_float("LEARNING_MAX_SCORE_BONUS", 10.0),
            learning_max_target_adjustment=env_float(
                "LEARNING_MAX_TARGET_ADJUSTMENT", 0.35
            ),
            learning_review_minutes=max(5, env_int("LEARNING_REVIEW_MINUTES", 10)),
            heartbeat_minutes=max(10, env_int("HEARTBEAT_MINUTES", 30)),
            performance_report_hours=max(1, env_int("PERFORMANCE_REPORT_HOURS", 24)),
            send_startup_message=env_bool("SEND_STARTUP_MESSAGE", True),
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
    ema9: float
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
    recent_high_50: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    body_percent: float
    close_position: float
    adx14: float
    plus_di: float
    minus_di: float
    roc5: float


@dataclass(frozen=True)
class MarketContext:
    btc_symbol: str
    btc_price: float
    btc_change_24h: float
    btc_15m_ok: bool
    btc_1h_ok: bool
    btc_4h_ok: bool
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
    initial_stop: float
    initial_target: float
    target_zone_high: float
    stop_percent: float
    initial_target_percent: float
    risk_reward: float
    trail_activation_r: float
    trail_atr_multiplier: float
    max_hold_hours: int
    setup_strength: str
    change_24h: float
    quote_volume_24h: float
    spread_percent: float
    reasons: List[str]
    warnings: List[str]
    feature_tags: List[str]
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
            "User-Agent": "CoinRadarAI/1.0",
            "Accept": "application/json",
            "Connection": "keep-alive",
        })

    def get_json(self, url: str, params: Optional[dict] = None, attempts: int = 3) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {418, 429}:
                    retry_after = int(response.headers.get("Retry-After", "10"))
                    time.sleep(min(retry_after, 30))
                    raise RuntimeError(f"rate limit: {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("code") not in (None, 0, "0"):
                    raise RuntimeError(str(payload))
                return payload.get("data", payload) if isinstance(payload, dict) else payload
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    delay = min(15.0, 2 ** (attempt - 1) + random.random())
                    logging.warning("HTTP denemesi başarısız: %s; %.1fs", exc, delay)
                    time.sleep(delay)
        raise RuntimeError(f"GET başarısız {url}: {last_error}")

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
                    time.sleep(min(10.0, 2 ** attempt))
        raise RuntimeError(f"POST başarısız: {last_error}")


class BinanceTRClient:
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
        cached = self._base_cache.get(cache_key)
        bases = ([cached] if cached else []) + [
            base for base in self.MARKET_BASES if base != cached
        ]
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
            "ticker24", ("/api/v3/ticker/24hr", "/api/v1/ticker/24hr")
        )
        rows = payload if isinstance(payload, list) else [payload]
        result = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).replace("_", "").upper()
            if symbol.endswith("TRY"):
                result[symbol] = row
        if not result:
            raise RuntimeError("TRY ticker bulunamadı")
        return result

    def book_tickers(self) -> Dict[str, dict]:
        payload = self._get_from_bases(
            "book", ("/api/v3/ticker/bookTicker", "/api/v1/ticker/bookTicker")
        )
        rows = payload if isinstance(payload, list) else [payload]
        result = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).replace("_", "").upper()
            if symbol.endswith("TRY"):
                result[symbol] = row
        return result

    def klines(self, symbol: str, interval: str, limit: int = 240) -> pd.DataFrame:
        payload = self._get_from_bases(
            f"klines:{symbol}:{interval}",
            ("/api/v1/klines", "/api/v3/klines"),
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        if not isinstance(payload, list) or len(payload) < 60:
            raise RuntimeError(f"{symbol} {interval}: yetersiz mum")
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore",
        ][: len(payload[0])]
        df = pd.DataFrame(payload, columns=columns)
        for col in [
            "open", "high", "low", "close", "volume", "quote_volume",
            "trades", "taker_base", "taker_quote",
        ]:
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


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return adx, plus_di, minus_di


def calculate_metrics(df: pd.DataFrame) -> Metrics:
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    quote_volume = df["quote_volume"].astype(float) if "quote_volume" in df else close * volume

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    true_range = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    rvol = volume / volume.shift(1).rolling(20).mean().replace(0, np.nan)
    quote_rvol = quote_volume / quote_volume.shift(1).rolling(20).mean().replace(0, np.nan)
    volume_trend = volume.iloc[-5:].mean() / max(volume.iloc[-10:-5].mean(), 1e-12)

    adx, plus_di, minus_di = calculate_adx(high, low, close)
    roc5 = close.pct_change(5) * 100

    candle_range = max(float(high.iloc[-1] - low.iloc[-1]), 1e-12)
    body = abs(float(close.iloc[-1] - open_.iloc[-1]))
    upper_wick = float(high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1]))
    lower_wick = float(min(open_.iloc[-1], close.iloc[-1]) - low.iloc[-1])
    close_position = float((close.iloc[-1] - low.iloc[-1]) / candle_range)
    price = float(close.iloc[-1])

    return Metrics(
        price=price,
        previous_close=float(close.iloc[-2]),
        ema9=safe_float(ema9.iloc[-1], price),
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
        recent_high_50=float(high.iloc[-50:].max()),
        upper_wick_ratio=upper_wick / candle_range,
        lower_wick_ratio=lower_wick / candle_range,
        body_percent=body / max(price, 1e-12) * 100,
        close_position=close_position,
        adx14=safe_float(adx.iloc[-1]),
        plus_di=safe_float(plus_di.iloc[-1]),
        minus_di=safe_float(minus_di.iloc[-1]),
        roc5=safe_float(roc5.iloc[-1]),
    )


# ============================================================
# LEARNING ENGINE
# ============================================================

class LearningEngine:
    """Bounded online learning; never relaxes hard risk limits."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.enabled = cfg.learning_enabled
        self.db_path = Path(cfg.learning_db_path)
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    entry REAL NOT NULL,
                    initial_stop REAL NOT NULL,
                    active_stop REAL NOT NULL,
                    initial_target REAL NOT NULL,
                    target_zone_high REAL NOT NULL,
                    risk REAL NOT NULL,
                    score REAL NOT NULL,
                    max_hold_hours INTEGER NOT NULL,
                    trail_activation_r REAL NOT NULL,
                    trail_atr_multiplier REAL NOT NULL,
                    features_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    highest_price REAL NOT NULL,
                    lowest_price REAL NOT NULL,
                    mfe_percent REAL NOT NULL DEFAULT 0,
                    mae_percent REAL NOT NULL DEFAULT 0,
                    target_touched INTEGER NOT NULL DEFAULT 0,
                    outcome TEXT,
                    exit_reason TEXT,
                    exit_price REAL,
                    realized_return REAL,
                    closed_at REAL,
                    last_notified_stop REAL
                );
                CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, created_at);
                CREATE TABLE IF NOT EXISTS feature_stats (
                    feature TEXT PRIMARY KEY,
                    wins REAL NOT NULL DEFAULT 3,
                    losses REAL NOT NULL DEFAULT 2,
                    samples INTEGER NOT NULL DEFAULT 0,
                    avg_mfe REAL NOT NULL DEFAULT 0,
                    avg_return REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
            """)
            conn.commit()

    @staticmethod
    def normalize_feature(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"-?\d+(?:[.,]\d+)?%?x?", "#", text)
        return re.sub(r"\s+", " ", text)[:100]

    def tags(self, reasons: Sequence[str]) -> List[str]:
        return sorted({self.normalize_feature(x) for x in reasons if x.strip()})

    def score_bonus(self, tags: Sequence[str]) -> float:
        if not self.enabled:
            return 0.0
        total = 0.0
        with self._connect() as conn:
            for tag in tags:
                row = conn.execute(
                    "SELECT wins, losses, samples, avg_return FROM feature_stats WHERE feature=?",
                    (tag,),
                ).fetchone()
                if not row or row["samples"] < self.cfg.learning_min_samples:
                    continue
                posterior = row["wins"] / max(row["wins"] + row["losses"], 1e-12)
                edge = (posterior - 0.58) * 10 + max(-1.0, min(1.0, row["avg_return"] / 4))
                total += max(-2.0, min(2.0, edge))
        return max(-self.cfg.learning_max_score_bonus, min(self.cfg.learning_max_score_bonus, total))

    def target_multiplier(self, tags: Sequence[str], fallback: float) -> float:
        if not self.enabled:
            return fallback
        weighted: List[Tuple[float, int]] = []
        with self._connect() as conn:
            for tag in tags:
                row = conn.execute(
                    "SELECT samples, avg_mfe FROM feature_stats WHERE feature=?", (tag,)
                ).fetchone()
                if row and row["samples"] >= self.cfg.learning_min_samples and row["avg_mfe"] > 0:
                    weighted.append((float(row["avg_mfe"]), int(row["samples"])))
        if not weighted:
            return fallback
        learned_mfe = sum(v * min(n, 100) for v, n in weighted) / sum(min(n, 100) for _, n in weighted)
        learned_r = max(1.6, min(6.0, learned_mfe / 1.4))
        low = fallback * (1 - self.cfg.learning_max_target_adjustment)
        high = fallback * (1 + self.cfg.learning_max_target_adjustment)
        return max(low, min(high, learned_r))

    def record(self, signal: Signal) -> None:
        if not self.enabled:
            return
        risk = signal.entry_high - signal.initial_stop
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO signals (
                    symbol, created_at, entry, initial_stop, active_stop,
                    initial_target, target_zone_high, risk, score, max_hold_hours,
                    trail_activation_r, trail_atr_multiplier, features_json,
                    highest_price, lowest_price, last_notified_stop
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol, time.time(), signal.entry_high, signal.initial_stop,
                signal.initial_stop, signal.initial_target, signal.target_zone_high,
                risk, signal.score, signal.max_hold_hours, signal.trail_activation_r,
                signal.trail_atr_multiplier, json.dumps(signal.feature_tags, ensure_ascii=False),
                signal.entry_high, signal.entry_high, signal.initial_stop,
            ))
            conn.commit()

    def recently_sent(self, symbol: str, cooldown_minutes: int, score: float, improvement: float) -> bool:
        if not self.enabled:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at, score FROM signals WHERE symbol=? ORDER BY created_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if not row:
            return False
        age = time.time() - row["created_at"]
        return age < cooldown_minutes * 60 and score < row["score"] + improvement

    def open_rows(self) -> List[sqlite3.Row]:
        if not self.enabled:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT * FROM signals WHERE status='OPEN' ORDER BY created_at").fetchall()

    def update_live(
        self,
        signal_id: int,
        highest: float,
        lowest: float,
        mfe: float,
        mae: float,
        active_stop: float,
        target_touched: int,
        last_notified_stop: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            if last_notified_stop is None:
                conn.execute("""
                    UPDATE signals SET highest_price=?, lowest_price=?, mfe_percent=?,
                    mae_percent=?, active_stop=?, target_touched=? WHERE id=?
                """, (highest, lowest, mfe, mae, active_stop, target_touched, signal_id))
            else:
                conn.execute("""
                    UPDATE signals SET highest_price=?, lowest_price=?, mfe_percent=?,
                    mae_percent=?, active_stop=?, target_touched=?, last_notified_stop=? WHERE id=?
                """, (highest, lowest, mfe, mae, active_stop, target_touched, last_notified_stop, signal_id))
            conn.commit()

    def close(self, row: sqlite3.Row, outcome: str, reason: str, exit_price: float) -> None:
        if not self.enabled:
            return
        realized = (exit_price - row["entry"]) / row["entry"] * 100
        now = time.time()
        tags = json.loads(row["features_json"])
        if outcome == "WIN":
            win_credit, loss_credit = 1.0, 0.0
        elif outcome == "LOSS":
            win_credit, loss_credit = 0.0, 1.0
        else:
            win_credit, loss_credit = 0.5, 0.5
        with self._connect() as conn:
            conn.execute("""
                UPDATE signals SET status='CLOSED', outcome=?, exit_reason=?, exit_price=?,
                realized_return=?, closed_at=? WHERE id=?
            """, (outcome, reason, exit_price, realized, now, row["id"]))
            for tag in tags:
                old = conn.execute("SELECT * FROM feature_stats WHERE feature=?", (tag,)).fetchone()
                if old:
                    n = old["samples"]
                    new_n = n + 1
                    avg_mfe = (old["avg_mfe"] * n + row["mfe_percent"]) / new_n
                    avg_return = (old["avg_return"] * n + realized) / new_n
                    conn.execute("""
                        UPDATE feature_stats SET wins=?, losses=?, samples=?, avg_mfe=?,
                        avg_return=?, updated_at=? WHERE feature=?
                    """, (old["wins"] + win_credit, old["losses"] + loss_credit,
                          new_n, avg_mfe, avg_return, now, tag))
                else:
                    conn.execute("""
                        INSERT INTO feature_stats(feature,wins,losses,samples,avg_mfe,avg_return,updated_at)
                        VALUES(?,?,?,?,?,?,?)
                    """, (tag, 3 + win_credit, 2 + loss_credit, 1,
                          row["mfe_percent"], realized, now))
            conn.commit()

    def summary(self) -> Dict[str, float]:
        if not self.enabled:
            return {"closed": 0, "wins": 0, "losses": 0, "win_rate": 0, "avg_return": 0, "avg_mfe": 0}
        with self._connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) closed,
                       SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) wins,
                       SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) losses,
                       AVG(realized_return) avg_return,
                       AVG(mfe_percent) avg_mfe
                FROM signals WHERE status='CLOSED'
            """).fetchone()
        closed = int(row["closed"] or 0)
        wins = int(row["wins"] or 0)
        losses = int(row["losses"] or 0)
        return {
            "closed": closed,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / closed * 100 if closed else 0,
            "avg_return": float(row["avg_return"] or 0),
            "avg_mfe": float(row["avg_mfe"] or 0),
        }


# ============================================================
# STRATEGY
# ============================================================

def ticker_float(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if row.get(key) not in (None, ""):
            return safe_float(row[key], default)
    return default


def build_market_context(client: BinanceTRClient, tickers: Dict[str, dict]) -> MarketContext:
    symbol = "BTCTRY"
    try:
        m15 = calculate_metrics(client.klines(symbol, "15m"))
        m1h = calculate_metrics(client.klines(symbol, "1h"))
        m4h = calculate_metrics(client.klines(symbol, "4h"))
        change = ticker_float(tickers.get(symbol, {}), "priceChangePercent")
        ok15 = m15.ema20 >= m15.ema50 and m15.rsi14 >= 43 and m15.macd_hist >= m15.macd_hist_previous
        ok1h = m1h.ema20 >= m1h.ema50 and m1h.rsi14 >= 42
        ok4h = m4h.price >= m4h.ema20 and m4h.ema20 >= m4h.ema50
        risk_off = (not ok15 and not ok1h and m15.macd_hist < 0 and m1h.macd_hist < 0 and change < -1.5)
        return MarketContext(symbol, m15.price, change, ok15, ok1h, ok4h, risk_off)
    except Exception as exc:
        logging.warning("BTC bağlamı alınamadı: %s", exc)
        return MarketContext("UNKNOWN", 0, 0, True, True, True, False)


def prefilter_score(ticker: dict, book: dict, cfg: Settings) -> Optional[float]:
    change = ticker_float(ticker, "priceChangePercent")
    quote_volume = ticker_float(ticker, "quoteVolume", "quoteQty")
    bid = ticker_float(book, "bidPrice", "bid")
    ask = ticker_float(book, "askPrice", "ask")
    if quote_volume < cfg.min_24h_try_volume or not (cfg.min_24h_change <= change <= cfg.max_24h_change):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    spread = (ask - bid) / max((ask + bid) / 2, 1e-12) * 100
    if spread > cfg.max_spread_percent:
        return None
    return min(28, math.log10(max(quote_volume, 1)) * 3.2) + max(0, 16 - abs(change - 3.5)) + max(0, 10 - spread * 12)


def setup_tags(m5: Metrics, m15: Metrics, m1h: Metrics, m4h: Metrics, market: MarketContext) -> List[str]:
    tags = []
    tags.append("rvol_high" if max(m15.rvol, m15.quote_rvol) >= 1.8 else "rvol_mid")
    tags.append("adx_strong" if m15.adx14 >= 25 else "adx_normal")
    tags.append("trend_4h" if m4h.price > m4h.ema20 else "counter_4h")
    tags.append("btc_aligned" if market.btc_15m_ok and market.btc_1h_ok else "btc_mixed")
    tags.append("breakout" if m15.price >= m15.prior_high_20 else "pre_breakout")
    tags.append("momentum_fast" if m5.roc5 >= 1.2 else "momentum_normal")
    tags.append("rsi_healthy" if 52 <= m15.rsi14 <= 66 else "rsi_other")
    return tags


def dynamic_target_plan(
    entry: float,
    stop: float,
    m5: Metrics,
    m15: Metrics,
    m1h: Metrics,
    m4h: Metrics,
    market: MarketContext,
    tags: Sequence[str],
    learner: LearningEngine,
    cfg: Settings,
) -> Tuple[float, float, float, float, str]:
    risk = max(entry - stop, entry * 0.005)
    strength = 0.0
    strength += 0.8 if m15.ema9 > m15.ema20 > m15.ema50 else 0
    strength += 0.8 if m1h.ema20 > m1h.ema50 else 0
    strength += 0.6 if m4h.price > m4h.ema20 else 0
    strength += 0.7 if m15.adx14 >= 25 and m15.plus_di > m15.minus_di else 0
    strength += 0.6 if max(m15.rvol, m15.quote_rvol) >= 1.6 else 0
    strength += 0.4 if m15.macd_hist > m15.macd_hist_previous > 0 else 0
    strength += 0.4 if market.btc_15m_ok and market.btc_1h_ok else 0
    strength -= 0.5 if m15.rsi14 > 70 else 0
    strength -= 0.5 if m15.upper_wick_ratio > 0.5 else 0

    base_r = 1.8 + strength * 0.65
    base_r = max(cfg.min_risk_reward, min(5.5, base_r))
    adaptive_r = learner.target_multiplier(tags, base_r)

    resistance_candidates = [x for x in [m15.recent_high_20, m15.recent_high_50, m1h.recent_high_20, m4h.recent_high_20] if x > entry * 1.003]
    resistance_target = min(resistance_candidates) if resistance_candidates else entry + adaptive_r * risk
    projected_target = entry + adaptive_r * risk

    # Yakın direnç hedefi boğmasın: direnç çok yakınsa kırılım payıyla genişlet.
    if resistance_target < entry + cfg.min_risk_reward * risk:
        first_target = entry + cfg.min_risk_reward * risk
    else:
        first_target = min(projected_target, resistance_target + 0.30 * m15.atr14)

    min_target = entry * (1 + cfg.min_initial_target_percent / 100)
    max_target = entry * (1 + cfg.max_initial_target_percent / 100)
    first_target = max(min_target, min(max_target, first_target))

    # Güçlü trendlerde hedef bölgesinin üst sınırı daha geniş; sabit satış yok.
    extension_r = adaptive_r + 1.2 + max(0, strength - 2.0) * 0.8
    zone_high = min(entry * 1.35, entry + extension_r * risk)
    trail_activation_r = max(1.0, min(1.8, 1.45 - strength * 0.08))
    trail_atr_multiplier = max(1.15, min(2.2, 2.05 - strength * 0.16))
    label = "GÜÇLÜ TREND" if strength >= 3.0 else "ORTA-GÜÇLÜ" if strength >= 2.0 else "KONTROLLÜ"
    return first_target, zone_high, trail_activation_r, trail_atr_multiplier, label


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
    if m15.atr_percent <= 0.15 or m15.atr_percent > 6:
        return None

    score = 0.0
    reasons: List[str] = []
    warnings: List[str] = []

    if m15.ema9 > m15.ema20 > m15.ema50:
        score += 16; reasons.append("15 dk EMA dizilimi güçlü")
    elif m15.ema20 > m15.ema50:
        score += 10; reasons.append("15 dk trend yukarı")
    else:
        score -= 15

    if m1h.ema20 > m1h.ema50:
        score += 15; reasons.append("1 saat trend yukarı")
    else:
        score -= 15

    if m4h.price > m4h.ema20:
        score += 8; reasons.append("4 saat ana trend destekli")
    else:
        score -= 8

    effective_rvol = max(m15.rvol, m15.quote_rvol)
    if 1.35 <= effective_rvol <= 4.5:
        score += 18; reasons.append(f"göreceli hacim {effective_rvol:.1f}x")
    elif 1.05 <= effective_rvol < 1.35:
        score += 7
    elif effective_rvol > 6:
        score -= 12; warnings.append("hacim aşırı patlamış")
    else:
        score -= 8

    if m15.adx14 >= 25 and m15.plus_di > m15.minus_di:
        score += 10; reasons.append(f"ADX trend gücü {m15.adx14:.0f}")
    elif m15.plus_di < m15.minus_di:
        score -= 7

    if 51 <= m15.rsi14 <= 67:
        score += 12; reasons.append(f"RSI sağlıklı {m15.rsi14:.0f}")
    elif m15.rsi14 > 73:
        score -= 15; warnings.append("RSI aşırı ısınmış")

    if m15.macd_hist > 0 and m15.macd_hist >= m15.macd_hist_previous:
        score += 10; reasons.append("MACD ivmesi pozitif")
    elif m15.macd_hist < m15.macd_hist_previous:
        score -= 5

    if m15.bb_width_previous < 0.07 and m15.bb_width > m15.bb_width_previous:
        score += 10; reasons.append("Bollinger sıkışması açılıyor")

    breakout_distance = (price - m15.prior_high_20) / max(m15.prior_high_20, 1e-12) * 100
    if -1.0 <= breakout_distance <= 1.8:
        score += 9; reasons.append("kırılıma yakın kontrollü bölge")
    elif breakout_distance > 3.5:
        score -= 12; warnings.append("hareket fazla uzamış")

    if m15.upper_wick_ratio > 0.55:
        score -= 9; warnings.append("üst fitil satış baskısı")
    if m15.close_position > 0.75 and m15.body_percent >= 0.5:
        score += 5
    if m5.roc5 > 3.5:
        score -= 7; warnings.append("5 dk ivme aşırı hızlı")

    if 0.3 <= change <= 7.5:
        score += 8; reasons.append(f"24s hareket kontrollü %{change:.1f}")
    elif change > 11:
        score -= 10

    if cfg.btc_filter_enabled:
        if market.btc_15m_ok and market.btc_1h_ok:
            score += 8; reasons.append("BTC yönü destekli")
        elif not market.btc_1h_ok:
            score -= 7; warnings.append("BTC 1 saat zayıf")

    atr = max(m15.atr14, price * 0.0035)
    support = max(m15.ema20, m15.bb_mid, m15.recent_low_20 + 0.30 * (m15.recent_high_20 - m15.recent_low_20))
    entry_high = max(ask, price)
    entry_low = max(support, price - 0.30 * atr)
    structural_stop = min(m15.ema20 - 0.55 * atr, m15.recent_low_20 - 0.15 * atr)
    max_stop = entry_high * (1 - cfg.max_stop_percent / 100)
    min_stop = entry_high * (1 - cfg.min_stop_percent / 100)
    initial_stop = max(max_stop, min(min_stop, structural_stop))
    if initial_stop >= entry_low:
        initial_stop = entry_high - max(0.85 * atr, entry_high * cfg.min_stop_percent / 100)

    stop_percent = (entry_high - initial_stop) / entry_high * 100
    if not (cfg.min_stop_percent <= stop_percent <= cfg.max_stop_percent):
        return None

    tags = setup_tags(m5, m15, m1h, m4h, market)
    learning_bonus = learner.score_bonus(tags)
    raw_score = score
    score += learning_bonus
    if learning_bonus:
        reasons.append(f"öğrenme etkisi {learning_bonus:+.1f} puan")
    if score < cfg.min_score:
        return None

    initial_target, zone_high, trail_activation_r, trail_atr_multiplier, strength = dynamic_target_plan(
        entry_high, initial_stop, m5, m15, m1h, m4h, market, tags, learner, cfg
    )
    risk = entry_high - initial_stop
    rr = (initial_target - entry_high) / max(risk, 1e-12)
    if rr < cfg.min_risk_reward:
        return None

    return Signal(
        symbol=symbol,
        score=round(min(score, 100), 1),
        raw_score=round(raw_score, 1),
        learning_bonus=round(learning_bonus, 1),
        price=price,
        entry_low=entry_low,
        entry_high=entry_high,
        initial_stop=initial_stop,
        initial_target=initial_target,
        target_zone_high=zone_high,
        stop_percent=stop_percent,
        initial_target_percent=(initial_target - entry_high) / entry_high * 100,
        risk_reward=rr,
        trail_activation_r=trail_activation_r,
        trail_atr_multiplier=trail_atr_multiplier,
        max_hold_hours=cfg.max_hold_hours,
        setup_strength=strength,
        change_24h=change,
        quote_volume_24h=quote_volume,
        spread_percent=spread,
        reasons=reasons[:10],
        warnings=warnings[:5],
        feature_tags=tags,
    )


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
            {"chat_id": self.cfg.telegram_chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        )

    def startup(self) -> None:
        self.send(
            "🤖 <b>Coin Radar AI başladı</b>\n\n"
            "• Sabit %4 hedef kaldırıldı\n"
            "• Dinamik hedef + hedef bölgesi\n"
            "• Takip eden stop yönetimi\n"
            "• Kontrollü çevrim içi öğrenme\n"
            "• Günlük sinyal sınırı yok\n\n"
            "Otomatik emir vermez."
        )

    def signal(self, s: Signal, market: MarketContext) -> None:
        reasons = "\n".join(f"• {x}" for x in s.reasons)
        warnings = "\n\n⚠️ <b>Uyarılar</b>\n" + "\n".join(f"• {x}" for x in s.warnings) if s.warnings else ""
        self.send(
            "🚨 <b>COIN RADAR AI — ADAY</b>\n\n"
            f"<b>{s.symbol.replace('TRY', '/TRY')}</b> | {s.setup_strength}\n"
            f"Puan: <b>{s.score:.0f}/100</b> | 24s: <b>%{s.change_24h:.2f}</b>\n\n"
            f"🟢 Giriş: <b>{format_price(s.entry_low)} – {format_price(s.entry_high)}</b>\n"
            f"🔴 İlk stop: <b>{format_price(s.initial_stop)}</b> (%{s.stop_percent:.2f})\n"
            f"🎯 İlk karar bölgesi: <b>{format_price(s.initial_target)}</b> (%{s.initial_target_percent:.2f})\n"
            f"🚀 Trend sürerse hedef alanı: <b>{format_price(s.initial_target)} – {format_price(s.target_zone_high)}</b>\n"
            f"🛡 Takip eden stop: <b>{s.trail_activation_r:.2f}R</b> sonrası, <b>{s.trail_atr_multiplier:.2f} ATR</b>\n"
            f"⚖️ İlk risk/ödül: <b>{s.risk_reward:.2f}</b>\n"
            f"⏱ Azami izleme: <b>{s.max_hold_hours} saat</b>\n\n"
            f"<b>Neden?</b>\n{reasons}{warnings}\n\n"
            f"BTC: {market.btc_symbol} | 24s %{market.btc_change_24h:.2f}\n\n"
            "İlk hedef sabit satış emri değildir. Güçlü trendde bot stopu yükseltip hareketi takip eder."
        )

    def trail_update(self, symbol: str, old_stop: float, new_stop: float, highest: float, r_multiple: float) -> None:
        self.send(
            f"🛡 <b>{symbol.replace('TRY','/TRY')} STOP GÜNCELLE</b>\n\n"
            f"Eski stop: {format_price(old_stop)}\n"
            f"Yeni stop: <b>{format_price(new_stop)}</b>\n"
            f"Görülen tepe: {format_price(highest)}\n"
            f"Hareket: <b>{r_multiple:.2f}R</b>\n\n"
            "Momentum sürdüğü için erken tam satış yerine kâr korunuyor."
        )

    def closed(self, symbol: str, reason: str, exit_price: float, realized: float, mfe: float) -> None:
        icon = "✅" if realized > 0 else "❌" if realized < 0 else "➖"
        self.send(
            f"{icon} <b>{symbol.replace('TRY','/TRY')} KÂĞIT İŞLEM SONUCU</b>\n\n"
            f"Çıkış nedeni: <b>{reason}</b>\n"
            f"Çıkış: {format_price(exit_price)}\n"
            f"Sonuç: <b>%{realized:.2f}</b>\n"
            f"İşlem içi maksimum potansiyel: <b>%{mfe:.2f}</b>\n\n"
            "Sonuç öğrenme verisine işlendi."
        )

    def heartbeat(self, scanned: int, candidates: int, open_count: int, elapsed: float) -> None:
        self.send(
            "💓 <b>Coin Radar AI çalışıyor</b>\n\n"
            f"İncelenen: {scanned}\nAday: {candidates}\nAçık kâğıt işlem: {open_count}\n"
            f"Tur süresi: {elapsed:.1f} sn"
        )

    def performance(self, summary: Dict[str, float]) -> None:
        self.send(
            "📊 <b>Coin Radar öğrenme özeti</b>\n\n"
            f"Kapanan: {int(summary['closed'])}\n"
            f"Kazanan: {int(summary['wins'])}\n"
            f"Kaybeden: {int(summary['losses'])}\n"
            f"Başarı: %{summary['win_rate']:.1f}\n"
            f"Ortalama sonuç: %{summary['avg_return']:.2f}\n"
            f"Ortalama maksimum potansiyel: %{summary['avg_mfe']:.2f}"
        )


# ============================================================
# POSITION REVIEW
# ============================================================

def review_positions(exchange: BinanceTRClient, learner: LearningEngine, telegram: TelegramNotifier) -> int:
    """Açık işlemleri mum mum yeniden oynatır; gelecekte oluşan stopu geçmişe uygulamaz."""
    rows = learner.open_rows()
    for row in rows:
        try:
            df = exchange.klines(row["symbol"], "5m", 500)
            bars = df[df["open_time"] >= int(row["created_at"] * 1000)]
            if bars.empty:
                continue

            entry = float(row["entry"])
            risk = float(row["risk"])
            initial_stop = float(row["initial_stop"])
            active_stop = initial_stop
            highest = entry
            lowest = entry
            target_touched = 0
            last_notified = float(row["last_notified_stop"] or initial_stop)
            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None

            # Her mumda önce mevcut stop kontrol edilir, sonra mum kapanışındaki
            # bilgilerle bir sonraki mum için stop yükseltilir.
            for i, (_, candle) in enumerate(bars.iterrows()):
                low = float(candle["low"])
                high = float(candle["high"])
                close = float(candle["close"])
                lowest = min(lowest, low)

                if low <= active_stop:
                    exit_reason = "TAKİP EDEN STOP" if active_stop > initial_stop else "İLK STOP"
                    exit_price = active_stop
                    break

                highest = max(highest, high)
                r_multiple = (highest - entry) / max(risk, 1e-12)
                if highest >= float(row["initial_target"]):
                    target_touched = 1

                history = bars.iloc[: i + 1]
                if len(history) >= 60:
                    metrics = calculate_metrics(history)
                    next_stop = active_stop
                    if r_multiple >= float(row["trail_activation_r"]):
                        atr_stop = highest - float(row["trail_atr_multiplier"]) * metrics.atr14
                        structure_stop = max(
                            metrics.ema20 - 0.25 * metrics.atr14,
                            entry + 0.05 * risk,
                        )
                        next_stop = max(next_stop, min(atr_stop, structure_stop))
                    if target_touched:
                        next_stop = max(next_stop, entry + 0.50 * risk)
                    # Stop hiçbir zaman son kapanışın üzerine çıkarılmaz.
                    active_stop = min(next_stop, close - 0.05 * metrics.atr14)

            mfe = (highest - entry) / entry * 100
            mae = (lowest - entry) / entry * 100
            r_multiple = (highest - entry) / max(risk, 1e-12)

            notify_threshold = max(entry * 0.0025, 0.20 * risk)
            if exit_reason is None and active_stop - last_notified >= notify_threshold:
                telegram.trail_update(row["symbol"], last_notified, active_stop, highest, r_multiple)
                last_notified = active_stop

            latest_close = float(bars.iloc[-1]["close"])
            age_hours = (time.time() - float(row["created_at"])) / 3600
            latest_metrics = calculate_metrics(bars)
            if exit_reason is None and age_hours >= float(row["max_hold_hours"]):
                exit_reason = "SÜRE DOLDU"
                exit_price = latest_close
            if (
                exit_reason is None
                and target_touched
                and latest_metrics.macd_hist < latest_metrics.macd_hist_previous
                and latest_metrics.rsi14 < 52
                and latest_close < latest_metrics.ema9
            ):
                exit_reason = "MOMENTUM ZAYIFLADI"
                exit_price = latest_close

            learner.update_live(
                row["id"], highest, lowest, mfe, mae, active_stop,
                target_touched, last_notified,
            )
            if exit_reason is not None and exit_price is not None:
                realized = (exit_price - entry) / entry * 100
                outcome = "WIN" if realized >= 0.5 else "LOSS" if realized <= -0.5 else "NEUTRAL"
                mutable = dict(row)
                mutable["mfe_percent"] = mfe
                learner.close(_DictRow(mutable), outcome, exit_reason, exit_price)
                telegram.closed(row["symbol"], exit_reason, exit_price, realized, mfe)
        except Exception as exc:
            logging.warning("Açık pozisyon değerlendirilemedi %s: %s", row["symbol"], exc)
    return len(learner.open_rows())


class _DictRow(dict):
    """sqlite Row ile aynı [] erişimini sağlayan küçük adaptör."""
    pass


# ============================================================
# APP
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
        self.learner = LearningEngine(cfg)
        self.last_review = 0.0
        self.last_heartbeat = 0.0
        self.last_report = 0.0

    def scan_once(self) -> Tuple[int, int]:
        started = time.time()
        tickers = self.exchange.tickers_24h()
        books = self.exchange.book_tickers()
        market = build_market_context(self.exchange, tickers)

        ranked: List[Tuple[float, str]] = []
        for symbol, ticker in tickers.items():
            if symbol == "BTCTRY" or symbol not in books:
                continue
            pre = prefilter_score(ticker, books[symbol], self.cfg)
            if pre is not None:
                ranked.append((pre, symbol))
        ranked.sort(reverse=True)
        selected = [s for _, s in ranked[: self.cfg.max_prefilter_symbols]]

        candidates: List[Signal] = []
        scanned = 0
        for index, symbol in enumerate(selected, start=1):
            try:
                m5 = calculate_metrics(self.exchange.klines(symbol, "5m"))
                m15 = calculate_metrics(self.exchange.klines(symbol, "15m"))
                m1h = calculate_metrics(self.exchange.klines(symbol, "1h"))
                m4h = calculate_metrics(self.exchange.klines(symbol, "4h"))
                signal = evaluate_signal(
                    symbol, tickers[symbol], books[symbol], m5, m15, m1h, m4h,
                    market, self.cfg, self.learner,
                )
                scanned += 1
                if signal:
                    candidates.append(signal)
                if index % 10 == 0:
                    logging.info("Tarama %s/%s", index, len(selected))
                time.sleep(self.cfg.request_pause_seconds)
            except Exception as exc:
                logging.warning("%s atlandı: %s", symbol, exc)

        candidates.sort(key=lambda s: (s.score, s.risk_reward, s.quote_volume_24h), reverse=True)
        for signal in candidates:
            if self.learner.recently_sent(
                signal.symbol, self.cfg.cooldown_minutes, signal.score,
                self.cfg.resignal_score_improvement,
            ):
                continue
            self.telegram.signal(signal, market)
            self.learner.record(signal)
            logging.info("Sinyal: %s puan %.1f", signal.symbol, signal.score)

        elapsed = time.time() - started
        now = time.time()
        open_count = len(self.learner.open_rows())
        if now - self.last_heartbeat >= self.cfg.heartbeat_minutes * 60:
            self.telegram.heartbeat(scanned, len(candidates), open_count, elapsed)
            self.last_heartbeat = now
        if now - self.last_report >= self.cfg.performance_report_hours * 3600:
            self.telegram.performance(self.learner.summary())
            self.last_report = now
        logging.info("Tur bitti: taranan=%s aday=%s açık=%s süre=%.1f", scanned, len(candidates), open_count, elapsed)
        return scanned, len(candidates)

    def run(self) -> None:
        logging.info("Coin Radar AI başlatılıyor")
        if self.cfg.send_startup_message:
            self.telegram.startup()
        while True:
            started = time.time()
            try:
                now = time.time()
                if now - self.last_review >= self.cfg.learning_review_minutes * 60:
                    review_positions(self.exchange, self.learner, self.telegram)
                    self.last_review = now
                self.scan_once()
            except KeyboardInterrupt:
                logging.info("Kapatıldı")
                return
            except Exception:
                logging.exception("Ana döngü hatası")
            elapsed = time.time() - started
            time.sleep(max(5.0, self.cfg.scan_interval_seconds - elapsed))


if __name__ == "__main__":
    RadarApp(Settings.from_env()).run()
