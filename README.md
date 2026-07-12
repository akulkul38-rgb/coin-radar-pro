# Coin Radar AI — Stable Adaptive Edition

Bu sürüm Binance TR TRY paritelerini 5 dk, 15 dk, 1 saat ve 4 saat zaman dilimlerinde tarar.

## Ana fark

Sabit `%4` satış hedefi kaldırılmıştır. İlk karar bölgesi ve hedef alanı şu verilerden dinamik hesaplanır:

- ATR ve stop mesafesi
- Yakın/uzak dirençler
- EMA trend dizilimi
- ADX trend gücü
- RSI, MACD ve Bollinger yapısı
- Göreceli hacim
- BTC piyasa bağlamı
- Geçmiş sinyallerin MFE ve getiri sonuçları

İlk hedefe ulaşılması otomatik tam satış anlamına gelmez. Trend güçlü kalırsa takip eden stop yükseltilir.

## Railway değişkenleri

Zorunlu:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Önerilen:

- `SCAN_INTERVAL_SECONDS=180`
- `MIN_SCORE=80`
- `MIN_24H_TRY_VOLUME=15000000`
- `MAX_SPREAD_PERCENT=0.70`
- `MAX_STOP_PERCENT=3.20`
- `MIN_STOP_PERCENT=0.70`
- `MIN_RISK_REWARD=1.60`
- `MAX_HOLD_HOURS=12`
- `LEARNING_ENABLED=true`
- `LEARNING_MIN_SAMPLES=20`
- `LEARNING_DB_PATH=/data/radar_learning.db`

Railway Volume mount path: `/data`

## Kurulum

```bash
pip install -r requirements.txt
python main.py
```

## Güvenlik

- Otomatik emir vermez.
- Binance API anahtarı istemez.
- Kazanç garantisi vermez.
- Öğrenme sistemi maksimum stop sınırını gevşetemez.
