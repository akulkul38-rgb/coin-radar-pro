# Coin Radar AI — Stable Adaptive + GitHub Persistence

Binance TR TRY paritelerini 5 dk, 15 dk, 1 saat ve 4 saat zaman dilimlerinde tarar. Sabit `%4` hedef kullanmaz; hedef alanını ATR, dirençler, hacim, trend gücü, BTC bağlamı ve geçmiş sonuçlara göre dinamik belirler. Güçlü harekette takip eden stopla pozisyonu taşır.

## Railway zorunlu değişkenleri

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## GitHub kalıcı öğrenme değişkenleri

- `GITHUB_BACKUP_ENABLED=true`
- `GITHUB_BACKUP_TOKEN=<GitHub fine-grained token>`
- `GITHUB_BACKUP_REPOSITORY=kullaniciadi/repo-adi`
- `GITHUB_BACKUP_BRANCH=main`
- `GITHUB_BACKUP_DB_PATH=radar-data/radar_learning.db`
- `GITHUB_BACKUP_INTERVAL_MINUTES=30`
- `GITHUB_BACKUP_MIN_CHANGES=10`
- `LEARNING_DB_PATH=/tmp/radar_learning.db`

Token için yalnızca seçilen repository üzerinde **Contents: Read and write** izni yeterlidir. Token'ı koda veya GitHub repository içine yazmayın; yalnızca Railway Variables bölümüne ekleyin.

Bot açılışta GitHub'daki son SQLite yedeğini geri yükler. Çalışırken en geç 30 dakikada bir veya 10 veritabanı değişikliğinde bir güvenli SQLite snapshot'ını GitHub'a gönderir. Railway yeniden başlasa veya yeniden deploy edilse bile öğrenme geçmişi geri gelir.

## Diğer önerilen değişkenler

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

## Çalıştırma

```bash
pip install -r requirements.txt
python main.py
```

## Güvenlik

- Otomatik emir vermez.
- Binance API anahtarı istemez.
- Kazanç garantisi vermez.
- Öğrenme sistemi maksimum stop sınırını gevşetemez.
