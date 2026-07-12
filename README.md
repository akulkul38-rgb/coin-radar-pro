# Coin Radar Stable v1

Bu sürüm, tekrar tekrar tüm projeyi değiştirmemek için hazırlanmış kararlı
tek-dosya sürümüdür.

## Dahil edilenler

- Binance TR TRY paritelerini otomatik keşfetme
- 5 dk, 15 dk, 1 saat, 4 saat analiz
- EMA, RSI, MACD, Bollinger, ATR, RVOL, spread ve BTC filtresi
- Giriş, stop, tam satış, risk/ödül ve maksimum bekleme süresi
- Günlük sinyal sınırı yok
- Coin bazlı tekrar sinyali kontrolü
- Kâğıt işlem sonucu takibi
- Kendi performansından sınırlı öğrenme
- Telegram sağlık mesajı
- Telegram performans raporu
- Railway Variables üzerinden ayar değiştirme
- SQLite kalıcı öğrenme verisi

## Bundan sonra kod değiştirmeden ayarlanabilecek başlıca değerler

- MIN_SCORE
- MIN_24H_TRY_VOLUME
- MAX_24H_CHANGE
- MAX_SPREAD_PERCENT
- MAX_STOP_PERCENT
- MIN_RISK_REWARD
- PREFERRED_TARGET_PERCENT
- MAX_HOLD_HOURS
- COOLDOWN_MINUTES
- SCAN_INTERVAL_SECONDS

## Railway Volume

Öğrenme verisinin deploy sonrasında silinmemesi için Worker servisine bir
Volume bağla ve mount path olarak:

`/data`

kullan.

Kod veritabanını burada tutar:

`/data/radar_learning.db`

## İlk güvenli kullanım

İlk 30–50 sonuç kâğıt işlem verisi olarak toplanmalı. Öğrenme katmanı en az
10 örnek oluşmadan herhangi bir özellik ağırlığını değiştirmez.

## Güvenlik

- Telegram token'ını GitHub'a yazma.
- Binance API anahtarı gerekmez.
- Sistem otomatik emir vermez.
- Bu sistem kazanç garantisi vermez.
