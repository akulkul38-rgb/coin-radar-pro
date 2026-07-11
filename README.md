# Coin Radar v1

Bu proje Binance TR'deki TRY paritelerini otomatik tarar ve uygun teknik kurulum bulduğunda Telegram'a bildirim gönderir.

## Ne yapar?

- TRY paritelerini tarar
- 15 dakika ve 1 saat mumlarını inceler
- EMA20/EMA50, RSI, Bollinger, ATR ve göreceli hacmi değerlendirir
- Günlük hareketi aşırı uzamış coinleri eler
- Giriş, stop, kâr alma ve maksimum bekleme süresi üretir
- Aynı coin için bildirim yağmurunu önlemek üzere bekleme süresi uygular

Bu sürüm yalnızca **uyarı üretir**. Otomatik alım-satım yapmaz.

## 1) Telegram botunu oluştur

1. Telegram'da `@BotFather` hesabını aç.
2. `/newbot` yaz.
3. Bot adı ve kullanıcı adı seç.
4. Verilen token'ı sakla.
5. Yeni botuna mesaj olarak `/start` gönder.
6. Tarayıcıda şu adresi aç:
   `https://api.telegram.org/botTOKEN/getUpdates`
7. Sonuçtaki `chat.id` değerini kaydet.

## 2) Bilgisayarda çalıştırma

Python 3.11+ kurulu olmalı.

```bash
pip install -r requirements.txt
```

Mac/Linux:
```bash
export TELEGRAM_BOT_TOKEN="TOKEN"
export TELEGRAM_CHAT_ID="CHAT_ID"
python main.py
```

Windows PowerShell:
```powershell
$env:TELEGRAM_BOT_TOKEN="TOKEN"
$env:TELEGRAM_CHAT_ID="CHAT_ID"
python main.py
```

## 3) Railway'de 7/24 çalıştırma

1. GitHub'da yeni bir repo oluştur.
2. Bu klasördeki dosyaları repoya yükle.
3. Railway'de `New Project > Deploy from GitHub Repo` seç.
4. Variables kısmına şunları ekle:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `SCAN_INTERVAL_SECONDS=60`
   - `MIN_TRY_VOLUME=10000000`
   - `MIN_SCORE=78`
   - `MAX_24H_CHANGE=12`
   - `COOLDOWN_MINUTES=180`
5. Start Command:
   `python main.py`
6. Deploy et.

## Ayar mantığı

- `MIN_SCORE`: Yükseldikçe daha az ama daha seçici sinyal gelir.
- `MIN_TRY_VOLUME`: Likiditesi düşük coinleri eler.
- `MAX_24H_CHANGE`: Zaten aşırı yükselmiş coinleri kovalamayı engeller.
- `COOLDOWN_MINUTES`: Aynı coin için tekrar bildirim süresi.

İlk kullanım önerisi:
- `MIN_SCORE=78`
- `MIN_TRY_VOLUME=10000000`
- `MAX_24H_CHANGE=12`
- `COOLDOWN_MINUTES=180`

## Önemli

- Bu sistem garanti kazanç sağlamaz.
- Otomatik sinyal, işlem emri değildir.
- Bildirim geldiğinde grafik, haber, BTC yönü ve emir defteri ayrıca kontrol edilmelidir.
- API hız sınırlarına saygı gösterir; çok sık istek atacak şekilde değiştirmeyin.
