# Coin Radar Pro v3

Bu sürüm, Binance TR'nin `www.binance.tr/open/v1/common/symbols` uç noktasını
kullanmaz. Bazı bulut bölgelerinde görülen HTTP 451 ve başlangıçta takılma
sorununu önlemek için TRY sembolleri doğrudan market-wide ticker yanıtından
keşfedilir.

## Önemli özellikler

- Günlük sinyal sınırı yoktur.
- Kalite filtresini geçen bütün adaylar gönderilir.
- Aynı kurulumun spam olmasını engellemek için coin bazlı cooldown vardır.
- Skor anlamlı biçimde yükselirse cooldown dolmadan yeniden sinyal gelebilir.
- Başlangıç mesajı ve periyodik heartbeat mesajı gönderir.
- Ayrıntılı Railway logları üretir.
- Otomatik alım-satım yapmaz.

## GitHub güncellemesi

Mevcut repository'de şu dosyaları yenileriyle değiştir:

- `main.py`
- `requirements.txt`
- `railway.json`
- `Procfile`

Railway mevcut Telegram değişkenlerini korur. Yeni filtre değişkenleri
eklenmezse kod güvenli varsayılanlarla çalışır.

## Önerilen Railway değişkenleri

`settings.template` içindeki değerler kullanılabilir.

İlk testte Telegram'a:
- `Coin Radar Pro v3 çalışıyor`
- ardından en geç HEARTBEAT_MINUTES süresinde durum mesajı

gelmelidir.
