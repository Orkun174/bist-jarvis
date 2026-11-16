# bist-jarvis: Bütün BIST'i kapsayan, her gün kendini eğiten sistem — Tasarım

Tarih: 2026-11-16 · Durum: onay bekliyor

## 1. Amaç ve neden

Kullanıcı, bist-jarvis'in 20 hisseyle sınırlı kalmamasını, bütün BIST'i
taramasını ve **her işlem günü güvenilir yeni veriyle kendini eğitmesini**
istiyor. Çıktıyı iki yerden kullanacak:

- **Telegram:** Açtığında o günün önerilen hisselerini hazır görmek.
- **PC uygulaması:** Hisseleri araştırmak; haberlerini ve indikatörlerinin
  durumunu görmek.

Kullanıcının açık şartı: **manipülasyona açık hisseler eğitimde yer almasın.**

## 2. Bugünkü durum

- `daily_scan.py` + `core/daily_radar.py`: GitHub Actions hafta içi 18:30'da
  (TRT) sabit 20 hisseyi sabit teknik kurallarla puanlar, ilk 3'ü Telegram'a
  yollar. ML modelini kullanmaz.
- `train_xgboost.py`: 20 hisselik havuzda XGBoost (CPCV + Optuna + SHAP
  seçimi) eğitir; elle çalıştırılır. Hedef: 5 seanslık getirinin XU100'ü
  geçmesi. Özellik sözleşmesi `bist-dynamic-alpha-v3` (10 yerel + 8 makro).
- Tahminlerin sonradan tutup tutmadığını kaydeden hiçbir şey yok.
- `portfolio_backtest_results/summary.json` eğitim tarihleriyle çakışıyor
  (`strictly_post_training_dates: false`); gerçek performans bilinmiyor.
- `app.py` (Streamlit) + `launcher.py` (pywebview, `.exe`): tek tek hisse
  analizi, indikatör grafiği, Google News RSS haberleri, Ollama yorumu.
- Depoda takip edilmemesi gereken ~5.900 dosya var (`build/`, `backups/`,
  `__pycache__/`, `work/backfill/*/http/`, `.exe`); `.gitignore` yok.

## 3. Hedefler, hedef olmayanlar, kaçınılacaklar

**Hedefler**
1. Evren: bütün BIST pay senetleri; her gün güncellenen liste.
2. Uygunluk filtresi: manipülasyona açık hisseler eğitime ve önerilere girmez.
3. Her işlem günü: veri güncelle → kontrol et → vadesi dolan tahminleri
   notla → aday model eğit → sınavdan geçerse terfi ettir → öneri üret.
4. Karne: her tahmin kayıtlı, gerçekleşen sonuçla karşılaştırılmış.
5. Telegram'da günlük öneri mesajı.
6. PC uygulamasında her BIST hissesi için araştırma ekranı.

**Hedef olmayanlar**
- Emir gönderme / aracı kurum bağlantısı.
- Gün içi (intraday) sinyal.
- Komut alan (çift yönlü) Telegram botu.
- Git geçmişini yeniden yazarak eski büyük dosyaları silmek.

**Kaçınılacaklar (çalışsa bile başarısızlık sayılır)**
- Yeni modelin sınavsız devreye girip sistemi sessizce kötüleştirmesi.
- Geleceğe bakma (lookahead) sızıntısı: eğitimde veya sınavda, o gün
  bilinemeyecek veri kullanılması.
- Filtrenin dışladığı bir hissenin önerilmesi.
- Bozuk/eksik verinin fark edilmeden eğitime girmesi.
- Çıktının "kesin kazanç" gibi sunulması. Her mesajda "yatırım tavsiyesi
  değildir" notu ve gerçek karne bulunur.

## 4. Kısıtlar

- GitHub Actions ücretsiz koşucu: 2 çekirdek, iş başına 45 dk sınır
  (mevcut ayar). Günlük iş bu sürede bitmeli.
- Veri kaynakları ücretsiz: Yahoo Finance (`yfinance`), İş Yatırım
  (`isyatirimhisse`), KAP, Google News RSS. Hepsi kesintiye uğrayabilir.
- Türkçe duygu modeli (`savasy/bert-base-turkish-sentiment-cased`) CPU'da
  yavaş; 600 hisse için her gün çalıştırılamaz.
- Depo herkese açık: gizli anahtarlar yalnızca Actions secrets'ta durur.

## 5. Yaklaşımlar

| Yaklaşım | Artı | Eksi |
|---|---|---|
| **A. Bulutta GitHub Actions (seçildi)** | PC kapalıyken de çalışır; ücretsiz; mevcut iş akışının devamı | Zayıf makine; haber analizi dar tutulmalı |
| B. Kullanıcının PC'sinde zamanlanmış görev | Güçlü donanım, geniş haber analizi | PC kapalıysa o gün eğitim ve mesaj yok |
| C. Kiralık sunucu | Güçlü ve hep açık | Aylık maliyet, bakım yükü |

Kullanıcı **A**'yı seçti: "her gün" sözünü ancak bu tutuyor.

## 6. Mimari

```
GitHub Actions (hafta içi 18:30 TRT)
  1 evren + fiyat güncelle ─► 2 veri kontrolü ─► 3 uygunluk filtresi
  4 vadesi dolan tahminleri notla (karne)
  5 aday model eğit ─► 6 sınav (şampiyon vs aday) ─► terfi / ret
  7 bugünün tahminleri (şampiyon model) ─► deftere yaz
  8 ilk adaylar için haber + duygu ─► 9 Telegram mesajı
  10 hepsini `data` dalına yaz
                         │
            `data` dalı (aynı depo)
                         │
PC uygulaması (Streamlit + pywebview) ── açılışta `data` dalını indirir
```

### 6.1 Depolama: `data` dalı
*(Benim kararım.)* Üretilen veri `main`'i kirletmesin diye aynı depoda ayrı,
geçmişsiz (orphan) bir `data` dalında tutulur. İş akışına `contents: write`
izni verilir; her çalışma tek commit atar.

```
prices/<TICKER>.parquet        düzeltilmiş günlük OHLCV
universe/universe.parquet      hisse listesi + günlük uygunluk + neden
quality/<YYYY-MM-DD>.json      veri kontrol raporu
ledger/predictions.parquet     her günün tahminleri ve gerçekleşen sonuç
models/champion.json           etkin model (+ metadata)
models/history.jsonl           her günün aday/şampiyon kararı ve skorları
scorecard/scorecard.json       özet karne
daily/<YYYY-MM-DD>.json        o günün önerileri (Telegram + PC için)
```

Depo herkese açık olduğu için PC uygulaması bu dosyaları oturum açmadan
HTTPS ile indirebilir.

### 6.2 Evren ve veri kontrolü
- Hisse listesi her çalışmada İş Yatırım'dan alınır.
- Geçmiş fiyat: Yahoo, `auto_adjust=True` (mevcut sözleşme). İlk kurulumda
  5 yıl; sonra günde yalnızca eksik seanslar eklenir.
- Son seansın kapanışı İş Yatırım'la karşılaştırılır (mevcut `fetch_t0`
  mantığı). Fark %1'i aşarsa o hissenin o günü **karantinaya** alınır:
  tahmin üretilmez, eğitime girmez, rapora yazılır.
- Diğer kontroller: eksik seans (XIST takvimine göre), sıfır/negatif fiyat,
  sıfır hacim, bir günde %±25'i aşan ve düzeltmeyle açıklanamayan sıçrama.
- Hisselerin %20'sinden fazlası karantinaya düşerse o gün **eğitim
  yapılmaz**; şampiyon model kalır, Telegram mesajı bunu belirtir.

### 6.3 Uygunluk (manipülasyon) filtresi
Bir hisse bir gün için "uygun" sayılmak için hepsini sağlamalı:

1. En az 252 seans geçmiş.
2. Son 20 seansın medyan günlük işlem hacmi (TL) eşiğin üstünde.
   *(Benim kararım: başlangıç eşiği 50 milyon TL; ayar dosyasında.)*
3. Yakın İzleme Pazarı veya Piyasa Öncesi İşlem Platformu'nda değil.
4. Üzerinde etkin VBTS tedbiri (brüt takas, kredili işlem/açığa satış
   yasağı, tek fiyat vb.) yok.
5. Son 20 seansta 5'ten fazla tavan/taban kapanışı yok.

Eğitimde 1, 2 ve 5 **o günün bilgisiyle** (point-in-time) hesaplanır. 3 ve
4 için geçmiş listeler güvenilir biçimde bulunamayabilir; bu durumda bu iki
kural yalnızca ileriye dönük uygulanır ve her gün `universe.parquet`'e
kaydedilerek zamanla geçmiş oluşur. *(Uygulayıcı: 3 ve 4 için makineyle
okunabilir resmi kaynak plan aşamasında doğrulanmalı; bulunamazsa
kullanıcıya dönülür, sessizce atlanmaz.)*

Her dışlanan hissenin nedeni kayıtlıdır; PC uygulaması bunu gösterir.

### 6.4 Model ve günlük eğitim
- Tek, havuzlanmış kesitsel XGBoost; mevcut özellik sözleşmesi ve hedef
  korunur (5 seans sonra XU100'ü geçme olasılığı).
- **Her gün:** aday model, şampiyonun hiperparametreleriyle, son 3 yılın
  uygun (hisse, gün) satırlarında yeniden eğitilir. Etiketi henüz
  kesinleşmemiş son 5 seans ve ek 7 seans tampon eğitime girmez
  (mevcut purge/embargo mantığı).
- **Haftada bir (cumartesi):** Optuna ile hiperparametre araması ve SHAP
  özellik seçimi (mevcut `train_xgboost.py` akışı, yeni evrene uyarlanmış).
- Haber duygusu modelin girdisi **değildir** (600 hisse için geçmiş haber
  kapsamı yok). Yalnızca öneri mesajında ve PC'de bilgi olarak gösterilir.

### 6.5 Sınav: şampiyon ve aday
Aday model yalnızca şu durumda şampiyon olur:

- Sınav penceresi: adayın eğitiminde hiç görülmemiş son 60 seans
  (adayın sınav sürümü bu pencereden önce kesilerek eğitilir).
- Ölçüt: günlük ilk-N seçimin XU100'e göre ortalama fazla getirisi
  (20 baz puan işlem maliyeti düşülmüş) ve sıra korelasyonu (rank IC).
- Aday iki ölçütte de şampiyondan iyi olmalı ve fark bootstrap ile
  %80 güvenle sıfırdan büyük olmalı. *(Benim kararım; ayarlanabilir.)*
- Aksi halde şampiyon kalır. Karar ve skorlar `models/history.jsonl`'e
  yazılır; önceki şampiyon saklanır (geri dönüş tek dosya kopyası).

Dürüst beklenti: günlük eğitimde çoğu gün şampiyon değişmeyecek. Sistemin
"kendini iyileştirmesi", yalnızca gerçekten daha iyi olan modelin içeri
alınması demektir.

### 6.6 Tahmin defteri ve karne
- Her gün, uygun her hisse için: tarih, hisse, olasılık, sıra, model
  sürümü, önerildi mi.
- 5 seans sonra gerçekleşen hisse ve XU100 getirisi eklenir.
- Karne (son 20 / 60 / tüm seanslar): önerilerin isabet oranı, ortalama
  fazla getiri, olasılık kalibrasyonu, öneri sayısı.
- İlk gün karne boş olmasın diye son 250 seans için ileriye yürüyen
  (walk-forward) bir geçmiş simülasyonu çalıştırılır ve **"simülasyon"**
  etiketiyle ayrı tutulur; canlı sonuçlarla karıştırılmaz.

### 6.7 Telegram mesajı
Her işlem günü tek mesaj (mevcut `send_report` yolu):

```
📡 BIST Jarvis — 17 Kas 2026 için öneriler
1. XXXXX  olasılık %64 · neden: hacim artışı, EMA20 üstü
   haber: olumlu (3 haber)
2. ...
5. ...
Karne (son 60 seans, canlı): isabet %57 · ort. fazla getiri +%0,8
Model: bugün yeniden eğitildi, şampiyon değişmedi (v12)
Veri: 412 hisse uygun · 9 karantina · 187 filtre dışı
Yatırım tavsiyesi değildir.
```

- *(Benim kararım.)* 5 öneri; yalnızca olasılığı modelin eşiğini geçenler.
  Hiçbiri geçmezse "bugün öneri yok" yazılır, liste doldurulmaz.
- "Neden" satırı: o hissenin en etkili 2-3 SHAP özelliği, sade Türkçe.
- Haber duygusu yalnızca bu 5 hisse için hesaplanır.

### 6.8 PC uygulaması
Mevcut Streamlit + pywebview yapısı korunur, üç ekran olur:

1. **Hisse araştır:** Arama kutusu (bütün BIST). Seçilen hisse için fiyat
   grafiği ve indikatörler (mevcut `core/market_data.py`), her indikatörün
   sade durumu (ör. "RSI 71 — aşırı alım"), haberler ve duygu skoru,
   modelin o günkü olasılığı ve sırası, uygunluk durumu ve dışlandıysa
   nedeni. Ollama yorumu isteğe bağlı kalır.
2. **Günün önerileri:** Telegram mesajının aynısı, tıklanınca 1. ekrana gider.
3. **Karne ve model:** Karne grafikleri, model geçmişi (terfi/ret kararları),
   veri kalite raporu.

Açılışta `data` dalından güncel dosyaları indirir; indirilemezse son
yerel kopyayla çalışır ve bunu belirtir.

### 6.9 Depo temizliği
`.gitignore` eklenir; `build/`, `backups/`, `__pycache__/`,
`work/backfill/*/http/`, `*.exe`, `data_cache/`, `training_reports/`
takipten çıkarılır (dosyalar yerelde kalır, geçmiş yeniden yazılmaz).
Bağımlılıklar `requirements.txt`'e taşınır.

## 7. Uygulama sırası

Her parça kendi planıyla yapılır ve tek başına çalışır halde biter:

1. Depo temizliği, `data` dalı, bütün BIST verisi, veri kontrolü, filtre.
2. Tahmin defteri, karne, günlük eğitim, şampiyon/aday sınavı.
3. Yeni Telegram mesajı; eski 20 hisselik tarama kaldırılır.
4. PC uygulamasının üç ekranı ve `.exe`'nin yeniden paketlenmesi.

## 8. Test

- Birim testleri: filtre kuralları (point-in-time), veri kontrolü, defterin
  vade doldurması, sınav kararı (aday iyi / kötü / eşit), mesaj biçimi.
- Sızıntı testi: herhangi bir (hisse, gün) satırının özelliği, o günden
  sonraki veri değiştirildiğinde değişmemeli.
- Her parça sonunda `--dry-run` ile uçtan uca çalıştırma; Telegram'a
  gönderim yalnızca kullanıcının onayıyla.

## 9. Uygulayıcıya bırakılanlar

- Modül ve dosya adları, parquet şemalarının ayrıntısı.
- Yahoo indirmelerinin toplu/parçalı yapılması ve yeniden deneme düzeni.
- Günlük işin 45 dakikaya sığması için gereken önbellekleme.
- 6.3'teki 3. ve 4. kural için veri kaynağının seçimi (doğrulanamazsa
  kullanıcıya dönülür).
