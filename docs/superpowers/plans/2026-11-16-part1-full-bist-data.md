# Parça 1 — Bütün BIST verisi, veri kontrolü ve uygunluk filtresi: Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Her işlem günü bütün BIST hisselerinin fiyatını güncelleyen, veriyi denetleyen, manipülasyona açık hisseleri işaretleyen ve sonucu deponun `data` dalına yazan bir günlük iş.

**Architecture:** Yeni, küçük modüllerden oluşan `jarvis/` paketi; mevcut `core/` ve kök betiklere dokunulmaz (onlar Parça 2-4'te taşınır). Tek giriş noktası `python -m jarvis.pipeline`; bir veri klasörünü okur/yazar. GitHub Actions bu klasörü `data` dalının çalışma kopyası olarak bağlar ve commit eder.

**Tech Stack:** Python 3.11, pandas<3, numpy<2, pyarrow, yfinance, isyatirimhisse>=5.0.0, exchange-calendars==4.13.2, requests, beautifulsoup4, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-self-improving-full-bist-design.md` (bölüm 6.1, 6.2, 6.3, 6.9)

## Global Constraints

- Python 3.11 (Actions ile aynı). Yerelde 3.14 var; `.venv` 3.11 ile kurulur.
- Fiyat temeli: `yfinance`, `auto_adjust=True`; Yahoo sembolü `<TICKER>.IS`.
- Saat dilimi `Europe/Istanbul`; işlem takvimi `exchange_calendars` `XIST`.
- İlk kurulumda 5 yıl geçmiş; sonra yalnızca eksik seanslar.
- Kapanış çapraz kontrolü toleransı: %1.
- Karantina oranı > %20 ise `training_allowed = false`.
- Uygunluk: ≥ 252 seans; son 20 seans medyan TL hacmi ≥ 50.000.000;
  son 20 seansta tavan/taban günü ≤ 5; dışlama listesinde değil.
- Tavan/taban günü: |günlük kapanış getirisi| ≥ 0,095.
- Şüpheli sıçrama: |günlük kapanış getirisi| > 0,25 (düzeltilmiş seride).
- Bütün eşikler `jarvis/config.py` içinde sabit olarak durur.
- Hiçbir test ağa çıkmaz; ağ çağrıları parametreyle verilen fonksiyonlardır.
- Gizli anahtar depoya yazılmaz. Git geçmişi yeniden yazılmaz.
- Mevcut `daily_scan.yml` ve Telegram akışı bu parçada değişmez.

## Review Focus

1. Yahoo bir hisse için boş/eksik veri döndürür → o hisse atlanır, rapora
   `fetch_failed` yazılır, iş çökmez (Task 4).
2. İş Yatırım sayfası biçim değiştirir, 300'den az sembol çıkar → iş hata
   verip durur, eski `universe.parquet` korunur (Task 3).
3. Yeni halka arz: 252 seanstan kısa geçmiş → uygun değil, neden
   `short_history`; iş çökmez (Task 6).
4. Hacmi sıfır olan ya da işleme kapalı günler → medyan hesabı NaN üretmez,
   hisse `low_liquidity` olur (Task 6).
5. Aynı gün iş iki kez çalışır → satırlar çoğalmaz, `data` dalına boş
   commit atılmaz (Task 2, Task 8).

## Dosya yapısı

```
jarvis/__init__.py
jarvis/config.py        eşikler ve yollar
jarvis/store.py         veri klasöründe parquet/json oku-yaz
jarvis/universe.py      İş Yatırım sayfasından sembol listesi
jarvis/prices.py        Yahoo'dan artımlı fiyat güncelleme
jarvis/quality.py       veri kontrolleri ve karantina
jarvis/exclusions.py    tedbirli / Yakın İzleme dışlama listesi
jarvis/eligibility.py   uygunluk kuralları (point-in-time)
jarvis/pipeline.py      CLI: hepsini sırayla çalıştırır
config/exclusions.csv   elle tutulan dışlama listesi
tests/jarvis/test_*.py
requirements.txt, .gitignore
.github/workflows/daily_data.yml
```

---

### Task 1: Depo temizliği ve geliştirme ortamı

**Files:**
- Create: `.gitignore`, `requirements.txt`, `jarvis/__init__.py`, `jarvis/config.py`, `tests/jarvis/__init__.py`, `tests/jarvis/test_config.py`
- Delete from index only: `build/`, `backups/`, `__pycache__/`, `core/__pycache__/`, `work/backfill/THYAO/http/`, `BIST_AI_Radar.exe`, `data_cache/`, `training_reports/`, `5.0.0`, `debug_llm_payload.json`

**Interfaces:**
- Produces: `jarvis.config` sabitleri — `TRT="Europe/Istanbul"`, `CALENDAR="XIST"`, `HISTORY_YEARS=5`, `CROSS_CHECK_TOLERANCE=0.01`, `MAX_QUARANTINE_RATIO=0.20`, `MIN_HISTORY_SESSIONS=252`, `LIQUIDITY_WINDOW=20`, `MIN_MEDIAN_TURNOVER_TRY=50_000_000`, `LIMIT_MOVE=0.095`, `MAX_LIMIT_DAYS=5`, `SUSPECT_JUMP=0.25`, `MIN_UNIVERSE_SIZE=300`, `OHLCV=["Open","High","Low","Close","Volume"]`.

- [ ] **Step 1:** Python 3.11 yoksa kur ve ortamı oluştur.
  Run: `winget install -e --id Python.Python.3.11` sonra `py -3.11 -m venv .venv`
  Expected: `.venv\Scripts\python.exe --version` → `Python 3.11.x`
- [ ] **Step 2:** `requirements.txt` yaz: `daily_scan.yml` içindeki pip listesinin aynısı (sürüm sınırlarıyla) + `pyarrow`, `pytest`. `torch` satırı yorumda kalır (CPU indeksinden ayrı kurulur).
- [ ] **Step 3:** `.venv\Scripts\python -m pip install -r requirements.txt` → hatasız biter.
- [ ] **Step 4:** `.gitignore` yaz: `.venv/`, `__pycache__/`, `*.pyc`, `build/`, `dist/`, `backups/`, `*.exe`, `data_cache/`, `training_reports/`, `work/`, `data/`, `debug_llm_payload.json`, `.env`.
- [ ] **Step 5:** `git rm -r --cached` ile yukarıdaki "Delete from index only" yollarını takipten çıkar. `work/backfill/THYAO/` altındaki `THYAO.csv`, `corpus.jsonl`, `coverage.csv`, `sentiment.jsonl`, `training_report.json` **takipte kalır** (`live_inference.py` bunları kullanıyor); `.gitignore`'a `!work/backfill/THYAO/*.csv` vb. istisnaları ekle.
  Verify: `git ls-files | wc -l` < 150 ve `git status` yerel dosyaların silinmediğini gösterir.
- [ ] **Step 6:** `tests/jarvis/test_config.py` — `test_thresholds_match_spec`: yukarıdaki her sabitin değerini `assert` eder. Önce çalıştır (FAIL: modül yok), sonra `jarvis/config.py`'yi yaz, tekrar çalıştır (PASS).
  Run: `.venv\Scripts\python -m pytest tests/jarvis -v`
- [ ] **Step 7:** Mevcut testler bozulmadı: `.venv\Scripts\python -m unittest test_live_sessions test_daily_radar -v` → OK.
- [ ] **Step 8:** Commit: `chore: untrack build artifacts, add requirements and jarvis config`

### Task 2: Veri deposu (`jarvis/store.py`)

**Files:** Create `jarvis/store.py`, `tests/jarvis/test_store.py`

**Interfaces:**
- Produces:
  - `class Store(root: Path)`
  - `Store.read_prices(ticker: str) -> pd.DataFrame` — `DatetimeIndex` (tz'siz, normalize), sütunlar `OHLCV`; dosya yoksa boş çerçeve.
  - `Store.write_prices(ticker: str, frame: pd.DataFrame) -> bool` — var olanla birleştirir, aynı tarihte yeni satır kazanır, sıralar; içerik değişmediyse dosyaya dokunmaz ve `False` döner.
  - `Store.read_table(name: str) -> pd.DataFrame`, `Store.write_table(name: str, frame) -> bool` (`name` örn. `"universe/universe"` → `universe/universe.parquet`).
  - `Store.write_json(name: str, payload: dict) -> None` (`allow_nan=False`, `ensure_ascii=False`, atomik: geçici dosya + `os.replace`).

- [ ] **Step 1:** Testler (`tmp_path` ile):
  - `test_read_missing_returns_empty` — sütunlar `OHLCV`, uzunluk 0.
  - `test_write_then_read_roundtrip` — 3 satır yazılır, aynısı okunur.
  - `test_merge_overwrites_same_date_and_sorts` — ikinci yazımda çakışan tarihin `Close` değeri yenisidir, indeks artan sıradadır, tekrar yok.
  - `test_identical_write_returns_false_and_keeps_mtime` (Review Focus 5).
  - `test_write_json_rejects_nan` — `ValueError`.
- [ ] **Step 2:** Çalıştır → FAIL. **Step 3:** `store.py`'yi yaz. **Step 4:** Çalıştır → PASS.
- [ ] **Step 5:** Commit: `feat(jarvis): parquet store with idempotent writes`

### Task 3: Evren (`jarvis/universe.py`)

**Files:** Create `jarvis/universe.py`, `tests/jarvis/test_universe.py`, `tests/jarvis/fixtures/isyatirim_sample.html`

**Interfaces:**
- Produces:
  - `ISYATIRIM_URL = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx"`
  - `parse_symbols(html: str) -> list[str]` — `sirket-karti.aspx?hisse=<KOD>` bağlantılarından benzersiz, sıralı kodlar (regex `hisse=([A-Z0-9]{3,6})`).
  - `fetch_symbols(get: Callable[[str], str] = http_get) -> list[str]` — `len < MIN_UNIVERSE_SIZE` ise `UniverseError` fırlatır.
  - `class UniverseError(RuntimeError)`

Not: bu sayfa 2026-11-16'da 629 kod döndürdü.

- [ ] **Step 1:** Fixture: sayfadan kesilmiş, 5 hisse bağlantısı (biri iki kez) içeren küçük HTML. Testler:
  - `test_parse_symbols_unique_sorted` → `["A1CAP","AAGYO","ACSEL","ADEL","THYAO"]`.
  - `test_fetch_symbols_raises_when_too_few` — 5 sembollü HTML ile `UniverseError` (Review Focus 2).
  - `test_fetch_symbols_ok` — 300 sahte bağlantı üreten `get` ile 300 kod.
- [ ] **Step 2-4:** FAIL → yaz → PASS. `http_get`: `requests`, tarayıcı `User-Agent`, 40 sn zaman aşımı, 3 deneme.
- [ ] **Step 5:** Commit: `feat(jarvis): full BIST symbol list from Is Yatirim`

### Task 4: Artımlı fiyat güncelleme (`jarvis/prices.py`)

**Files:** Create `jarvis/prices.py`, `tests/jarvis/test_prices.py`

**Interfaces:**
- Consumes: `Store.read_prices/write_prices`, `config.HISTORY_YEARS`.
- Produces:
  - `Downloader = Callable[[list[str], pd.Timestamp, pd.Timestamp], dict[str, pd.DataFrame]]` — Yahoo sembolleri, başlangıç, bitiş (dahil) → sembol başına OHLCV.
  - `yahoo_download: Downloader` — `yf.download(..., auto_adjust=True, group_by="ticker", threads=True)`, 50'lik partiler, parti başına 3 deneme.
  - `update_prices(store: Store, tickers: list[str], session: pd.Timestamp, download: Downloader = yahoo_download) -> dict[str, str]` — her ticker için durum: `"updated" | "current" | "fetch_failed"`. Depoda verisi olmayan hisse için başlangıç `session - 5 yıl`; olan için `son_tarih + 1 gün`. Son tarihi `session` olan hisse indirilmez.
  - `last_completed_session(as_of: pd.Timestamp | None = None) -> pd.Timestamp` — mevcut `live_inference.last_completed_bist_session`'ı çağıran ince sarmalayıcı (tz'siz, normalize).

- [ ] **Step 1:** Testler (sahte `download` ile):
  - `test_new_ticker_requests_five_years`.
  - `test_existing_ticker_requests_only_missing_days`.
  - `test_current_ticker_is_not_downloaded` — `download` o sembolle çağrılmaz.
  - `test_empty_download_marks_fetch_failed_and_continues` — iki hisseden biri boş döner; diğeri yazılır (Review Focus 1).
  - `test_rows_with_nan_close_are_dropped`.
- [ ] **Step 2-4:** FAIL → yaz → PASS.
- [ ] **Step 5:** Commit: `feat(jarvis): incremental Yahoo price updates`

### Task 5: Veri kontrolü (`jarvis/quality.py`)

**Files:** Create `jarvis/quality.py`, `tests/jarvis/test_quality.py`

**Interfaces:**
- Produces:
  - `check_prices(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> list[str]` — bulunan sorun kodları: `"nonpositive_price"`, `"zero_volume_last"` (son seans hacmi 0), `"missing_sessions"` (son 20 XIST seansından eksik olan var), `"suspect_jump"` (son 20 seansta |getiri| > 0,25), `"stale"` (son satır `sessions[-1]` değil).
  - `cross_check(yahoo_close: float, reference_close: float | None) -> str` — `"ok" | "mismatch" | "unavailable"`; `mismatch` = göreli fark > 0,01.
  - `ReferenceCloses = Callable[[list[str], pd.Timestamp], dict[str, float]]`
  - `isyatirim_closes: ReferenceCloses` — `isyatirimhisse.fetch_stock_data(symbols=..., start_date=..., end_date=...)`; hata olursa boş sözlük döner ve uyarı loglar.
  - `build_report(store, tickers, session, statuses: dict[str,str], reference: ReferenceCloses = isyatirim_closes) -> dict` — şema:
    `{"session": "YYYY-MM-DD", "tickers": N, "quarantined": {ticker: [kodlar]}, "cross_check_unavailable": [..], "fetch_failed": [..], "quarantine_ratio": float, "training_allowed": bool}`.
    Karantina: `check_prices` herhangi bir kod döndürdüyse, `cross_check == "mismatch"` ise ya da `fetch_failed` ise. `unavailable` karantina sebebi değildir. *(Kararım: referans kaynağın kesintisi bütün günü iptal etmemeli; sayı raporda görünür.)*

- [ ] **Step 1:** Testler:
  - her sorun kodu için birer test (5 adet) + `test_clean_frame_has_no_issues`.
  - `test_cross_check_boundaries` — fark %0,9 → `ok`; %1,1 → `mismatch`; `None` → `unavailable`.
  - `test_report_blocks_training_above_twenty_percent` — 10 hisseden 3'ü karantinada → `training_allowed is False`; 2'si → `True`.
  - `test_reference_outage_does_not_quarantine`.
- [ ] **Step 2-4:** FAIL → yaz → PASS.
- [ ] **Step 5:** Commit: `feat(jarvis): price quality checks and quarantine report`

### Task 6: Dışlama listesi ve uygunluk (`jarvis/exclusions.py`, `jarvis/eligibility.py`)

**Files:** Create `jarvis/exclusions.py`, `jarvis/eligibility.py`, `config/exclusions.csv`, `tests/jarvis/test_exclusions.py`, `tests/jarvis/test_eligibility.py`

**Interfaces:**
- Produces:
  - `config/exclusions.csv` başlığı: `ticker,reason,start,end,source` (`reason` ∈ `vbts`, `watchlist_market`, `manual`; `end` boş olabilir).
  - `load_exclusions(path: Path) -> pd.DataFrame`
  - `excluded_on(exclusions: pd.DataFrame, day: pd.Timestamp) -> dict[str, str]` — o gün etkin olan ticker → reason.
  - `eligibility_history(frame: pd.DataFrame) -> pd.DataFrame` — fiyat çerçevesiyle aynı indeks; sütunlar `sessions` (o güne kadarki satır sayısı), `median_turnover` (`Close*Volume`'un 20 seanslık kayan medyanı, `min_periods=20`), `limit_days` (son 20 seansta |getiri| ≥ 0,095 gün sayısı), `eligible_pit` (bool), `reason_pit` (`""`, `short_history`, `low_liquidity`, `limit_moves`; ilk tutmayan kural, bu sırayla). Her satır yalnızca o satıra kadarki veriyle hesaplanır.
  - `eligibility_today(store, tickers, session, exclusions, quarantined: set[str]) -> pd.DataFrame` — sütunlar `ticker, session, eligible, reason, median_turnover, limit_days, sessions`. Öncelik: `quarantined` → `excluded:<reason>` → `reason_pit`.

- [ ] **Step 1:** Testler:
  - `test_short_history_ineligible` — 100 seanslık hisse → `short_history` (Review Focus 3).
  - `test_liquidity_threshold_boundary` — medyan 49,9M → `low_liquidity`; 50,0M → uygun.
  - `test_zero_volume_days_do_not_produce_nan` — 20 günün 12'si hacim 0 → `low_liquidity`, `median_turnover == 0` (Review Focus 4).
  - `test_limit_days_boundary` — 5 tavan günü → uygun; 6 → `limit_moves`.
  - `test_point_in_time_no_lookahead` — çerçevenin son 30 satırı değiştirilince ilk N-30 satırın `eligible_pit` değerleri aynı kalır.
  - `test_exclusion_active_window` — `start ≤ gün ≤ end` etkin; `end` boş → süresiz; pencere dışı → etkin değil.
  - `test_priority_quarantine_over_exclusion_over_rules`.
- [ ] **Step 2-4:** FAIL → yaz → PASS. `config/exclusions.csv` yalnızca başlık satırıyla oluşturulur.
- [ ] **Step 5:** Commit: `feat(jarvis): eligibility filter with point-in-time rules`

### Task 6b: Otomatik dışlama kaynakları (`jarvis/measures.py`)

**Files:** Create `jarvis/measures.py`, `tests/jarvis/test_measures.py`, `tests/jarvis/fixtures/kap_pazarlar_sample.html`, `tests/jarvis/fixtures/gcm_tedbir_sample.html`

Kaynaklar (2026-11-16'da doğrulandı):
- **Pazar bilgisi — resmi:** `https://www.kap.org.tr/tr/Pazarlar`. Sayfanın içine gömülü JSON'da her pazar için `"marketName":"…","marketDetailContentList":[{"stockCode":"…"}…]` var (tırnaklar `\"` olarak kaçışlı). O gün: Yıldız 149, Ana 393, Alt 48, Yakın İzleme 22, Piyasa Öncesi İşlem Platformu 19, Gözaltı 8.
- **Tedbirler — aracı kurum sayfası:** `https://www.gcmyatirim.com.tr/borsa/tedbir-kapsamindaki-pay-senetleri`. `table.gcmTable`; sütunlar: hisse kodu, başlangıç, bitiş (`GG.AA.YYYY`), ardından 7 tedbir sütunu (`✓` işaretli). O gün yalnızca 1 satır vardı; **listenin tam olduğu doğrulanmadı.**

**Interfaces:**
- Produces:
  - `ALLOWED_MARKETS = ("YILDIZ PAZAR", "ANA PAZAR", "ALT PAZAR")`
  - `parse_kap_markets(html: str) -> dict[str, str]` — ticker → pazar adı. Bir `stockCode` virgülle birden çok kod içeriyorsa her biri ayrı girdidir.
  - `parse_gcm_measures(html: str) -> pd.DataFrame` — sütunlar `ticker, start, end, measures` (`measures`: işaretli sütun başlıklarının `;` ile birleşimi).
  - `fetch_exclusions(tickers: list[str], session: pd.Timestamp, get: Callable[[str], str] = http_get) -> tuple[pd.DataFrame, dict[str, str]]` — birinci değer `exclusions.csv` şemasında satırlar: pazarı `ALLOWED_MARKETS` dışında olanlar `reason="watchlist_market"`, KAP listesinde hiç bulunmayanlar `reason="unknown_market"` (ikisi de `start=session`, `end=session`), tedbir satırları `reason="vbts"` (sayfadaki tarihlerle). İkinci değer kaynak durumu: `{"kap_markets": "ok"|"failed", "measures": "ok"|"failed"}`. Bir kaynak hata verirse ya da KAP'tan 300'den az kod çıkarsa o kaynak `failed` olur ve satır üretmez; istisna fırlatılmaz.
- Pipeline (Task 7) bu satırları `config/exclusions.csv` ile birleştirir, `universe/measures.parquet`'e günlük anlık görüntü olarak ekler ve kaynak durumunu kalite raporuna `"exclusion_sources"` anahtarıyla yazar. Bir kaynak `failed` ise depodaki **en son başarılı** anlık görüntü kullanılır (yoksa o kaynak o gün kimseyi dışlamaz) ve durum raporda görünür.

- [ ] **Step 1:** Fixture'lar: gerçek sayfalardan kesilmiş küçük parçalar (KAP: 3 pazar, 6 kod; GCM: başlık + 2 satır). Testler:
  - `test_parse_kap_markets` — `{"ATEKS": "YAKIN İZLEME PAZARI", "THYAO": "YILDIZ PAZAR", …}`.
  - `test_parse_gcm_measures` — `KARTN`, `2026-11-10`, `2026-12-09`, `measures == "AÇIĞA SATIŞ YASAĞI"`.
  - `test_fetch_marks_watchlist_and_unknown_market`.
  - `test_fetch_source_failure_is_reported_not_raised` — `get` istisna fırlatır → durum `failed`, boş çerçeve.
  - `test_kap_too_few_codes_is_failed`.
- [ ] **Step 2-4:** FAIL → yaz → PASS.
- [ ] **Step 5:** Commit: `feat(jarvis): automatic market and measure exclusions`

### Task 7: Günlük iş (`jarvis/pipeline.py`)

**Files:** Create `jarvis/pipeline.py`, `tests/jarvis/test_pipeline.py`

**Interfaces:**
- Consumes: Task 2-6'nın hepsi.
- Produces:
  - `run(data_dir: Path, as_of: pd.Timestamp | None = None, *, symbols=fetch_symbols, download=yahoo_download, reference=isyatirim_closes, exclusions_path=Path("config/exclusions.csv")) -> dict` — sırayla: sembol listesi → `update_prices` → `build_report` → `fetch_exclusions` (+ elle liste) → `eligibility_today` → yaz. `run` ek olarak `exclusions=fetch_exclusions` parametresi alır. Yazdıkları: `universe/measures.parquet`, `prices/<T>.parquet`, `universe/universe.parquet` (o günün satırları eklenir; aynı `(ticker, session)` yeniden yazılırsa üzerine yazar), `quality/<YYYY-MM-DD>.json`. Döndürdüğü özet: `{"session", "tickers", "eligible", "quarantined", "excluded", "training_allowed"}`.
  - CLI: `python -m jarvis.pipeline --data-dir data [--as-of "2026-11-13 18:30:00+03:00"] [--limit N]`; özeti JSON olarak yazdırır. `--limit` yalnızca ilk N sembolü işler (yerel deneme için). `fetch_symbols` hata verirse çıkış kodu 1 ve hiçbir dosya değişmez.

- [ ] **Step 1:** Testler (hepsi sahte bağımlılıklarla, `tmp_path`):
  - `test_run_writes_expected_files`.
  - `test_run_twice_same_session_is_idempotent` — ikinci çalıştırmada `universe` satır sayısı aynı, fiyat dosyalarının `mtime`'ı değişmez (Review Focus 5).
  - `test_universe_error_leaves_data_untouched` (Review Focus 2).
  - `test_summary_counts` — 4 hisse: 1 uygun, 1 karantina, 1 dışlanmış, 1 düşük likidite.
- [ ] **Step 2-4:** FAIL → yaz → PASS.
- [ ] **Step 5:** Gerçek veriyle küçük deneme: `.venv\Scripts\python -m jarvis.pipeline --data-dir data --limit 15`
  Expected: özet JSON; `data/prices/` altında ~15 parquet; `data/quality/<tarih>.json` var. Çıktıyı kullanıcıya göster.
- [ ] **Step 6:** Tam deneme: `--limit` olmadan çalıştır, süreyi ve uygun hisse sayısını not et. 50M TL eşiğiyle kaç hisse kaldığını kullanıcıya bildir (tasarımda açık bırakılan soru).
- [ ] **Step 7:** Commit: `feat(jarvis): daily data pipeline CLI`

### Task 8: GitHub Actions iş akışı ve `data` dalı

**Files:** Create `.github/workflows/daily_data.yml`

**Interfaces:**
- Consumes: `python -m jarvis.pipeline --data-dir data`.
- Produces: `data` dalında güncel `prices/`, `universe/`, `quality/`.

- [ ] **Step 1:** İş akışını yaz:
  - `on: workflow_dispatch` + `schedule: cron "15 15 * * 1-5"` (18:15 TRT; mevcut 18:30 taramasından önce).
  - `permissions: contents: write`; `concurrency: group: bist-daily-data`.
  - `timeout-minutes: 45`; Python 3.11; `pip install -r requirements.txt` (pip önbelleğiyle).
  - `main`'i checkout et; `data` dalını `data/` klasörüne ikinci checkout olarak al. Dal yoksa adım onu boş, geçmişsiz (orphan) olarak oluşturur.
  - `python -m pytest tests/jarvis -q` → sonra `python -m jarvis.pipeline --data-dir data`.
  - `data/` içinde `git add -A`; değişiklik varsa `data: <YYYY-MM-DD>` mesajıyla commit ve push; yoksa adım atlanır (Review Focus 5). Committer: `github-actions[bot]`.
  - Özet JSON'u `$GITHUB_STEP_SUMMARY`'ye yaz.
- [ ] **Step 2:** `main`'e birleştirildikten sonra iş akışını elle tetikle: `gh workflow run daily_data.yml` → `gh run watch`.
  Expected: iş yeşil, 45 dakikadan kısa; `data` dalında dosyalar var. İlk çalıştırma (5 yıl × ~630 hisse) süreyi aşarsa `--limit` ile iki elle çalıştırmada doldurulur; sonuç kullanıcıya bildirilir.
- [ ] **Step 3:** Aynı gün ikinci kez tetikle → yeni commit atılmaz.
- [ ] **Step 4:** Commit: `ci: daily data workflow writing to data branch`

---

## Bu parçanın bilinen açığı

Pazar bilgisi (Yakın İzleme, Piyasa Öncesi, Gözaltı) KAP'tan resmi olarak
alınır. Tedbir (VBTS) listesi ise bir aracı kurum sayfasından okunur ve
tam olduğu doğrulanmamıştır; sayfa biçim değiştirirse kaynak `failed` olur.
Bu yüzden tedbir kuralı tek başına güvenilen koruma değildir: likidite ve
tavan/taban kuralları asıl korumadır, `config/exclusions.csv` elle ekleme
için durur. Geçmiş tarihler için tedbir kaydı yoktur; anlık görüntüler
bugünden itibaren birikir.
