# Parça 2 — Günlük eğitim, tahmin defteri ve karne: Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Her işlem günü uygun hisselerle modeli yeniden eğiten, güvenlik kontrollerinden geçerse devreye alan, bütün tahminleri deftere yazıp vadesi dolunca notlayan ve karneyi `data` dalına yazan iş.

**Architecture:** Parça 1'in `jarvis/` paketine eklenir. Özellikler `core/ai_analyzer.py`'den **taşınır** (o modül streamlit/shap içe aktardığı için bulutta kullanılamaz). Tek giriş noktası `python -m jarvis.train`; `jarvis.pipeline`'dan sonra aynı veri klasöründe çalışır. Her şey saf fonksiyon + `Store`; ağ yalnızca makro verisi indirmede.

**Tech Stack:** Parça 1 + `xgboost`.

**Spec:** `docs/superpowers/specs/2026-11-16-self-improving-full-bist-design.md` (bölüm 6.4, 6.5, 6.6)

## Global Constraints

- Hedef: hissenin 5 seanslık getirisi − XU100'ün 5 seanslık getirisi > 0 (`HORIZON = 5`).
- Eğitim penceresi: son 3 yıl (`TRAIN_YEARS = 3`); yalnızca `eligible_pit` satırlar.
- Etiketi kesinleşmemiş son 5 seans + 7 seans tampon eğitime girmez (`EMBARGO_SESSIONS = 7`).
- Öneri: günlük ilk 5 (`TOP_N = 5`), olasılık ≥ VIX'e bağlı üst eşik (mevcut formül: `0.52 + 0.13 * clip((vix_pct − 25) / 50, 0, 1)`).
- İşlem maliyeti: 20 baz puan (`COST_BPS = 20`), gidiş-dönüş bir kez düşülür.
- Tarif sınavı: 60 seans (`EXAM_SESSIONS`), her 5 seansta yeniden eğitim (`REFIT_EVERY`), bootstrap 2000 örnek, güven 0,80, sabit tohum 42.
- Sağlık penceresi 120 seans (`HEALTH_SESSIONS`); kayma eşiği 0,5 (`DRIFT_MIN_RANK_CORR`).
- En az eğitim satırı: 20.000 (`MIN_TRAIN_ROWS`).
- Simülasyon karnesi: son 250 seans (`SIM_SESSIONS`), canlıdan ayrı dosyada.
- Geleceğe bakma yok: bir (hisse, gün) satırının özelliği o günden sonraki veriye bağlı olamaz; makro değerler en erken ertesi gün kullanılabilir.
- XGBoost: `objective="binary:logistic"`, `tree_method="hist"`, `random_state=42`, `n_jobs=2`.
- Hiçbir test ağa çıkmaz. Testler küçük sentetik veriyle çalışır.

## Review Focus

1. Uygun hisse sayısı o gün `TOP_N`'den az → eldeki kadar sıralanır, iş çökmez (Task 5, 8).
2. Vadesi dolacak hisse bu arada listeden çıkmış/karantinada, fiyatı yok → o defter satırı `matured=False` kalır, karne onu saymaz (Task 6).
3. Henüz şampiyon ağırlık yok (ilk çalıştırma) → kayma kontrolü atlanır, yeni ağırlıklar devreye girer ve bu `history.jsonl`'de yazar (Task 8).
4. XU100 ya da VIX verisi o gün indirilemedi → eğitim ve tahmin yapılmaz, dünkü durum korunur, çıkış kodu 0 ve özet `status: "skipped"` (Task 8).
5. Aynı gün iki kez çalıştırma → defterde aynı (tarih, hisse) iki kez yer almaz (Task 6, 8).

## Dosya yapısı

```
jarvis/config.py        (+ yeni sabitler)
jarvis/prices.py        (update_prices'a symbol_of parametresi)
jarvis/macro.py         makro seriler: indirme ve okuma
jarvis/features.py      yerel + makro özellikler (core/ai_analyzer'dan taşınır)
jarvis/dataset.py       (hisse, gün) satırları: özellik, etiket, uygunluk
jarvis/model.py         Recipe, eğit, tahmin et, kaydet/yükle
jarvis/metrics.py       ilk-N fazla getiri, rank IC, eşik
jarvis/walkforward.py   bir tarifi geçmişte ileriye yürütme
jarvis/ledger.py        tahmin defteri: yaz, vade doldur
jarvis/scorecard.py     karne özeti
jarvis/gate.py          tarif sınavı (bootstrap) + sağlık ve kayma kontrolleri
jarvis/train.py         CLI: günlük / --simulate / --search
tests/jarvis/test_*.py
.github/workflows/daily_data.yml   (+ eğitim adımı, + cumartesi araması)
```

---

### Task 1: Makro veriler (`jarvis/macro.py`)

**Files:** Modify `jarvis/prices.py`, `jarvis/config.py`; Create `jarvis/macro.py`, `tests/jarvis/test_macro.py`

**Interfaces:**
- Produces:
  - `update_prices(..., symbol_of: Callable[[str], str] = yahoo_symbol)` — mevcut davranış değişmez.
  - `MACRO_SYMBOLS = {"USDTRY": "USDTRY=X", "BRENT": "BZ=F", "GOLD": "GC=F", "SP500": "^GSPC", "VIX": "^VIX", "BIST100": "XU100.IS", "BISTBANK": "XBANK.IS"}` (sıra korunur).
  - `macro_store(data_dir: Path) -> Store` — `Store(data_dir / "macro")`.
  - `update_macro(data_dir, session, download=yahoo_download) -> dict[str, str]` — durumlar `update_prices` ile aynı.
  - `load_macro(data_dir) -> dict[str, pd.DataFrame]` — ad → OHLCV; eksik seri için boş çerçeve.
  - `REQUIRED_MACRO = ("BIST100", "VIX")` — bunlar `session` gününe kadar güncel değilse eğitim yapılamaz. Diğer makrolar eksikse özellikleri NaN kalır (XGBoost işler).

Not: makro seriler hacim sütununu 0/NaN getirebilir; `_clean` yalnızca `Close` NaN satırlarını atar, `Volume` NaN ise 0 yazılır.

- [ ] Testler: `test_symbol_of_maps_names_to_yahoo_symbols`, `test_update_macro_writes_under_macro_folder`, `test_load_macro_returns_empty_for_missing`, `test_nan_volume_becomes_zero`. FAIL → yaz → PASS.
- [ ] Commit: `feat(jarvis): macro series storage and updates`

### Task 2: Özellikler (`jarvis/features.py`)

**Files:** Create `jarvis/features.py`, `tests/jarvis/test_features.py`

**Interfaces:**
- Produces:
  - `LOCAL_FEATURES` (10 ad, `core/ai_analyzer.LOCAL_FEATURE_COLUMNS` ile aynı sıra), `MACRO_FEATURES` (7 makro × `_1d_lag`, `_5d_mean_lag` = 14 ad), `FEATURES = LOCAL_FEATURES + MACRO_FEATURES`, `VIX_PERCENTILE = "VIX_Percentile"`.
  - `local_features(equity: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame` — `core.ai_analyzer.build_stationary_features` gövdesinin birebir taşınmış hali.
  - `macro_features(histories: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.DataFrame` — `build_macro_features` + `_available_on_calendar` taşınmış hali (bir günlük gecikme, 7 günden eski değer NaN, VIX yüzdeliği 252/126). Boş seri → o makronun sütunları NaN.
  - `FEATURE_VERSION = "jarvis-features-v1"`.

- [ ] Testler:
  - `test_feature_names_and_order` — 24 ad, ilk 10 yerel.
  - `test_parity_with_core_ai_analyzer` — `pytest.importorskip("streamlit")`; rastgele (tohum 0) 400 günlük OHLCV'de `local_features` çıktısı `core.ai_analyzer.build_stationary_features` ile `assert_frame_equal` (rtol 1e-9). Makro için aynı karşılaştırma.
  - `test_local_features_are_causal` — son 30 satır değiştirilince ilk N−30 satır aynı.
  - `test_macro_value_is_lagged_one_day` — gün D'nin makro getirisi takvimde en erken D+1'de görünür.
  - `test_stale_macro_becomes_nan` — 7 günden eski kaynak → NaN.
  - `test_zero_volume_and_flat_prices_produce_no_inf`.
- [ ] FAIL → yaz → PASS. Commit: `feat(jarvis): port feature engineering without UI dependencies`

### Task 3: Veri kümesi (`jarvis/dataset.py`)

**Files:** Create `jarvis/dataset.py`, `tests/jarvis/test_dataset.py`

**Interfaces:**
- Consumes: `Store.read_prices`, `load_macro`, `local_features`, `macro_features`, `eligibility_history`.
- Produces:
  - `build_rows(store, macro, tickers, session) -> pd.DataFrame` — her (hisse, gün ≤ session) için bir satır. Sütunlar: `ticker`, `date`, `FEATURES`…, `VIX_Percentile`, `eligible_pit`, `fwd_excess` (hisse 5 seans getirisi − XU100 5 seans getirisi; vadesi dolmamışsa NaN), `target` (`fwd_excess > 0` → 1.0/0.0; NaN ise NaN), `label_end` (etiketin kesinleştiği seans; yoksa NaT).
    - Takvim: XU100 (`BIST100`) çerçevesinin indeksi. Hisse fiyatı takvime **yalnızca hizalanır, doldurulmaz**; hissenin işlem görmediği gün satır üretmez.
    - 5 seans ileri = takvimde 5 adım; o günde hissenin kapanışı yoksa `fwd_excess` NaN.
    - Yerel özelliklerinden herhangi biri NaN olan satır atılır (ısınma dönemi).
  - `training_slice(rows, session, calendar) -> pd.DataFrame` — `eligible_pit`, `target` dolu, `label_end ≤ calendar[konum(session) − EMBARGO_SESSIONS]`, `date ≥ session − 3 yıl`.
  - `prediction_slice(rows, session, eligible_today: set[str]) -> pd.DataFrame` — `date == session` ve `ticker ∈ eligible_today`.

- [ ] Testler (3 sentetik hisse + sentetik XU100, 400 seans):
  - `test_target_uses_five_sessions_ahead_excess` — elle hesaplanan bir satırla karşılaştır.
  - `test_last_five_sessions_have_no_label`.
  - `test_training_slice_respects_embargo` — en geç `label_end`, session'dan en az 7 seans önce.
  - `test_training_slice_excludes_ineligible_rows`.
  - `test_no_lookahead_in_features` — son 30 günün fiyatı değişince önceki satırların özellikleri aynı.
  - `test_ticker_missing_bar_at_horizon_has_nan_label`.
  - `test_prediction_slice_only_todays_eligible`.
- [ ] FAIL → yaz → PASS. Commit: `feat(jarvis): leak-free training rows`

### Task 4: Model ve tarif (`jarvis/model.py`)

**Files:** Create `jarvis/model.py`, `tests/jarvis/test_model.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Recipe: params: dict; features: tuple[str, ...]; train_years: int = 3` + `to_dict()`, `from_dict()`, `recipe_id` (içeriğin sha1'inin ilk 10 hanesi).
  - `DEFAULT_RECIPE` — `params = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 3.0, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0}`, `features = FEATURES`.
  - `fit(recipe: Recipe, train: pd.DataFrame) -> XGBClassifier`
  - `predict(model, recipe, rows) -> pd.Series` — `rows.index` ile hizalı olasılık.
  - `save_model(store_root: Path, model, recipe, meta: dict) -> None` — `models/champion.json` (ağırlıklar) + `models/champion.meta.json` (`recipe`, `recipe_id`, `trained_through`, `train_rows`, `feature_version`, `created_at`); kaydetmeden önce varsa mevcut ikiliyi `models/previous.*` olarak kopyalar.
  - `load_model(store_root) -> tuple[XGBClassifier, Recipe, dict] | None`

- [ ] Testler: `test_fit_predict_learns_separable_signal` (tek özelliğe bağlı sentetik hedefte AUC > 0,9), `test_predict_is_deterministic`, `test_save_load_roundtrip_gives_same_predictions`, `test_save_keeps_previous`, `test_load_missing_returns_none`, `test_recipe_id_changes_with_params`.
- [ ] FAIL → yaz → PASS. Commit: `feat(jarvis): recipe, training and model persistence`

### Task 5: Ölçütler ve ileriye yürütme (`jarvis/metrics.py`, `jarvis/walkforward.py`)

**Files:** Create both + `tests/jarvis/test_metrics.py`, `tests/jarvis/test_walkforward.py`

**Interfaces:**
- Produces (`metrics`):
  - `upper_threshold(vix_percentile: float) -> float` — NaN ise 0,52.
  - `daily_scores(predictions: pd.DataFrame, top_n=TOP_N, cost_bps=COST_BPS) -> pd.DataFrame` — girdi sütunları `date, ticker, prob, fwd_excess`; çıktı indeks `date`, sütunlar `top_excess` (o günün en yüksek `top_n` olasılıklı hissesinin ortalama `fwd_excess`'i − maliyet), `rank_ic` (o gün `prob` ile `fwd_excess` Spearman korelasyonu; 3'ten az hisse → NaN), `n`.
  - `summarize(scores) -> dict` — `mean_top_excess`, `mean_rank_ic`, `hit_rate` (`top_excess > 0` oranı), `days`.
- Produces (`walkforward`):
  - `walk_forward(rows, recipe, calendar, start, end, refit_every=REFIT_EVERY) -> pd.DataFrame` — `[start, end]` aralığındaki her seans için o gün `eligible_pit` olan satırların tahmini (`date, ticker, prob, fwd_excess, VIX_Percentile`). Her `refit_every` seansta `training_slice(rows, o_gün, calendar)` ile yeniden eğitir; arada son modeli kullanır. Eğitim satırı `MIN_TRAIN_ROWS`'tan azsa o blok atlanır.

- [ ] Testler:
  - `test_top_excess_subtracts_cost_once`, `test_top_n_larger_than_available_uses_all` (Review Focus 1), `test_rank_ic_perfect_and_inverse` (+1 / −1), `test_rank_ic_nan_below_three`.
  - `test_walk_forward_never_trains_on_future` — `fit`'i saran sahte ile her eğitimin en geç `label_end`'i tahmin gününden ≥ 7 seans önce.
  - `test_walk_forward_refits_on_schedule` — 20 seansta 4 eğitim.
  - `test_walk_forward_skips_when_too_few_rows`.
- [ ] FAIL → yaz → PASS. Commit: `feat(jarvis): metrics and walk-forward evaluation`

### Task 6: Tahmin defteri ve karne (`jarvis/ledger.py`, `jarvis/scorecard.py`)

**Files:** Create both + tests

**Interfaces:**
- Produces:
  - Defter tablosu `ledger/predictions` sütunları: `date, ticker, prob, rank, recommended, threshold, model_id, recipe_id, entry_close, matured, exit_date, exit_close, fwd_return, benchmark_return, fwd_excess`.
  - `record(store, session, predictions: pd.DataFrame, model_id, recipe_id, threshold) -> int` — `predictions`: `ticker, prob, entry_close`. `rank` olasılığa göre (1 en yüksek; eşitlikte ticker alfabetik), `recommended = rank ≤ TOP_N and prob ≥ threshold`. Aynı `date` için önceki satırlar değiştirilir (Review Focus 5). Dönen: yazılan satır sayısı.
  - `mature(store, price_store, benchmark: pd.DataFrame, calendar, session) -> int` — `matured=False` olup `calendar`'da 5 seans sonrası ≤ `session` olan satırları doldurur. Çıkış fiyatı yoksa satır `matured=False` kalır (Review Focus 2). **Düzeltme tabanı:** getiri, `entry_close` yerine o anki düzeltilmiş seriden giriş ve çıkış günlerinin kapanışlarıyla hesaplanır (bölünme sonrası tutarlı kalsın); `entry_close` yalnızca bilgi içindir.
  - `scorecard(ledger: pd.DataFrame, session, kind: str) -> dict` — `kind ∈ {"live", "simulated"}`; pencereler 20 / 60 / tümü; her biri için: `recommendations`, `hit_rate` (önerilenlerde `fwd_excess > 0`), `mean_excess`, `mean_rank_ic`, `days`; ayrıca kalibrasyon: olasılık dilimleri (0,5–0,55, 0,55–0,6, 0,6+) için gerçekleşen isabet. Veri yoksa değerler `null`, sayılar 0.

- [ ] Testler: `test_record_ranks_and_flags_recommendations`, `test_record_below_threshold_not_recommended`, `test_record_same_day_twice_replaces`, `test_mature_fills_after_five_sessions`, `test_mature_skips_when_exit_price_missing`, `test_mature_uses_current_adjusted_series`, `test_scorecard_windows_and_hit_rate`, `test_scorecard_empty_ledger`.
- [ ] FAIL → yaz → PASS. Commit: `feat(jarvis): prediction ledger and scorecard`

### Task 7: Sınav ve güvenlik kontrolleri (`jarvis/gate.py`)

**Files:** Create `jarvis/gate.py`, `tests/jarvis/test_gate.py`

**Interfaces:**
- Produces:
  - `compare_recipes(champion_scores, challenger_scores, samples=2000, confidence=0.80, seed=42) -> dict` — girdiler `daily_scores` çıktıları; ortak günlerde eşleştirilmiş fark. Çıktı: `{"promote": bool, "top_excess_diff", "rank_ic_diff", "p_top_excess", "p_rank_ic", "days"}`; `promote` = iki farkın ortalaması > 0 **ve** bootstrap'te farkın > 0 olma oranı her ikisi için ≥ `confidence`. Ortak gün < 30 → `promote=False`, `reason="too_few_days"`.
  - `health_check(scores: pd.DataFrame) -> dict` — `{"ok": mean_rank_ic > 0, "mean_rank_ic", "days"}`; `days < 60` → `ok=True`, `reason="insufficient_history"`.
  - `drift_check(new_probs: pd.Series, old_probs: pd.Series | None) -> dict` — Spearman; `old_probs is None` → `{"ok": True, "reason": "no_previous_model"}`; ortak hisse < 10 → `ok=True, reason="too_few_names"`; aksi halde `ok = corr ≥ 0.5`.

- [ ] Testler: `test_clearly_better_challenger_is_promoted`, `test_equal_recipes_are_not_promoted`, `test_better_on_one_metric_only_is_not_promoted`, `test_too_few_days_not_promoted`, `test_bootstrap_is_deterministic`, `test_health_negative_ic_fails`, `test_health_short_history_passes_with_reason`, `test_drift_uncorrelated_fails`, `test_drift_without_previous_model_passes` (Review Focus 3).
- [ ] FAIL → yaz → PASS. Commit: `feat(jarvis): recipe exam and daily safety checks`

### Task 8: Günlük eğitim işi (`jarvis/train.py`)

**Files:** Create `jarvis/train.py`, `tests/jarvis/test_train.py`

**Interfaces:**
- Consumes: Task 1-7'nin hepsi; `quality/<date>.json` (`training_allowed`), `universe/universe` (bugünün `eligible` hisseleri).
- Produces:
  - `run_daily(data_dir, as_of=None, *, download=yahoo_download) -> dict` — sıra:
    1. `session`; kalite raporu yoksa ya da başka güne aitse `status="skipped", reason="no_quality_report"`.
    2. `update_macro`; `REQUIRED_MACRO` güncel değilse `status="skipped", reason="macro_unavailable"` (Review Focus 4).
    3. `build_rows` → `mature` → karne (`scorecard/scorecard.json`, `kind="live"`).
    4. Tarif: `models/recipe.json` varsa o, yoksa `DEFAULT_RECIPE`.
    5. Kontroller: `training_allowed`; `len(training_slice) ≥ MIN_TRAIN_ROWS`; `health_check(walk_forward(son 120 seans))`.
    6. Geçerse `fit`; `drift_check(yeni tahmin, eski şampiyonun bugünkü tahmini)`; geçerse `save_model` (`model_id = "<session>-<recipe_id>"`). Geçmezse eski şampiyon kalır.
    7. Etkin şampiyonla bugünün tahminleri → `record`. Şampiyon hiç yoksa ya da sağlık kontrolü başarısızsa `recommended` hepsi False.
    8. `daily/<session>.json`: `session`, `recommendations` (önerilenler: `ticker, prob, rank`), `top` (ilk 10, önerilsin ya da önerilmesin), `threshold`, `model` (`model_id`, `retrained: bool`, `weights_changed: bool`, `checks: {...}`), `scorecard_live` (60 seanslık özet).
    9. `models/history.jsonl`'e bir satır ekle (aynı `session` için önceki satır değiştirilir).
    - Dönen özet: `{"status": "ok"|"skipped", "session", "retrained", "weights_changed", "recommendations": int, "matured_today": int, "reason"?}`.
  - `run_simulation(data_dir, as_of=None) -> dict` — şampiyon tarifle son 250 seans `walk_forward`; sonucu `ledger/simulated` tablosuna (defter şemasında, `model_id="simulation"`) ve `scorecard/simulated.json`'a yazar. Canlı deftere dokunmaz.
  - `run_search(data_dir, as_of=None, candidates=12, seed=None) -> dict` — `train_xgboost.sample_parameters` aralıklarından `candidates` rastgele tarif (tohum = seans tarihi) üretir; her biri ve şampiyon tarif son 60 seansta `walk_forward`; en iyi adayı (`mean_top_excess`) `compare_recipes` ile sınar; geçerse `models/recipe.json` yazılır. Sonuç `models/search/<session>.json`.
  - CLI: `python -m jarvis.train --data-dir data [--as-of ...] [--simulate | --search]`; özeti JSON yazdırır; `skipped` çıkış kodu 0.

- [ ] Testler (küçük sentetik evren; `MIN_TRAIN_ROWS` test içinde `monkeypatch` ile 200):
  - `test_first_run_trains_and_records` — şampiyon dosyaları, defter, `daily/*.json`, `history.jsonl` oluşur; `weights_changed=True` (Review Focus 3).
  - `test_second_run_same_day_is_idempotent` — defter satır sayısı aynı, `history.jsonl` tek satır (Review Focus 5).
  - `test_skips_without_quality_report`, `test_skips_when_benchmark_stale` (Review Focus 4).
  - `test_training_not_allowed_keeps_champion_but_still_predicts`.
  - `test_failed_health_check_makes_no_recommendations`.
  - `test_fewer_eligible_than_top_n` (Review Focus 1).
  - `test_matured_predictions_feed_live_scorecard`.
  - `test_simulation_does_not_touch_live_ledger`.
  - `test_search_promotes_only_through_gate` — `compare_recipes` sahte ile `promote=False` → `recipe.json` yazılmaz.
- [ ] FAIL → yaz → PASS.
- [ ] Gerçek veriyle yerel çalıştırma (önce `python -m jarvis.pipeline --data-dir data`): `python -m jarvis.train --data-dir data --simulate`, sonra `python -m jarvis.train --data-dir data`. Süreleri, eğitim satır sayısını ve simülasyon karnesini kullanıcıya göster. Simülasyon karnesi kötüyse (ör. negatif fazla getiri) **bunu olduğu gibi bildir**; iyileştirme için geri dönüp sonuca bakarak ayar yapma.
- [ ] Commit: `feat(jarvis): daily training job with simulation and recipe search`

### Task 9: İş akışı

**Files:** Modify `.github/workflows/daily_data.yml`, `requirements-data.txt`

- [ ] `requirements-data.txt`'e `xgboost` ekle.
- [ ] "Update data" adımından sonra "Train and predict": `python -m jarvis.train --data-dir data | tee train.json`; özeti `$GITHUB_STEP_SUMMARY`'ye ekle. Commit adımı aynı kalır.
- [ ] Yeni `schedule`: `cron: "0 7 * * 6"` (Cumartesi 10:00 TRT). O tetikte veri adımları atlanır, yalnızca `python -m jarvis.train --data-dir data --search` çalışır ve commit edilir. `workflow_dispatch`'e `mode` girdisi: `daily` (varsayılan) / `simulate` / `search`.
- [ ] `main`'e birleştikten sonra: `gh workflow run daily_data.yml -f mode=simulate`, ardından `-f mode=daily`. Beklenen: ikisi de yeşil, toplam 45 dakikanın altında; `data` dalında `models/`, `ledger/`, `scorecard/`, `daily/` var.
- [ ] Commit: `ci: daily training, weekly recipe search`

---

## Bu parçanın bilinen sınırları

- İlk haftalarda canlı karne boş olacak (ilk tahminler 5 seans sonra notlanır); o zamana kadar yalnızca simülasyon karnesi var ve o gerçek para sonucu değildir.
- Eski `bist_xgb_model.json` ve `train_xgboost.py` kaldırılmaz; Parça 3-4'te PC uygulaması yeni modele geçince emekliye ayrılır.
- Haber duygusu modelin girdisi değildir (tasarım 6.4).
