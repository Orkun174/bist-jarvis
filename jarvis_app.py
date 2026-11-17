"""BIST Jarvis PC application: today's picks, stock research, scorecard.

Run with:  streamlit run jarvis_app.py
The data comes from the repository's `data` branch, downloaded at start-up into
%LOCALAPPDATA%/BIST_AI_Radar/data (override with JARVIS_DATA_DIR, e.g. `data`
to look at a local working copy without downloading anything).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from jarvis import desktop

SCREENS = ["Günün önerileri", "Hisse araştır", "Karne ve model"]
TONE_MARK = {"up": "🟢", "down": "🔴", "flat": "⚪"}
CHART_SESSIONS = 130

st.set_page_config(page_title="BIST Jarvis", page_icon="📡", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .stApp, [data-testid="stHeader"] { background-color: #080d14; color: #e2e8f0; }
    [data-testid="stMetric"] { background: #101823; border: 1px solid #223047;
        border-radius: 8px; padding: 10px; }
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] { color: #a8b6ca; }
    h1, h2, h3, [data-testid="stMetricValue"] { color: #f1f5f9; }
    </style>
    """,
    unsafe_allow_html=True,
)


def data_dir() -> tuple[Path, bool]:
    """The folder to read, and whether it is ours to overwrite by downloading."""
    override = os.environ.get("JARVIS_DATA_DIR", "").strip()
    if override:
        return Path(override), False
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "BIST_AI_Radar"
    return base / "data", True


def percent(value, decimals: int = 2, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "–"
    text = f"{value * 100:+.{decimals}f}" if signed else f"{value * 100:.{decimals}f}"
    return "%" + text.replace(".", ",")


def open_stock(ticker: str) -> None:
    st.session_state["ticker"] = ticker
    st.session_state["screen"] = SCREENS[1]


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_news(ticker: str) -> list[dict]:
    from core.market_data import fetch_news
    return fetch_news(ticker, limit=5)


def news_for(ticker: str) -> list[dict] | None:
    """Headlines, or None when they cannot be fetched (failures are not cached)."""
    if os.environ.get("JARVIS_NO_NEWS"):
        return None
    try:
        return _fetch_news(ticker)
    except Exception:
        return None


# -- screens -----------------------------------------------------------------

def picks_screen(folder: Path) -> None:
    days = desktop.sessions(folder)
    if not days:
        st.info("Henüz günlük sonuç yok.")
        return
    day = st.selectbox("Gün", days, index=0)
    view = desktop.day_view(folder, day)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Günlük mesaj")
        st.text(view["message"])
    with right:
        st.subheader("Model sıralaması")
        ranked = desktop.rankings(folder, day)
        if ranked.empty:
            st.caption("Bu gün için sıralama kaydı yok.")
        for row in ranked.head(15).itertuples():
            label = f"{row.rank}. {row.ticker}" + ("  ★" if row.recommended else "")
            st.button(label, key=f"pick-{day}-{row.ticker}", on_click=open_stock,
                      args=(row.ticker,), width="stretch")
        st.caption("★ günün önerisi. Hisseye tıklayınca araştırma ekranı açılır.")
        if len(ranked) > 20:
            st.subheader("Modelin en zayıf gördükleri")
            for row in ranked.tail(5).iloc[::-1].itertuples():
                st.button(f"{row.rank}. {row.ticker}", key=f"last-{day}-{row.ticker}",
                          on_click=open_stock, args=(row.ticker,), width="stretch")
            st.caption("Sıralamanın sonundaki hisseler. Simülasyonda modelin en tutarlı olduğu "
                       "yer burasıydı: en alttakiler ortalamadan belirgin kötü gitti.")


def stock_chart(equity: pd.DataFrame, ticker: str) -> go.Figure:
    close = equity["Close"]
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    shown = equity.tail(CHART_SESSIONS)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                           vertical_spacing=0.03)
    figure.add_trace(go.Candlestick(x=shown.index, open=shown["Open"], high=shown["High"],
                                    low=shown["Low"], close=shown["Close"], name=ticker), 1, 1)
    figure.add_trace(go.Scatter(x=shown.index, y=ema20.reindex(shown.index), name="EMA 20",
                                line={"width": 1.3, "color": "#38bdf8"}), 1, 1)
    figure.add_trace(go.Scatter(x=shown.index, y=ema50.reindex(shown.index), name="EMA 50",
                                line={"width": 1.3, "color": "#f59e0b"}), 1, 1)
    figure.add_trace(go.Bar(x=shown.index, y=shown["Volume"], name="Hacim",
                            marker_color="#475569"), 2, 1)
    figure.update_layout(template="plotly_dark", height=520, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                         paper_bgcolor="#080d14", plot_bgcolor="#080d14",
                         xaxis_rangeslider_visible=False, legend={"orientation": "h"})
    figure.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    return figure


def stock_screen(folder: Path) -> None:
    table = desktop.universe(folder)
    if table.empty:
        st.info("Hisse listesi yok.")
        return
    tickers = list(table["ticker"])
    if st.session_state.get("ticker") not in tickers:
        st.session_state["ticker"] = tickers[0]
    ticker = st.selectbox("Hisse (yazarak ara)", tickers, key="ticker")
    status = desktop.stock_status(folder, ticker)
    equity = desktop.prices(folder, ticker)

    first, second, third, fourth = st.columns(4)
    if not equity.empty:
        last = equity["Close"].iloc[-1]
        change = equity["Close"].pct_change(fill_method=None).iloc[-1] if len(equity) > 1 else None
        first.metric("Kapanış", f"{last:,.2f}".replace(",", " ").replace(".", ","), percent(change))
    second.metric("Model sırası", f"{status['rank']} / {status['of']}" if status.get("rank") else "–")
    third.metric("Uygunluk", "Uygun" if status.get("eligible") else "Dışlandı")
    fourth.metric("Medyan hacim (20 gün)", f"{status.get('median_turnover', 0) / 1e6:,.0f} mn TL".replace(",", "."))

    if status.get("eligible"):
        note = status["reason_text"]
        if status.get("recommended"):
            note += " Bugünün önerileri arasında."
        st.success(note)
    else:
        st.warning(status.get("reason_text", "Bu hisse listede yok."))
    st.caption("Model sırası, hissenin 5 seansta endeksi geçme olasılığına göre bugünkü yeridir. "
               "Model henüz kanıtlanmadı; sıra bir tahmin değil, bir sıralamadır.")

    if equity.empty:
        st.info("Bu hisse için fiyat verisi yok.")
        return
    st.plotly_chart(stock_chart(equity, ticker), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("İndikatörler")
        readings = desktop.indicator_readings(equity, desktop.benchmark_close(folder))
        if not readings:
            st.caption("İndikatör hesaplamak için yeterli geçmiş yok.")
        for item in readings:
            st.markdown(f"{TONE_MARK[item['tone']]} **{item['name']}** {item['value']} — {item['reading']}")
        st.caption("🟢 yukarı işaret · 🔴 aşağı işaret · ⚪ yön bildirmez. Bunlar grafiğin "
                   "özetidir, öngörü değildir.")
    with right:
        st.subheader("Haberler")
        news = news_for(ticker)
        if news is None:
            st.caption("Haberler şu an alınamadı. Hisseyi yeniden seçince tekrar denenir.")
        elif not news:
            st.caption("Güncel haber bulunamadı.")
        else:
            for item in news:
                title, link = item.get("title", ""), item.get("url") or item.get("link")
                day = str(item.get("published_at") or "")[:10]
                line = f"[{title}]({link})" if link else title
                st.markdown(f"- {line}" + (f" · {day}" if day else ""))
            st.caption("Kaynak: Google Haberler. Başlıklar otomatik aranır; ilgisiz olabilir.")


def scorecard_screen(folder: Path) -> None:
    view = desktop.day_view(folder)
    live = ((view or {}).get("live") or {}).get("windows") or {}
    simulated = (view or {}).get("simulated") or {}

    st.subheader("Canlı karne")
    columns = st.columns(3)
    for column, (key, title) in zip(columns, (("20", "Son 20 seans"), ("60", "Son 60 seans"), ("all", "Tümü"))):
        window = live.get(key) or {}
        column.metric(title, f"{window.get('recommendations', 0)} öneri")
        column.caption(f"isabet {percent(window.get('hit_rate'), 0, signed=False)} · "
                       f"ort. fazla getiri {percent(window.get('mean_excess'))}")
    graded = desktop.graded_recommendations(folder)
    if graded.empty:
        st.caption("Henüz notlanan öneri yok. Öneriler 5 işlem günü sonra notlanır.")
    else:
        by_day = graded.groupby("date")["fwd_excess"].mean().sort_index()
        st.line_chart(by_day.cumsum().rename("Birikimli fazla getiri (öneri ortalaması)"))
        st.dataframe(graded, width="stretch", hide_index=True)

    st.subheader("Simülasyon (gerçek işlem değildir)")
    net = simulated.get("top5_net_of_cost") or {}
    first, second, third, fourth, fifth = st.columns(5)
    first.metric("Seans", net.get("days", "–"))
    second.metric("İlk 5 ort. fazla getiri", percent(net.get("mean_top_excess")))
    fifth.metric("Ertesi açılıştan alınsaydı", percent(net.get("mean_top_excess_open")))
    third.metric("Kazanan gün oranı", percent(net.get("hit_rate"), 0, signed=False))
    fourth.metric("Rank IC", f"{net['mean_rank_ic']:.3f}".replace(".", ",") if net.get("mean_rank_ic") is not None else "–")
    if net.get("mean_universe_excess") is not None:
        first, second, third = st.columns(3)
        first.metric("Ortalama uygun hisse", percent(net.get("mean_universe_excess")))
        second.metric("İlk 5 − ortalama hisse", percent(net.get("mean_top_vs_universe")))
        third.metric("Son 5", percent(net.get("mean_bottom_excess")))
        if net.get("mean_quintile_spread") is not None:
            st.metric("En iyi beşte bir − en kötü beşte bir", percent(net.get("mean_quintile_spread")))
            st.caption("Sıralamanın işe yarayıp yaramadığının en az gürültülü ölçüsü: çok sayıda "
                       f"hisseye dayanır. Günlerin {percent(net.get('quintile_spread_win_rate'), 0, signed=False)}"
                       " kadarında artı.")
        st.caption("Endeks büyük şirket ağırlıklıdır; ortalama uygun hisse aylarca endeksin gerisinde "
                   "ya da önünde kalabilir. 'İlk 5 − ortalama hisse' modelin kendi katkısını gösterir "
                   "(maliyet öncesi).")
    st.caption("Maliyet düşülmüş, geçmişe dönük ileriye yürütme; 5 seanslık getiri, endekse göre. "
               "Mesaj kapanıştan sonra geldiği için gerçekte alınabilecek ilk fiyat ertesi açılıştır. "
               "Sıfıra yakın getiri, modelin kazancının kanıtlanmadığı anlamına gelir.")

    st.subheader("Model geçmişi")
    history = desktop.model_history(folder)
    if history.empty:
        st.caption("Kayıt yok.")
    else:
        st.dataframe(history.rename(columns={
            "session": "Seans", "recipe_id": "Tarif", "retrained": "Eğitildi",
            "weights_changed": "Ağırlık değişti", "data_ok": "Veri", "health_ok": "Sağlık",
            "drift_ok": "Kayma", "rank_ic": "Rank IC (120 seans)", "train_rows": "Satır",
        }), width="stretch", hide_index=True)

    st.subheader("Haftalık tarif sınavları")
    exams = desktop.recipe_exams(folder)
    if exams.empty:
        st.caption("Henüz sınav yapılmadı (cumartesi günleri).")
    else:
        st.dataframe(exams.rename(columns={
            "session": "Seans", "candidates": "Aday", "promoted": "Terfi", "champion": "Şampiyon",
            "challenger": "Rakip", "top_excess_diff": "İlk 5 farkı", "rank_ic_diff": "Rank IC farkı",
            "pick_metric": "Sınav ölçüsü", "pick_diff": "Ölçü farkı", "p_pick": "Güven (ölçü)",
            "p_rank_ic": "Güven (rank IC)", "confirmed": "250 seans doğrulaması",
        }), width="stretch", hide_index=True)

    quality = (view or {}).get("quality")
    st.subheader("Veri kalitesi")
    if not quality:
        st.caption("Kalite raporu yok.")
    else:
        reasons = quality.get("reasons") or {}
        st.dataframe(pd.DataFrame(
            [{"Durum": desktop.REASON_TEXT.get("" if key == "eligible" else key, key), "Hisse": count}
             for key, count in sorted(reasons.items(), key=lambda item: -item[1])]
        ), width="stretch", hide_index=True)
        quarantined = quality.get("quarantined") or {}
        if quarantined:
            st.caption("Karantinadaki hisseler: " + ", ".join(sorted(quarantined)))


# -- page --------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def synced(folder_text: str) -> dict:
    """Download once per app process, not once per browser tab or session."""
    return desktop.sync_data(Path(folder_text))


folder, downloadable = data_dir()
if downloadable:
    with st.spinner("Güncel veri indiriliyor…"):
        st.session_state["sync"] = synced(str(folder))

st.title("BIST Jarvis")
sync = st.session_state.get("sync")
header, refresh = st.columns([5, 1])
if not downloadable:
    header.caption(f"Yerel klasör okunuyor: {folder}")
elif sync and sync["ok"]:
    header.caption(f"Veri güncel (indirildi: {sync['synced_at']} UTC).")
elif sync and sync["has_local"]:
    header.warning(f"Güncel veri indirilemedi ({sync['error']}). Son yerel kopya gösteriliyor"
                   + (f": {sync['synced_at']}" if sync.get("synced_at") else "."))
if downloadable and refresh.button("Veriyi yenile"):
    synced.clear()
    st.cache_data.clear()
    st.rerun()

if not (folder / "universe" / "universe.parquet").exists():
    st.error("Veri yok: indirilemedi ve yerel kopya bulunamadı. İnternet bağlantısını kontrol edin.")
    st.stop()

screen = st.radio("Ekran", SCREENS, key="screen", horizontal=True, label_visibility="collapsed")
if screen == SCREENS[0]:
    picks_screen(folder)
elif screen == SCREENS[1]:
    stock_screen(folder)
else:
    scorecard_screen(folder)
st.caption("Yatırım tavsiyesi değildir.")
