# app.py
"""BIST AI Radar frontend with backend-owned ML feature validation."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from numbers import Real

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.ai_analyzer import analyze
from core.market_data import (
    MarketDataError,
    NewsError,
    fetch_market_data,
    fetch_news,
    market_snapshot,
    normalize_tickers,
)


LOGGER = logging.getLogger(__name__)

# This versions only the UI's stored result structure, not the ML contract.
# Incrementing it clears obsolete session results without triggering inference.
UI_STATE_VERSION = 5

st.set_page_config(
    page_title="BIST AI Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stHeader"] {
        background-color: #080d14;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"], [data-testid="stForm"] {
        background-color: #101823;
    }
    [data-testid="stMetric"] {
        background: #101823;
        border: 1px solid #223047;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {
        color: #a8b6ca;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    .stFormSubmitButton > button {
        border: 1px solid #25d0a0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


METHODOLOGY = {
    "RSI · 14": (
        "100 − 100/(1 + average gain/average loss), using Wilder smoothing.",
        "Above 70 indicates overbought conditions; below 30 indicates oversold "
        "conditions. Neither guarantees a reversal.",
        "Provides secondary momentum evidence after trend and volume assessment.",
    ),
    "MACD · 12/26/9": (
        "EMA12 − EMA26; the signal is EMA9 of MACD; the histogram is "
        "MACD minus its signal.",
        "Compare MACD with its signal and zero. A crossover requires prior bars.",
        "Checks momentum alignment and supplies local statistical model inputs.",
    ),
    "EMA · 20/50": (
        "Recursive weighted averages with alpha = 2/(period + 1).",
        "Price > EMA20 > EMA50 supports bullish alignment; the inverse supports "
        "bearish alignment.",
        "Provides primary direction, qualified by ADX and volume evidence.",
    ),
    "Bollinger Bands · 20, 2σ": (
        "SMA20 ± two population standard deviations; %B locates price within "
        "the band range.",
        "%B ≤0.2 is near the lower band; ≥0.8 is near the upper. "
        "Extremes can persist during trends.",
        "Provides continuation or mean-reversion context when combined with ADX.",
    ),
    "ATR · 14": (
        "Wilder average of max(high−low, |high−previous close|, "
        "|low−previous close|).",
        "Measures movement size in TRY; ATR/close expresses relative range.",
        "Adds nondirectional volatility context and a local ML risk feature.",
    ),
    "OBV · five-bar change": (
        "Cumulative volume signed by close-to-close direction; the initial "
        "level is zero.",
        "Use the sign of OBV[t]−OBV[t−5]. It does not establish monotonic movement.",
        "Checks volume support without identifying institutional participants.",
    ),
    "ADX · 14": (
        "Wilder average of DX, where DX=100×|+DI−−DI|/(+DI+−DI).",
        "ADX ≥25 supports a trending regime; <20 indicates weak trend evidence.",
        "Controls trust in EMA alignment; low ADX requires confirmation "
        "before considering a fade.",
    ),
    "CMF · 20": (
        "Sum of volume×(2×close−high−low)/(high−low), divided by 20-bar volume.",
        "Positive values support closes near daily highs; negative values "
        "near daily lows. Zero-volume windows are unavailable.",
        "Validates volume support and supplies an ML feature; it does not "
        "prove institutional accumulation.",
    ),
    "StochRSI · 14/3/3": (
        "RSI14 normalized within its 14-bar range, followed by SMA3 %K and SMA3 %D.",
        "On a 0–100 scale, >80 is overbought and <20 oversold. "
        "Flat RSI ranges receive a neutral raw value of 50.",
        "Provides sensitive timing evidence; it is not independent of RSI.",
    ),
    "Macro return features": (
        "Native-observation percentage returns and five-return rolling means "
        "for USDTRY, Brent, Gold, S&P500, VIX, BIST100, and BIST Bank.",
        "Interpret as lagged momentum inputs. Signals become eligible the next "
        "calendar day and are forward-filled across market holidays.",
        "Adds currency, commodity, global risk, and domestic index context "
        "to the statistical model.",
    ),
    "XGBoost · three-session event": (
        "A binary classifier scores Close[t+3]/Close[t]−1 ≥2%, using local "
        "technical indicators and macro features.",
        "XGB_Prob is an uncalibrated percentage score. A low score does not "
        "necessarily predict a decline; gains below 2% belong to class 0.",
        "Provides statistical evidence for Qwen to reconcile with news. "
        "Shared technical inputs must not be counted as independent votes.",
    ),
}


def initialize_state() -> None:
    """Initialize UI storage and discard snapshots from older UI implementations."""
    if st.session_state.get("radar_state_version") != UI_STATE_VERSION:
        st.session_state.update(
            radar_state_version=UI_STATE_VERSION,
            radar_results={},
            radar_completed_at=None,
            radar_running=False,
            radar_run_id=0,
        )


def valid_probability(value: object) -> float | None:
    """Validate the displayed percentage only, without inspecting model contracts."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None

    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 100.0:
        return None
    return probability


def extract_probability(hybrid: dict) -> tuple[float | None, str | None]:
    """
    Prefer the canonical backend evidence score.

    Feature counts, feature names, and model version strings are intentionally
    not checked here. Those checks belong to core.ai_analyzer.

    XGB_Prob remains supported as the backend's compatibility alias.
    """
    evidence = hybrid.get("ML_Evidence")
    evidence = evidence if isinstance(evidence, dict) else {}

    # Do not resurrect a stale alias when the backend explicitly reports failure.
    if evidence.get("status") == "unavailable":
        reason = hybrid.get("ML_Error") or evidence.get("reason")
        return None, str(reason or "The statistical score is unavailable.")

    canonical = evidence.get("XGBoost_Uptrend_Probability_Percent")
    if canonical is not None:
        probability = valid_probability(canonical)
        if probability is None:
            return None, "The backend returned an invalid probability percentage."
        return probability, None

    probability = valid_probability(hybrid.get("XGB_Prob"))
    if probability is not None:
        return probability, None

    reason = hybrid.get("ML_Error") or evidence.get("reason")
    return None, str(reason or "No statistical probability was returned.")


def make_chart(
    frame: pd.DataFrame,
    ticker: str,
    run_id: int,
) -> go.Figure:
    """Preserve candles, EMA/Bollinger overlays, and the volume subplot."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
    )

    # Consecutive traces allow tonexty to fill the Bollinger region correctly.
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["BB_Lower"],
            name="BB lower",
            mode="lines",
            line={"color": "#64748b", "width": 1},
            legendgroup="bollinger",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["BB_Upper"],
            name="BB upper",
            mode="lines",
            line={"color": "#64748b", "width": 1},
            fill="tonexty",
            fillcolor="rgba(100,116,139,0.12)",
            legendgroup="bollinger",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["BB_Middle"],
            name="BB SMA20",
            mode="lines",
            line={"color": "#94a3b8", "width": 1, "dash": "dot"},
            legendgroup="bollinger",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name=ticker,
            increasing_line_color="#25d0a0",
            decreasing_line_color="#ff637d",
        ),
        row=1,
        col=1,
    )

    for column, label, color in (
        ("EMA_20", "EMA20", "#38bdf8"),
        ("EMA_50", "EMA50", "#fbbf24"),
    ):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=label,
                mode="lines",
                line={"color": color, "width": 1.8},
            ),
            row=1,
            col=1,
        )

    volume_colors = [
        "#25d0a0" if close >= opening else "#ff637d"
        for opening, close in zip(frame["Open"], frame["Close"])
    ]
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["Volume"],
            marker_color=volume_colors,
            name="Volume",
            showlegend=False,
            hovertemplate="%{x}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080d14",
        plot_bgcolor="#080d14",
        font={"color": "#cbd5e1", "family": "Consolas, monospace"},
        height=550,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
        uirevision=f"{ticker}-{run_id}",
    )
    fig.update_xaxes(
        showgrid=False,
        rangebreaks=[{"bounds": ["sat", "mon"]}],
    )
    fig.update_yaxes(gridcolor="#1c293b")
    fig.update_yaxes(title_text="TRY", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def render_metrics(snapshot: dict) -> None:
    """Retain the compact three-row technical indicator grid."""
    row = st.columns(4)
    row[0].metric(
        "Close · TRY",
        f"{snapshot['price']:,.2f}",
        f"{snapshot['change_pct']:+.2f}%",
    )
    row[1].metric("RSI · 14", f"{snapshot['rsi']:.2f}")
    row[2].metric(
        "MACD · 12/26",
        f"{snapshot['macd']:.3f}",
        help=f"Signal: {snapshot['macd_signal']:.4f}",
    )
    row[3].metric(
        "ATR · 14",
        f"{snapshot['atr_14']:.2f}",
        help=f"ATR / close: {snapshot['atr_pct']:.2f}%",
    )

    row = st.columns(4)
    row[0].metric("EMA · 20", f"{snapshot['ema_20']:,.2f}")
    row[1].metric("EMA · 50", f"{snapshot['ema_50']:,.2f}")
    row[2].metric(
        "BB position",
        snapshot["bb_position"],
        help=(
            f"Lower: {snapshot['bb_lower']:.2f}; "
            f"upper: {snapshot['bb_upper']:.2f}"
        ),
    )
    row[3].metric(
        "OBV · five bars",
        snapshot["obv_trend"],
        help=f"Net change: {snapshot['obv_change']:,.0f}",
    )

    row = st.columns(4)
    row[0].metric(
        "ADX · 14",
        f"{snapshot['adx_14']:.2f}",
        help=f"Regime: {snapshot['adx_regime']}",
    )
    cmf = snapshot["cmf_20"]
    row[1].metric("CMF · 20", "N/A" if cmf is None else f"{cmf:+.3f}")
    row[2].metric("StochRSI · %K", f"{snapshot['stochrsi_k']:.2f}")
    row[3].metric("StochRSI · %D", f"{snapshot['stochrsi_d']:.2f}")


def run_analysis(tickers: list[str]) -> None:
    """Fetch and infer exclusively after explicit form submission."""
    st.session_state.radar_running = True
    st.session_state.radar_run_id += 1
    st.session_state.radar_results = {}
    st.session_state.radar_completed_at = None
    progress = st.progress(0.0, text="Starting hybrid analysis…")

    try:
        for position, ticker in enumerate(tickers):
            progress.progress(
                position / len(tickers),
                text=f"{ticker}: technical data → macro-aware XGBoost → Qwen…",
            )
            result = {
                "frame": None,
                "snapshot": None,
                "news": [],
                "warnings": [],
                "error": None,
                "hybrid": None,
                "hybrid_error": None,
            }

            try:
                frame = fetch_market_data(ticker)
                snapshot = market_snapshot(frame)
                result["frame"] = frame
                result["snapshot"] = snapshot

                news_status = "unavailable"
                try:
                    result["news"] = fetch_news(ticker)
                    news_status = "available" if result["news"] else "empty"
                    if not result["news"]:
                        result["warnings"].append(
                            "No matching RSS headlines were returned."
                        )
                except NewsError as exc:
                    result["warnings"].append(str(exc))

                try:
                    # Store the backend envelope unchanged. No feature-count
                    # assertion or model-version assertion belongs in the UI.
                    hybrid = analyze(
                        ticker=ticker,
                        snapshot=snapshot,
                        headlines=result["news"],
                        news_status=news_status,
                    )
                    if not isinstance(hybrid, dict):
                        raise TypeError("The analysis backend returned no dictionary.")
                    result["hybrid"] = hybrid
                except Exception:
                    LOGGER.exception("Unexpected hybrid failure for %s", ticker)
                    result["hybrid_error"] = (
                        "Unexpected hybrid inference error. Check the terminal."
                    )

            except MarketDataError as exc:
                result["error"] = str(exc)
            except Exception:
                LOGGER.exception("Unexpected market failure for %s", ticker)
                result["error"] = (
                    "Unexpected market processing error. Check the terminal."
                )

            updated = dict(st.session_state.radar_results)
            updated[ticker] = result
            st.session_state.radar_results = updated

        st.session_state.radar_completed_at = datetime.now(
            timezone.utc
        ).isoformat()
    finally:
        st.session_state.radar_running = False
        progress.empty()


def render_hybrid(result: dict) -> None:
    """Render the backend's probability without imposing an ML contract."""
    hybrid = result.get("hybrid")
    hybrid = hybrid if isinstance(hybrid, dict) else {}

    evidence = hybrid.get("ML_Evidence")
    evidence = evidence if isinstance(evidence, dict) else {}

    probability, probability_error = extract_probability(hybrid)

    st.markdown("### Statistical prediction")
    st.metric(
        "XGB_Prob (%)",
        "N/A" if probability is None else f"{probability:.1f}%",
        help=(
            "Raw XGBoost score for a ≥2% terminal close-to-close gain "
            "over three trading sessions."
        ),
    )
    st.caption(
        "Three trading sessions · ≥2% terminal gain · Uncalibrated model score"
    )

    # Only actual backend/data errors can explain an unavailable score.
    # A feature count or contract version is never used to suppress a valid score.
    if result.get("hybrid_error"):
        st.error(result["hybrid_error"])
    elif probability is None and probability_error:
        st.warning(probability_error)

    if probability is not None:
        if evidence.get("ticker_in_training_universe") is False:
            st.warning("This ticker was not part of the model's training universe.")

        validation = evidence.get("validation")
        if isinstance(validation, dict):
            loss = validation.get("log_loss")
            baseline = validation.get("baseline_log_loss")
            if (
                isinstance(loss, Real)
                and isinstance(baseline, Real)
                and math.isfinite(float(loss))
                and math.isfinite(float(baseline))
                and loss >= baseline
            ):
                st.warning(
                    "The validation model did not outperform its frequency "
                    "baseline on log loss."
                )

    if evidence:
        with st.expander("ML provenance and validation"):
            # Feature metadata remains visible for diagnostics, never gating UI.
            st.json(evidence)

    st.markdown("### Portfolio manager assessment")

    if hybrid.get("AI_Error"):
        st.error(str(hybrid["AI_Error"]))

    analysis = hybrid.get("Analysis")
    if isinstance(analysis, dict):
        sentiment = analysis.get("Sentiment", "Neutral")
        renderer = {
            "Bullish": st.success,
            "Bearish": st.error,
            "Neutral": st.info,
        }.get(sentiment, st.info)
        renderer(f"Sentiment: {sentiment}")

        score = analysis.get("Conviction_Score")
        st.metric(
            "Qwen conviction",
            f"{score} / 10" if score is not None else "N/A",
        )
        st.caption("Qualitative conviction is separate from XGB_Prob.")

        # Preserve Markdown formatting without enabling model-supplied HTML.
        st.markdown("**Technical context**")
        st.markdown(str(analysis.get("Technical_Context", "")))

        st.markdown("**News impact**")
        st.markdown(str(analysis.get("News_Impact", "")))

        with st.expander("Validated Qwen JSON"):
            st.json(analysis)


def render_result(ticker: str, result: dict) -> None:
    """Render persisted data without invoking network calls or either model."""
    st.subheader(ticker)
    if result["error"]:
        st.error(result["error"])
        return

    snapshot = result["snapshot"]
    st.caption(
        f"{ticker}.IS · Latest daily bar: {snapshot['bar_time']} · "
        f"Retrieved: {snapshot['fetched_at']}"
    )

    chart_column, analysis_column = st.columns([1.9, 1], gap="large")

    with chart_column:
        render_metrics(snapshot)
        st.plotly_chart(
            make_chart(
                result["frame"],
                ticker,
                st.session_state.radar_run_id,
            ),
            use_container_width=True,
            theme=None,
            key=f"chart_{ticker}",
            on_select="ignore",
            config={"displaylogo": False, "scrollZoom": True},
        )

    with analysis_column:
        render_hybrid(result)

        for warning in result["warnings"]:
            st.warning(warning)

        with st.expander("Latest matching headlines", expanded=True):
            for number, item in enumerate(result["news"], start=1):
                st.text(f"{number}. {item['title']}")
                st.caption(
                    item["published_at"] or "Publication time unavailable"
                )
                if item["url"]:
                    st.link_button(f"Open article {number}", item["url"])

    st.divider()


def render_methodology() -> None:
    """Generate the educational section without model inference."""
    with st.expander("Indicator Methodology & Logic", expanded=False):
        st.markdown(
            "The dashboard combines **regime-aware technical evidence**, "
            "**macro-aware statistical prediction**, and **headline analysis**."
        )

        headings = (
            "What it measures",
            "How to use it",
            "Why it matters in this AI model",
        )
        for indicator, explanations in METHODOLOGY.items():
            st.markdown(f"#### {indicator}")
            for heading, explanation in zip(headings, explanations):
                st.markdown(f"**{heading}:** {explanation}")
            st.divider()

        st.markdown(
            "**Training logic:** Local indicators are calculated independently "
            "for each equity. Macro returns use native observations before "
            "lagged alignment. Validation uses a chronological split and purges "
            "training labels crossing its boundary."
        )
        st.caption(
            "Prices are unadjusted. Corporate actions, initialization, and "
            "changing price scales can affect inputs. Training uses completed "
            "sessions; live daily bars may be unfinished. Model scores are "
            "not calibrated success rates."
        )


initialize_state()

st.title("BIST AI RADAR")
st.caption(
    "Technical indicators · Macro-aware XGBoost statistics · Local Qwen synthesis"
)

# Input editing is batched; only submission triggers data retrieval and inference.
with st.form("radar_form", clear_on_submit=False):
    raw_tickers = st.text_input(
        "BIST tickers, separated by commas",
        value="THYAO, KCHOL, FROTO",
        max_chars=200,
        help="FROTO and FROTO.IS are accepted. Maximum 10 unique tickers.",
    )
    submitted = st.form_submit_button(
        "Analyze / Refresh",
        type="primary",
        disabled=st.session_state.radar_running,
    )

if submitted and not st.session_state.radar_running:
    try:
        selected_tickers = normalize_tickers(raw_tickers)
    except ValueError as exc:
        st.error(str(exc))
    else:
        with st.spinner("Running the hybrid analysis…"):
            run_analysis(selected_tickers)

if st.session_state.radar_completed_at:
    st.caption(
        f"Snapshot completed: {st.session_state.radar_completed_at} · "
        "Click Analyze / Refresh to update."
    )

if st.session_state.radar_results:
    for symbol, stored_result in st.session_state.radar_results.items():
        render_result(symbol, stored_result)
else:
    st.info("Enter BIST tickers and click Analyze / Refresh.")

st.caption(
    "Charts display three months; local indicators use six months for "
    "initialization. Macro histories are cached by the backend. "
    "Yahoo daily bars may be delayed or unfinished."
)

render_methodology()