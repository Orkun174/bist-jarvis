"""Offline tests for causal watchlists, reporting, news, and safe delivery."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import requests

import daily_scan as scan
from core import daily_radar as radar


def prices(n=170):
    dates = pd.bdate_range("2026-01-01", periods=n)
    x = np.arange(n)
    close = 100 + .12*x + 1.5*np.sin(x/3)
    return pd.DataFrame({"Open": close-.15, "High": close+1,
                         "Low": close-1, "Close": close,
                         "Volume": 2_000_000.0}, index=dates)


def candidate_frame(benchmark):
    frame = radar.technical_frame(benchmark, benchmark)
    day = frame.index[-1]
    frame.loc[day, ["ready", "score", "ema20", "ema50", "relative5", "cmf",
                    "volume_ratio", "rsi", "atr_pct", "extension_atr", "value20",
                    "obv5", "atr", "resistance", "support"]] = [
        True, 90.0, 100.0, 95.0, .03, .12, 1.5, 60.0, 2.0, 1.0,
        100_000_000.0, 1_000_000.0, 2.0, 130.0, 110.0,
    ]
    return frame


def sample_report(candidates=None):
    benchmark = prices()
    frame = candidate_frame(benchmark)
    day = benchmark.index[-1]
    news = {"THYAO": {"status": "scored", "sentiment": .3, "count": 1,
                       "headlines": ["Şirketin üretim verileri açıklandı"]}}
    selected, context = radar.select_candidates({"THYAO": frame}, benchmark, day, news)
    return {
        "status": "ok", "session": str(day.date()), "target_session": "2026-10-30",
        "scanned_at": "2026-10-29T18:30:00+03:00", "coverage": 20,
        "universe_size": 20, "market": context, "macros": {}, "warnings": [],
        "failures": {}, "candidates": selected if candidates is None else candidates,
        "history": {"signal_sessions": 40, "positive_session_rate": .55,
                    "mean_net_return": .001, "cost_bps": 20.0},
    }


class TechnicalTests(unittest.TestCase):
    def test_future_prices_cannot_change_past_indicators(self):
        equity = prices()
        benchmark = prices()
        baseline = radar.technical_frame(equity, benchmark)
        altered = equity.copy()
        altered.loc[altered.index[-10:], radar.OHLCV[:4]] *= 3
        changed = radar.technical_frame(altered, benchmark)
        pd.testing.assert_frame_equal(baseline.iloc[:-10], changed.iloc[:-10])

    def test_missing_recent_exchange_session_invalidates_latest_row(self):
        benchmark = prices()
        equity = benchmark.drop(benchmark.index[-3])
        frame = radar.technical_frame(equity, benchmark)
        self.assertFalse(bool(frame.ready.iloc[-1]))

    def test_warmup_and_zero_volume_do_not_produce_candidates(self):
        benchmark = prices()
        frame = radar.technical_frame(benchmark, benchmark)
        self.assertFalse(frame.ready.iloc[:49].any())
        altered = benchmark.copy()
        altered.loc[altered.index[-1], "Volume"] = 0
        self.assertFalse(bool(radar.technical_frame(altered, benchmark).ready.iloc[-1]))

    def test_corrupt_prices_are_rejected(self):
        equity = prices()
        equity.loc[equity.index[-1], "High"] = 1
        with self.assertRaises(ValueError):
            radar.validate_prices(equity)

    def test_no_candidate_is_forced_when_thresholds_fail(self):
        benchmark = prices()
        frame = candidate_frame(benchmark)
        frame.loc[frame.index[-1], "score"] = 20.0
        selected, _ = radar.select_candidates({"THYAO": frame}, benchmark, frame.index[-1])
        self.assertEqual(selected, [])

    def test_negative_news_excludes_otherwise_qualified_stock(self):
        benchmark = prices()
        frame = candidate_frame(benchmark)
        day = frame.index[-1]
        selected, _ = radar.select_candidates({"THYAO": frame}, benchmark, day)
        self.assertEqual(len(selected), 1)
        selected, _ = radar.select_candidates(
            {"THYAO": frame}, benchmark, day,
            {"THYAO": {"sentiment": -.7, "status": "scored"}},
        )
        self.assertEqual(selected, [])

    def test_adjusted_reference_prices_are_reported_on_raw_scale(self):
        benchmark = prices()
        frame = candidate_frame(benchmark)
        frame["AdjustmentFactor"] = .5
        selected, _ = radar.select_candidates({"THYAO": frame}, benchmark, frame.index[-1])
        self.assertAlmostEqual(selected[0]["close"], frame.Close.iloc[-1]/.5)
        self.assertEqual(selected[0]["resistance"], 260.0)


class DiagnosticTests(unittest.TestCase):
    def test_outcome_is_next_open_to_close_net_of_explicit_costs(self):
        benchmark = prices(70)
        frame = candidate_frame(benchmark)
        last = frame.index[-1]
        frame.loc[last, ["Open", "Close"]] = [100.0, 101.0]
        with patch.object(radar, "select_candidates", return_value=([{"ticker": "THYAO"}], {})):
            result = radar.historical_diagnostics({"THYAO": frame}, benchmark, lookback=1, cost_bps=20)
        self.assertEqual(result["signal_sessions"], 1)
        self.assertAlmostEqual(result["mean_net_return"], .008)
        self.assertEqual(result["trades"][0]["outcome_date"], str(last.date()))
        self.assertEqual(result["trades"][0]["signal_date"], str(frame.index[-2].date()))

    def test_missing_next_bar_is_not_replaced_by_later_return(self):
        benchmark = prices(70)
        frame = candidate_frame(benchmark).iloc[:-1]
        with patch.object(radar, "select_candidates", return_value=([{"ticker": "THYAO"}], {})):
            result = radar.historical_diagnostics({"THYAO": frame}, benchmark, lookback=1)
        self.assertEqual(result["signal_sessions"], 0)
        self.assertEqual(result["missing_outcomes"], 1)
        self.assertIsNone(result["mean_net_return"])


class ReportTests(unittest.TestCase):
    def test_report_is_readable_and_never_calls_score_probability(self):
        report = sample_report()
        message = radar.format_report(report)
        self.assertIn("Kapsama: 20/20", message)
        self.assertIn("THYAO", message)
        self.assertIn("Hedef seans:", message)
        self.assertIn("Destek", message)
        self.assertIn("Haber duygu", message)
        self.assertIn("Puan başarı olasılığı değildir", message)
        self.assertLessEqual(len(message.encode("utf-16-le"))//2, 4000)

    def test_no_candidates_and_partial_coverage_are_explicit(self):
        report = sample_report([])
        report.update(coverage=18, status="partial", failures={"FROTO": "ValueError"})
        message = radar.format_report(report)
        self.assertIn("18/20", message)
        self.assertIn("yalnızca doğrulanan", message)
        self.assertIn("ölçütleri sağlayan aday yok", message)

    def test_full_three_stock_report_fits_telegram(self):
        report = sample_report()
        candidate = report["candidates"][0]
        report["candidates"] = [dict(candidate, ticker=name) for name in ["THYAO", "FROTO", "ASELS"]]
        for item in report["candidates"]:
            item["news"]["headlines"] = ["Uzun haber başlığı "*50]
            item["risks"] = ["Destek altı kapanış teknik görünümü zayıflatır.",
                             "Geniş günlük fiyat aralığı.", "Haber analizi doğrulanamadı."]
        report["macros"] = {name: {"close": 100.0, "return1_pct": .2, "date": "2026-10-27"}
                             for name in scan.MACROS}
        report["warnings"] = ["Veri kaynağı hakkında ayrıntı "*30]
        report["history"]["evidence_status"] = "no_positive_net_return"
        message = radar.format_report(report)
        self.assertLessEqual(len(message.encode("utf-16-le"))//2, 4000)
        self.assertIn("kanıtlanmış kazanç avantajı yok", message)


class NewsTests(unittest.TestCase):
    def test_news_is_batched_once_and_future_headlines_are_excluded(self):
        since = pd.Timestamp("2026-10-28", tz=scan.TRT)
        cutoff = pd.Timestamp("2026-10-29 18:30:00+03:00")
        articles = [
            {"title": "Geçerli haber", "published_at": "2026-10-29T10:00:00+03:00"},
            {"title": "Gelecek haber", "published_at": "2026-10-30T10:00:00+03:00"},
            {"title": "Tarihsiz haber"},
        ]
        scorer = Mock(side_effect=lambda frame, *args: frame.assign(score=.5))
        provider = SimpleNamespace(fetch_news=Mock(return_value=articles))
        with (
            patch.dict("sys.modules", {"core.market_data": provider,
                                       "retrain_model": SimpleNamespace(score_corpus=scorer)}),
            patch.object(scan.time, "sleep"),
        ):
            result = scan.score_news(["THYAO", "FROTO"], cutoff, since)
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(len(scorer.call_args.args[0]), 2)
        self.assertEqual(result["THYAO"]["count"], 1)
        self.assertEqual(result["THYAO"]["sentiment"], .5)

    def test_nlp_failure_is_unknown_not_neutral(self):
        provider = SimpleNamespace(fetch_news=Mock(return_value=[
            {"title": "Başlık", "published_at": "2026-10-29T10:00:00+03:00"}
        ]))
        with (
            patch.dict("sys.modules", {"core.market_data": provider,
                "retrain_model": SimpleNamespace(score_corpus=Mock(side_effect=OSError("offline")))}),
            patch.object(scan.time, "sleep"),
        ):
            result = scan.score_news(["THYAO"], pd.Timestamp("2026-10-29 18:30+03:00"),
                                     pd.Timestamp("2026-10-28", tz=scan.TRT))
        self.assertEqual(result["THYAO"]["status"], "unavailable")
        self.assertIsNone(result["THYAO"]["sentiment"])


class IntegrationTests(unittest.TestCase):
    def test_report_uses_twenty_stocks_and_next_real_session(self):
        benchmark = prices(210)
        # End a realistic history on the Republic Day half session.
        benchmark.index = pd.bdate_range(end="2026-10-28", periods=len(benchmark))
        benchmark["AdjustmentFactor"] = 1.0
        benchmark["PriceSource"] = "yahoo_adjusted_daily"
        with (
            patch.object(scan, "download_prices", side_effect=lambda *a, **k: benchmark.copy()) as fetch,
            patch.object(scan, "score_news", return_value={}) as news,
            patch.object(scan.time, "sleep"),
        ):
            report = scan.run_scan("2026-10-29 18:30:00+03:00")
        self.assertEqual(report["coverage"], 20)
        self.assertEqual(len(report["universe"]), 20)
        self.assertEqual(report["session"], "2026-10-28")
        self.assertEqual(report["target_session"], "2026-10-30")
        self.assertEqual(fetch.call_count, 25)  # benchmark + 20 stocks + four macros
        self.assertEqual(news.call_count, 1)
        self.assertNotIn("probability", report)

    def test_benchmark_failure_produces_failure_report(self):
        with patch.object(scan, "download_prices", side_effect=ValueError("no data")):
            report = scan.run_scan("2026-10-29 18:30:00+03:00")
        self.assertEqual(report["status"], "unavailable")
        self.assertIn("aday üretilmedi", radar.format_report(report))

    def test_telegram_requires_delivery_receipt(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        response.__enter__.return_value = response
        with (
            patch.dict(scan.os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_CHAT_ID": "test-chat"}),
            patch.object(scan.requests, "post", return_value=response) as post,
        ):
            receipt = scan.send_report("test report")
        self.assertEqual(receipt, {"status": "sent", "message_id": 123})
        self.assertEqual(post.call_count, 1)

    def test_telegram_exception_does_not_log_token(self):
        with (
            patch.dict(scan.os.environ, {"TELEGRAM_BOT_TOKEN": "SECRET", "TELEGRAM_CHAT_ID": "test"}),
            patch.object(scan.requests, "post", side_effect=requests.Timeout("https://api.telegram.org/botSECRET")),
            self.assertLogs(scan.LOG, level="WARNING") as logs,
        ):
            result = scan.send_report("test")
        self.assertEqual(result["status"], "unknown")
        self.assertNotIn("SECRET", " ".join(logs.output))

    def test_windows_console_and_dry_run_do_not_send(self):
        report = sample_report()
        output = io.BytesIO()
        stream = io.TextIOWrapper(output, encoding="cp1252")
        with (
            patch.object(scan.sys, "argv", ["daily_scan.py", "--dry-run"]),
            patch.object(scan.sys, "stdout", stream),
            patch.object(scan, "load_local_environment"),
            patch.object(scan, "run_scan", return_value=report),
            patch.object(scan.Path, "mkdir"),
            patch.object(scan.Path, "write_text"),
            patch.object(scan, "send_report") as sender,
        ):
            code = scan.main()
            stream.flush()
            self.assertEqual(stream.encoding, "utf-8")
            self.assertIn("Ertesi Seans Radarı", output.getvalue().decode("utf-8"))
        sender.assert_not_called()
        self.assertEqual(code, 0)


class DownloadTests(unittest.TestCase):
    def test_consistent_adjusted_history_preserves_raw_reference_scale(self):
        frame = prices()
        frame["Adj Close"] = frame.Close*.5
        day = frame.index[-1]
        yahoo = SimpleNamespace(download=Mock(return_value=frame))
        with patch.dict("sys.modules", {"yfinance": yahoo}):
            result = scan.download_prices("THYAO.IS", day)
        np.testing.assert_allclose(result.Close, frame.Close*.5)
        np.testing.assert_allclose(result.AdjustmentFactor, .5)
        np.testing.assert_allclose(result.TradedValue, frame.Close*frame.Volume)

    def test_uncertain_proxy_adjustment_factor_is_rejected(self):
        frame = prices()
        frame["Adj Close"] = frame.Close*.5
        day = frame.index[-1]+pd.Timedelta(days=1)
        yahoo = SimpleNamespace(download=Mock(return_value=frame))
        with (
            patch.dict("sys.modules", {"yfinance": yahoo}),
            patch.object(scan, "fetch_t0") as fallback,
            self.assertRaisesRegex(ValueError, "katsayısı doğrulanamadı"),
        ):
            scan.download_prices("THYAO.IS", day, session_close=day.tz_localize(scan.TRT))
        fallback.assert_not_called()

    def test_missing_bar_uses_existing_exact_session_fallback(self):
        frame = prices()
        frame["Adj Close"] = frame.Close
        day = frame.index[-1]+pd.Timedelta(days=1)
        bar = frame.iloc[[-1]][radar.OHLCV].copy()
        bar.index = pd.DatetimeIndex([day])
        yahoo = SimpleNamespace(download=Mock(return_value=frame))
        with (
            patch.dict("sys.modules", {"yfinance": yahoo}),
            patch.object(scan, "fetch_t0", return_value=(bar, {"ohlv_source": "yahoo_1d"})),
        ):
            result = scan.download_prices("THYAO.IS", day, session_close=day.tz_localize(scan.TRT))
        self.assertEqual(result.index[-1], day)
        self.assertEqual(result.loc[day, "PriceSource"], "yahoo_1d")


if __name__ == "__main__":
    unittest.main()
