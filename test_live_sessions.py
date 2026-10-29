"""Offline regression tests for live session selection and missing price data.

Run with: python -m unittest test_live_sessions -v
No market, NLP, or Telegram network requests are made by these tests.
"""

import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import live_inference as live


class RecordingModel:
    predict_proba = None

    def __init__(self):
        self.inputs = []
        self.booster = SimpleNamespace(
            feature_names=["Close", "news_sentiment"],
            attr=lambda name: None,
        )

    def get_booster(self):
        return self.booster

    def predict(self, frame):
        self.inputs.append(frame.copy())
        return np.array([1])


class CompletedSessionTests(unittest.TestCase):
    def test_calendar_boundaries(self):
        # Includes public holidays, religious holidays, early closes, weekends,
        # pre-open, intraday, UTC conversion, and crossing a year boundary.
        cases = [
            ("2026-10-29 02:58:55+03:00", "2026-10-28", "12:40"),
            ("2026-10-28 12:39:59+03:00", "2026-10-27", "18:10"),
            ("2026-10-28 12:40:00+03:00", "2026-10-28", "12:40"),
            ("2026-10-28 18:35:00+03:00", "2026-10-28", "12:40"),
            ("2026-10-30 08:00:00+03:00", "2026-10-28", "12:40"),
            ("2026-10-30 14:00:00+03:00", "2026-10-28", "12:40"),
            ("2026-10-30 18:09:59+03:00", "2026-10-28", "12:40"),
            ("2026-10-30 18:10:00+03:00", "2026-10-30", "18:10"),
            ("2026-10-31 10:00:00+03:00", "2026-10-30", "18:10"),
            ("2026-11-01 18:35:00+03:00", "2026-10-30", "18:10"),
            ("2026-11-02 00:01:00+03:00", "2026-10-30", "18:10"),
            ("2026-10-30 15:10:00+00:00", "2026-10-30", "18:10"),
            ("2026-01-01 18:35:00+03:00", "2025-12-31", "18:10"),
            ("2026-03-20 18:35:00+03:00", "2026-03-19", "12:40"),
            ("2026-05-27 18:35:00+03:00", "2026-05-26", "12:40"),
        ]
        for as_of, expected_day, expected_close in cases:
            with self.subTest(as_of=as_of):
                session, close = live.last_completed_bist_session(as_of)
                self.assertEqual(str(session.date()), expected_day)
                self.assertEqual(close.strftime("%H:%M"), expected_close)
                self.assertEqual(str(close.tz), live.TRT)
                self.assertIsNone(session.tzinfo)
                self.assertLessEqual(close, pd.Timestamp(as_of))

    def test_invalid_timestamps_rejected(self):
        for as_of in ["2026-10-29 12:00:00", pd.NaT]:
            with self.subTest(as_of=as_of), self.assertRaises(ValueError):
                live.last_completed_bist_session(as_of)


class PriceFallbackTests(unittest.TestCase):
    def test_daily_fallback_is_preserved(self):
        close = pd.Timestamp("2026-10-28 12:40:00+03:00")
        daily = dict(Open=100.0, High=105.0, Low=99.0, Close=103.0, Volume=10.0)
        with (
            patch.object(live, "is_close", side_effect=RuntimeError("unavailable")),
            patch.object(live, "minute_bar", side_effect=ValueError("no minutes")) as minute,
            patch.object(live, "daily_bar", return_value=daily) as fetch_daily,
        ):
            bar, provenance = live.fetch_t0(
                "THYAO", close, session_close=close
            )
        fetch_daily.assert_called_once_with("THYAO.IS", date(2026, 10, 28))
        minute.assert_called_once_with(
            "THYAO.IS", date(2026, 10, 28), session_close=close
        )
        self.assertEqual(bar.index[0], pd.Timestamp("2026-10-28"))
        self.assertEqual(bar.iloc[0].to_dict(), daily)
        self.assertTrue(provenance["daily_fallback_used"])

    def test_all_sources_missing_raise_distinct_exception(self):
        close = pd.Timestamp("2026-10-28 12:40:00+03:00")
        with (
            patch.object(live, "is_close", side_effect=RuntimeError("unavailable")),
            patch.object(live, "minute_bar", side_effect=ValueError("no minutes")),
            patch.object(live, "daily_bar", side_effect=ValueError("no daily bar")),
            self.assertRaises(live.PriceDataUnavailable),
        ):
            live.fetch_t0("THYAO", close, session_close=close)

    def test_half_day_minute_bar_uses_early_close(self):
        close = pd.Timestamp("2026-10-28 12:40:00+03:00")
        index = pd.date_range(
            "2026-10-28 10:00", "2026-10-28 12:39", freq="min", tz=live.TRT
        )
        minutes = pd.DataFrame(
            dict(Open=100.0, High=101.0, Low=99.0, Close=100.5, Volume=1.0),
            index=index,
        )
        yahoo = SimpleNamespace(download=Mock(return_value=minutes))
        with patch.dict("sys.modules", {"yfinance": yahoo}):
            bar, last_minute = live.minute_bar(
                "THYAO.IS", date(2026, 10, 28), session_close=close
            )
        self.assertEqual(bar["Volume"], len(index))
        self.assertEqual(pd.Timestamp(last_minute), index[-1])


class LiveSessionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame(
            {column: [100.0, 110.0, 999.0] for column in live.OHLCV},
            index=pd.to_datetime(["2026-10-27", "2026-10-28", "2026-10-29"]),
        )
        self.bar = pd.DataFrame(
            {column: [200.0] for column in live.OHLCV},
            index=pd.to_datetime(["2026-10-28"]),
        )

    def test_selected_date_propagates_through_features_and_sentiment(self):
        model = RecordingModel()
        features = Mock(side_effect=lambda bars: bars)
        sentiment = Mock(return_value=0.25)
        as_of = "2026-10-29 02:58:55+03:00"
        close = pd.Timestamp("2026-10-28 12:40:00+03:00")
        with patch.object(
            live, "fetch_t0", return_value=(self.bar.copy(), {"close_source": "test"})
        ) as fetch:
            result = live.run_live(
                "THYAO", self.history, model, features, sentiment, as_of=as_of
            )
        fetch.assert_called_once_with(
            "THYAO", close, require_ohlcv=True, session_close=close
        )
        sentiment.assert_called_once_with("THYAO", close)
        input_bars = features.call_args.args[0]
        self.assertEqual(input_bars.index.max(), pd.Timestamp("2026-10-28"))
        self.assertTrue(input_bars.index.is_unique)
        self.assertEqual(input_bars.loc["2026-10-28", "Close"], 200.0)
        self.assertEqual(result["date"], "2026-10-28")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["provenance"]["used_previous_session"])
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.inputs[0].index[0], pd.Timestamp("2026-10-28"))
        message = live.format_telegram_message(result)
        self.assertIn("Trading date: 2026-10-28", message)
        self.assertIn("Session close (TRT): 2026-10-28T12:40:00+03:00", message)

    def test_price_outage_skips_features_sentiment_and_prediction(self):
        model = RecordingModel()
        features = Mock()
        sentiment = Mock()
        with patch.object(
            live, "fetch_t0", side_effect=live.PriceDataUnavailable("sources unavailable")
        ):
            result = live.run_live(
                "THYAO", self.history, model, features, sentiment,
                as_of="2026-10-29 02:58:55+03:00",
            )
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "price_data_unavailable")
        self.assertEqual(result["date"], "2026-10-28")
        self.assertIsNone(result["prediction"])
        features.assert_not_called()
        sentiment.assert_not_called()
        self.assertEqual(model.inputs, [])
        message = live.format_telegram_message(result)
        self.assertIn("SKIPPED", message)
        self.assertIn("tahmin üretilmedi", message)
        self.assertIn("2026-10-28", message)

    def test_main_still_notifies_on_skip(self):
        skipped = {"status": "skipped", "date": "2026-10-28"}
        model = Mock()
        factory = Mock(return_value=model)
        with (
            patch("sys.argv", ["live_inference.py"]),
            patch.object(live, "load_local_environment"),
            patch.object(live, "load_history_csv", return_value=self.history),
            patch.object(live, "resolve", side_effect=[factory, Mock(), Mock()]),
            patch.object(live, "run_live", return_value=skipped) as run,
            patch.object(live, "send_telegram_notification", return_value={"status": "sent"}) as notify,
            patch("builtins.print"),
        ):
            live.main()
        self.assertIsNone(run.call_args.kwargs["as_of"])
        notify.assert_called_once_with(skipped)


class PredictionContractTests(unittest.TestCase):
    def test_imputation_and_feature_alignment_are_preserved(self):
        model = RecordingModel()
        frame = pd.DataFrame({
            "news_sentiment": [np.nan, np.inf, 0.25],
            "unused": [9, 9, 9],
            "Close": [100.0, -np.inf, np.nan],
        })
        live.align_and_predict(model, frame)
        expected = pd.DataFrame({
            "Close": [100.0, 100.0, 100.0],
            "news_sentiment": [0.25, 0.25, 0.25],
        })
        pd.testing.assert_frame_equal(model.inputs[0], expected)
        self.assertTrue(np.isinf(frame.loc[1, "news_sentiment"]))

    def test_missing_column_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "features are missing"):
            live.align_and_predict(RecordingModel(), pd.DataFrame({"Close": [100.0]}))

    def test_duplicate_columns_still_rejected(self):
        frame = pd.DataFrame([[100.0, 100.0]], columns=["Close", "Close"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            live.align_and_predict(RecordingModel(), frame)


if __name__ == "__main__":
    unittest.main()
