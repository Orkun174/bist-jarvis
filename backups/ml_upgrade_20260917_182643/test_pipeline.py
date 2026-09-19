"""Focused offline regression tests for normalization, timing and model loading."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from core import ai_analyzer as ai
from core import training_data as data
from core.ml_contract import normalize_features, calibrated_probability, RENAMES, NORMALIZATION_VERSION
import train_xgboost as train

ROOT=Path(__file__).resolve().parent

class PipelineTests(unittest.TestCase):
    def setUp(self):
        dates=pd.bdate_range('2025-01-01',periods=90)
        self.macro=pd.DataFrame({'Close':100+np.arange(90)+np.sin(np.arange(90))},index=dates)
        self.date=dates[-3]
        self.values={'RSI':55.,'MACD':2.,'MACD_Signal':1.5,'EMA_20':98.,'EMA_50':95.,
            'ATR_14':3.,'ADX_14':23.,'CMF_20':.1,'StochRSI_K':65.,'StochRSI_D':60.}
        self.snapshot={ai.SNAPSHOT_KEYS[k]:v for k,v in self.values.items()}
        self.snapshot.update(price=100.,bar_time=self.date.isoformat())

    def test_price_scale_invariance(self):
        raw=pd.DataFrame([self.values]); other=raw.copy()
        for c in RENAMES:other[c]*=20
        pd.testing.assert_frame_equal(normalize_features(raw,[100]),normalize_features(other,[2000]))

    def test_training_live_parity_and_exact_order(self):
        macro=data.build_macro_features({n:self.macro for n in ai.MACRO_TICKERS},pd.DatetimeIndex([self.date]))
        local=ai.build_feature_matrix(pd.DataFrame([self.values],index=macro.index))
        expected=normalize_features(local.join(macro),[100.])
        names=list(reversed(expected.columns))
        with patch.object(ai,'_fetch_macro_history',return_value=(self.macro,'test')):
            actual,_=ai.build_inference_matrix(self.snapshot,names,NORMALIZATION_VERSION)
        pd.testing.assert_frame_equal(actual,expected.loc[:,names])

    def test_future_macro_cannot_change_input(self):
        a,_=ai._macro_row('USDTRY',self.macro,self.date)
        altered=self.macro.copy(); altered.loc[altered.index>=self.date,'Close']*=10
        b,_=ai._macro_row('USDTRY',altered,self.date)
        pd.testing.assert_frame_equal(a,b)

    def test_calibration_validation_and_identity(self):
        np.testing.assert_allclose(calibrated_probability([.2,.8],{'method':'identity'}),[.2,.8])
        np.testing.assert_allclose(calibrated_probability([.2,.8],{'method':'sigmoid','a':0,'b':0}),[.5,.5])
        with self.assertRaises(ValueError):calibrated_probability([.5],{'method':'sigmoid','a':-1,'b':0})

    def test_both_model_generations_load(self):
        paths=[ROOT/'bist_xgb_model.json',*sorted((ROOT/'training_reports').glob('*/bist_xgb_candidate.json'))]
        self.assertGreater(len(paths),1)
        for p in paths:
            model=ai._load_model_cached(str(p),p.stat().st_mtime_ns,p.stat().st_size)
            self.assertEqual(model.n_features_in_,24)

    def test_missing_day_splits_and_extra_date_does_not_drop_ticker(self):
        sessions=pd.bdate_range('2025-01-01',periods=100)
        ix=sessions.delete(60).union(pd.DatetimeIndex(['2025-01-04']))
        frame=pd.DataFrame({'Close':100.},index=ix)
        blocks=data.split_contiguous_history(frame,sessions)
        self.assertEqual([len(b) for b in blocks],[60,39])
        self.assertNotIn(pd.Timestamp('2025-01-04'),blocks[0].index)

    def test_calibration_purge(self):
        rng=np.random.default_rng(3); dates=pd.bdate_range('2020-01-01',periods=250)
        rows=pd.DataFrame({c:rng.normal(size=247) for c in train.RAW_COLS})
        rows['Close']=100.; rows['EMA_20']=99.; rows['EMA_50']=98.
        rows['Date']=dates[:-3]; rows['Target_Date']=dates[3:]
        rows['Target']=rng.integers(0,2,247)
        _,_,info=train.fit_candidate(rows,{'max_depth':1,'n_estimators':5,'min_child_weight':10,'window':None},'full')
        self.assertLess(info['fit_end'],info['calibration_start'])

    def test_candidate_probabilities_match_inference(self):
        candidate=sorted((ROOT/'training_reports').glob('*/bist_xgb_candidate.json'))[-1]
        model=ai._load_model_cached(str(candidate),candidate.stat().st_mtime_ns,candidate.stat().st_size)
        with patch.object(ai,'_fetch_macro_history',return_value=(self.macro,'test')), patch.object(ai,'load_xgboost_model',return_value=model):
            result=ai.predict_uptrend('THYAO',self.snapshot)
            b=model.get_booster()
            x,_=ai.build_inference_matrix(self.snapshot,b.feature_names,b.attr('feature_transform'),b.attr('feature_scope'))
            expected=calibrated_probability(model.predict_proba(x)[:,1],json.loads(b.attr('probability_calibration')))[0]*100
            self.assertAlmostEqual(result['XGBoost_Uptrend_Probability_Percent'],expected)

if __name__=='__main__':unittest.main()
