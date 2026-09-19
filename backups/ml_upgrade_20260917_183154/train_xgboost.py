"""Reproducible purged walk-forward model selection; no test-driven tuning.

Use python train_xgboost.py. Downloads are cached for 20 hours. --refresh-data
rebuilds the snapshot. Existing production weights are only replaced when the
predeclared final-test gate passes. Test reuse is disclosed in the saved report;
future observations are still required for prospective validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import yfinance as yf
from core import training_data as data
from core.ml_contract import normalize_features, mask_macros, calibrated_probability, NORMALIZATION_VERSION

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data_cache'
MODEL_PATH = ROOT / 'bist_xgb_model.json'
LOCAL_COLS = tuple(data.LOCAL_FEATURE_COLUMNS)
RAW_COLS = tuple(data.FEATURE_COLUMNS)
CONFIGS = [
    {'name':'weighted', 'max_depth':5, 'n_estimators':300, 'min_child_weight':60, 'window':None},
    {'name':'weighted_recent', 'max_depth':5, 'n_estimators':300, 'min_child_weight':60, 'window':504},
]

def matrix(rows, scope='full'):
    return mask_macros(normalize_features(rows.loc[:, RAW_COLS], rows.Close), scope)

def class_weight(labels):
    """Estimate class balance from this tree-fitting partition only."""
    values = np.asarray(labels)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError('Labels must be a one-dimensional binary array')
    positives = int(np.count_nonzero(values == 1))
    negatives = int(np.count_nonzero(values == 0))
    if positives == 0 or negatives == 0:
        raise ValueError('Both target classes are required for class weighting')
    return negatives / positives

def validate_dataset_target(rows):
    # Never reinterpret an old three-session CSV as a five-session dataset.
    if ('Target_Definition' not in rows or rows.empty
            or not rows['Target_Definition'].eq(data.TARGET_DEFINITION).all()):
        raise ValueError('Dataset target is missing/incompatible. Rebuild from OHLCV cache; do not reuse old labeled CSVs.')
    if not rows['Target'].isin([0, 1]).all():
        raise ValueError('Dataset contains invalid labels')
    if not (rows['Target_Date'] > rows['Date']).all():
        raise ValueError('Target dates must follow feature dates')

def classifier(config, labels):
    return XGBClassifier(objective='binary:logistic', eval_metric='logloss',
        max_depth=config['max_depth'], n_estimators=config['n_estimators'],
        min_child_weight=config['min_child_weight'], learning_rate=.05,
        reg_lambda=20., reg_alpha=.5, subsample=.8, colsample_bytree=.8,
        scale_pos_weight=class_weight(labels),
        tree_method='hist', n_jobs=min(4,os.cpu_count() or 1), random_state=42)

def calibration_fit(y, p):
    # Nonnegative slope prevents a noisy calibration set from reversing ranking.
    z = np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
    def objective(ab):
        eta = ab[0]*z+ab[1]
        return float(np.mean(np.logaddexp(0, eta)-np.asarray(y)*eta)+.001*(ab[0]-1)**2)
    fit = minimize(objective, [1.,0.], method='L-BFGS-B', bounds=[(0.,10.),(-10.,10.)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError('Calibration optimization failed')
    return {'method':'sigmoid','a':float(fit.x[0]),'b':float(fit.x[1])}

def fit_candidate(rows, config, scope):
    dates = np.sort(rows.Date.unique())
    if len(dates)<180:
        raise ValueError('Insufficient fitting dates')
    cut = pd.Timestamp(dates[int(.8*len(dates))])
    fit = rows[(rows.Date<cut)&(rows.Target_Date<cut)]
    cal = rows[rows.Date>=cut]
    if config['window']:
        fd = np.sort(fit.Date.unique())
        fit = fit[fit.Date>=fd[max(0,len(fd)-config['window'])]]
    if fit.Target.nunique()!=2 or cal.Target.nunique()!=2:
        raise ValueError('Fitting and calibration require both classes')
    model = classifier(config, fit.Target)
    model.fit(matrix(fit,scope), fit.Target)
    spec = calibration_fit(cal.Target, model.predict_proba(matrix(cal,scope))[:,1])
    return model,spec,{'fit_end':str(fit.Target_Date.max().date()),'calibration_start':str(cut.date()),
        'fit_rows':len(fit),'calibration_rows':len(cal),'calibration_end':str(cal.Target_Date.max().date()),
        'scale_pos_weight':class_weight(fit.Target),'fit_positive_count':int((fit.Target==1).sum()),
        'fit_negative_count':int((fit.Target==0).sum())}

def metrics(y,p):
    return {'log_loss':float(log_loss(y,p,labels=[0,1])),
        'brier_score':float(brier_score_loss(y,p)),
        'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}

def load_dataset(refresh=False):
    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE/'yahoo'))
    def fetch(symbol, macro):
        path = CACHE/(symbol.replace('^','').replace('=','_')+'.csv')
        if not refresh and path.exists() and (datetime.now().timestamp()-path.stat().st_mtime)<20*3600:
            f=pd.read_csv(path,index_col=0,parse_dates=True)
            return data.validate_macro(f,symbol) if macro else data.validate_equity(f,symbol)
        f=data.fetch_history(symbol,macro=macro)
        f.to_csv(path)
        return f
    macros={name:fetch(symbol,True) for name,symbol in data.MACRO_TICKERS.items()}
    equities, skipped={},{}
    for symbol in data.EQUITY_TICKERS:
        try:
            equities[symbol]=fetch(symbol,False)
        except Exception as exc:
            skipped[symbol]=str(exc)
    if len(equities)<data.MIN_SUCCESSFUL_TICKERS:
        raise RuntimeError('Too few equities; model unchanged')
    sessions=data.infer_bist_sessions(equities)
    macro_features=data.build_macro_features(macros,sessions)
    batches,quality=[],{}
    for symbol,history in equities.items():
        try:
            rows, info=data.make_training_rows(symbol,history,sessions,macro_features,20_000_000.)
            batches.append(rows); quality[symbol]=info
        except Exception as exc:
            skipped[symbol]=str(exc)
    if len(batches)<data.MIN_SUCCESSFUL_TICKERS:
        raise RuntimeError('Too few eligible equities; model unchanged')
    rows=pd.concat(batches,ignore_index=True).sort_values(['Date','Ticker']).reset_index(drop=True)
    if rows.duplicated(['Date','Ticker']).any():
        raise ValueError('Duplicate ticker dates')
    return rows,{'quality':quality,'skipped':skipped}

def select_candidate(dev):
    dates=np.sort(dev.Date.unique())
    # Three expanding folds, with all tickers sharing date boundaries.
    boundaries=[int(len(dates)*f) for f in (.50,.6667,.8333,1.)]
    scores={}
    for fold,(a,b) in enumerate(zip(boundaries[:-1],boundaries[1:]),1):
        start=pd.Timestamp(dates[a])
        end=pd.Timestamp(dates[b]) if b<len(dates) else None
        train=dev[(dev.Date<start)&(dev.Target_Date<start)]
        valid=dev[dev.Date>=start]
        if end is not None:
            valid=valid[(valid.Date<end)&(valid.Target_Date<end)]
        for config in CONFIGS:
            for scope in ('full','local'):
                model,spec,_=fit_candidate(train,config,scope)
                raw=model.predict_proba(matrix(valid,scope))[:,1]
                for method in ('identity','sigmoid'):
                    p=calibrated_probability(raw,spec if method=='sigmoid' else {'method':'identity'})
                    key=f"{config['name']}:{scope}:{method}"
                    scores.setdefault(key,[]).append(metrics(valid.Target,p)['log_loss'])
        print('Completed development fold',fold,flush=True)
    ranked=sorted(scores,key=lambda k:np.mean(scores[k]))
    selected=ranked[0]
    name,scope,method=selected.split(':')
    return next(c for c in CONFIGS if c['name']==name),scope,method,{
        'selected':selected,'fold_log_losses':scores,'ranking':ranked}

def block_interval(y,p,baseline,dates):
    # Resample blocks of whole dates, preserving cross-sectional dependence.
    def loss(q):
        q=np.clip(np.asarray(q),1e-8,1-1e-8)
        return -(np.asarray(y)*np.log(q)+(1-np.asarray(y))*np.log1p(-q))
    f=pd.DataFrame({'date':np.asarray(dates),'gain':loss(baseline)-loss(p)})
    daily=f.groupby('date').gain.mean().to_numpy()
    rng=np.random.default_rng(42); estimates=[]; n=len(daily)
    for _ in range(1000):
        starts=rng.integers(0,n,size=int(np.ceil(n/5)))
        indexes=np.concatenate([(s+np.arange(5))%n for s in starts])[:n]
        estimates.append(float(daily[indexes].mean()))
    return {'mean_daily_logloss_improvement':float(daily.mean()),
        'block_bootstrap_95pct':np.quantile(estimates,[.025,.975]).tolist(),
        'block_sessions':5,'replicates':1000}

def evaluate(rows):
    validate_dataset_target(rows)
    dates=np.sort(rows.Date.unique()); split=pd.Timestamp(dates[int(.8*len(dates))])
    dev=rows[(rows.Date<split)&(rows.Target_Date<split)]
    test=rows[rows.Date>=split]
    config,scope,method,selection=select_candidate(dev)
    model,spec,fit_info=fit_candidate(dev,config,scope)
    if method=='identity': spec={'method':'identity'}
    raw=model.predict_proba(matrix(test,scope))[:,1]
    p=calibrated_probability(raw,spec)
    # Refit the previous recipe on this SAME dataset and split; the deployed
    # artifact was fitted on these labels and cannot provide an honest comparator.
    legacy=XGBClassifier(objective='binary:logistic',eval_metric='logloss',n_estimators=450,
        max_depth=3,learning_rate=.03,min_child_weight=15,subsample=.85,colsample_bytree=.85,
        reg_alpha=.2,reg_lambda=6.,tree_method='hist',n_jobs=4,random_state=42)
    legacy.fit(dev.loc[:,RAW_COLS].astype(np.float32),dev.Target)
    oldp=legacy.predict_proba(test.loc[:,RAW_COLS].astype(np.float32))[:,1]
    baselines={'frequency':np.full(len(test),dev.Target.mean())}
    recent_dates=np.sort(dev.Date.unique())[-126:]
    baselines['recent_frequency']=np.full(len(test),dev[dev.Date.isin(recent_dates)].Target.mean())
    ticker=dev.groupby('Ticker').Target.agg(['sum','count'])
    rate=(ticker['sum']+50*dev.Target.mean())/(ticker['count']+50)
    baselines['ticker_frequency']=test.Ticker.map(rate).fillna(dev.Target.mean()).to_numpy()
    baseline_metrics={k:metrics(test.Target,v) for k,v in baselines.items()}
    best=min(baseline_metrics,key=lambda k:baseline_metrics[k]['log_loss'])
    candidate=metrics(test.Target,p); oldmetrics=metrics(test.Target,oldp)
    interval=block_interval(test.Target,p,baselines[best],test.Date)
    # A constant calibrated prediction may improve loss while providing no
    # stock-ranking information. Never promote that as a learned advantage.
    discrimination_ok=(float(np.std(p))>1e-4 and candidate['roc_auc'] is not None
                       and candidate['roc_auc']>0.5)
    passes=(discrimination_ok and candidate['log_loss']<oldmetrics['log_loss'] and
        all(candidate['log_loss']<m['log_loss'] and candidate['brier_score']<m['brier_score'] for m in baseline_metrics.values())
        and interval['block_bootstrap_95pct'][0]>0)
    report={**candidate,'target_definition':data.TARGET_DEFINITION,
        'horizon_sessions':data.HORIZON,'return_threshold':data.RETURN_THRESHOLD,
        'baseline_log_loss':baseline_metrics['frequency']['log_loss'],
        'baseline_brier_score':baseline_metrics['frequency']['brier_score'],
        'split_date':str(split.date()),'training_rows':len(dev),'validation_rows':len(test),
        'training_positive_rate':float(dev.Target.mean()),'validation_positive_rate':float(test.Target.mean()),
        'selection':selection,'selected_config':config,'feature_scope':scope,'calibration':spec,
        'fit_details':fit_info,'previous_recipe':oldmetrics,'baselines':baseline_metrics,
        'uncertainty':interval,'promotion_passed':bool(passes),
        'discrimination_passed':bool(discrimination_ok),'prediction_std':float(np.std(p)),
        'notes':'Candidate selection uses development folds only. This historical test overlaps an earlier inspected report; it is not prospective proof. No test-driven second search is performed. Fixed universe survivorship bias remains.'}
    predictions=test[['Date','Ticker','Target']].copy()
    predictions['candidate']=p; predictions['previous_recipe']=oldp
    for k,v in baselines.items():predictions[k]=v
    return report,predictions

def save_candidate(rows,report,quality,output):
    config=report['selected_config']; scope=report['feature_scope']
    # Keep the final chronological calibration tail separate from tree fitting.
    model,spec,fit_info=fit_candidate(rows,config,scope)
    if report['calibration']['method']=='identity':spec={'method':'identity'}
    cols=list(matrix(rows.head(1),scope).columns)
    model.get_booster().set_attr(feature_columns=json.dumps(cols),
        feature_contract_version=NORMALIZATION_VERSION+'+macro-native-returns-lag1-v1',
        feature_transform=NORMALIZATION_VERSION,feature_scope=scope,
        probability_calibration=json.dumps(spec),target_definition=data.TARGET_DEFINITION,
        target_horizon_sessions=str(data.HORIZON),target_return_threshold=str(data.RETURN_THRESHOLD),
        scale_pos_weight=str(fit_info['scale_pos_weight']),
        price_basis='unadjusted',macro_tickers=json.dumps(data.MACRO_TICKERS),
        macro_timing_policy=json.dumps(data.MACRO_TIMING_POLICY),
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_tickers=json.dumps(sorted(rows.Ticker.unique().tolist())),
        last_feature_date=str(rows.Date.max().date()),last_label_date=str(rows.Target_Date.max().date()),
        validation_report=json.dumps(report),data_quality_report=json.dumps(quality),
        final_fit_details=json.dumps(fit_info))
    model.save_model(str(output))
    restored=XGBClassifier(); restored.load_model(str(output))
    x=matrix(rows.tail(32),scope)
    np.testing.assert_allclose(model.predict_proba(x),restored.predict_proba(x),rtol=1e-6,atol=1e-7)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data',action='store_true')
    parser.add_argument('--dataset',type=Path,help='Reuse a locally generated research CSV')
    parser.add_argument('--no-promote',action='store_true',help='Evaluate/save candidate without replacing production')
    args=parser.parse_args()
    if args.dataset:
        rows=pd.read_csv(args.dataset,parse_dates=['Date','Target_Date'])
        quality={'source_dataset':str(args.dataset.resolve())}
    else:rows,quality=load_dataset(args.refresh_data)
    report,predictions=evaluate(rows)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'training_reports'/stamp; out.mkdir(parents=True)
    dataset_bytes=rows.to_csv(index=False).encode()
    report['dataset_sha256']=hashlib.sha256(dataset_bytes).hexdigest()
    report['tickers']=sorted(rows.Ticker.unique().tolist())
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    predictions.to_csv(out/'test_predictions.csv',index=False)
    rows.to_csv(out/'dataset.csv',index=False)
    candidate=out/'bist_xgb_candidate.json'
    save_candidate(rows,report,quality,candidate)
    if report['promotion_passed'] and not args.no_promote:
        if MODEL_PATH.exists():shutil.copy2(MODEL_PATH,out/'previous_model.json')
        fd,tmp=tempfile.mkstemp(suffix='.json',dir=ROOT); os.close(fd)
        try:
            shutil.copyfile(candidate,tmp); os.replace(tmp,MODEL_PATH)
        finally:Path(tmp).unlink(missing_ok=True)
        print('PROMOTED',MODEL_PATH)
    else:print('PRODUCTION UNCHANGED: candidate not promoted')
    print(json.dumps({k:report[k] for k in ('log_loss','baseline_log_loss','previous_recipe','promotion_passed','uncertainty')},indent=2))
    print('REPORT',out,flush=True)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    main()
