"""Evaluation protocol for the dementia benchmark.

For each prediction horizon (6 or 8 years) and each of 10 repeats r:

1. The development cohort (all eligible 2000 participants with a known outcome, complete or not) is split into
   train / validation / test = 60 / 20 / 20, stratified on the outcome (same scheme and seeds as the original
   notebooks: random_state = r).
2. Imputation (miceforest, 20 iterations, predictors only) is fitted on the training split and applied to the
   validation and test splits. For the refitted model, a second imputer is fitted on train + validation and
   applied to test. Imputations are cached and shared by all models.
3. Hyperparameters: every grid configuration is fitted on train and scored by AUPRC on validation. Each repeat
   selects its own best configuration, so no test observation of any repeat influences that repeat's selection.
4. Threshold: the F1-maximizing threshold on the validation predictions of the selected configuration.
5. The selected configuration is refitted on train + validation and evaluated once on test.

Temporal validation (2010 cohort): the configuration with the best mean validation AUPRC over the 10 repeats, the
median of its 10 validation thresholds, an imputer and model fitted on the full development cohort, and the 2010
cohort imputed with that imputer. All choices use development data only.

Scoring populations: selection and thresholds use all validation participants (missing predictors imputed by the
training imputer). Test and 2010 metrics are reported for two populations from the same fitted models: 'all'
(primary; every participant) and 'complete' (sensitivity; participants with complete predictors only).
"""
import importlib.metadata
import itertools
import json
import os
import hashlib
import pickle
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
HORIZONS = {
    6: {'dev': 'data/preprocessed data/2000(00-06y)encode.csv', 'ext': 'data/2000-2006/2010(10-16y).csv'},
    8: {'dev': 'data/preprocessed data/2000(00-08y)encode.csv', 'ext': 'data/2000-2008/2010(10-18y).csv'},
}
N_REPEATS = 10
MICE_ITERATIONS = 20
NON_PREDICTORS = ('hhidpn', 'demcls', 'y')
CACHE_DIR = os.path.join(ROOT, 'results', 'cache')
RESULTS_DIR = os.path.join(ROOT, 'results')


def model_specs():
    """name -> (class, hyperparameter grid, fixed arguments). Grids follow Table 1 of the manuscript."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from pytabkit import RealMLP_TD_Classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from tabicl import TabICLClassifier
    from neuralclassifier_fixed import GATClassifier
    return {
        'LR': (LogisticRegression, {'C': [1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']},
               {'random_state': 42}),
        'GNB': (GaussianNB, {'var_smoothing': [1e-10, 1e-7, 1e-4], 'priors': [None, [0.5, 0.5], [0.4, 0.6]]}, {}),
        'DT': (DecisionTreeClassifier, {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 4]}, {'random_state': 42}),
        'LightGBM': (LGBMClassifier, {'num_leaves': [20, 40], 'max_depth': [5, 10], 'learning_rate': [0.01, 0.05, 0.1]},
                     {'verbose': -1, 'random_state': 42}),
        'CatBoost': (CatBoostClassifier, {'iterations': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [4, 6, 8]},
                     {'verbose': 0, 'random_state': 42, 'allow_writing_files': False}),
        'GAT': (GATClassifier, {'learning_rate': [1e-4, 1e-2], 'dropout_p': [0.1, 0.5]},
                {'hidden_layer_sizes': (128, 64, 32), 'max_iter': 50, 'use_residual': True, 'use_batch_norm': False,
                 'batch_size': 256, 'random_state': 42, 'cuda': True, 'reference_size': 255, 'standardize': False,
                 'max_restarts': 3}),
        'RealMLP': (RealMLP_TD_Classifier, {'batch_size': [128, 256], 'n_epochs': [50, 100]}, {'random_state': 42}),
        'TabICL': (TabICLClassifier, {'n_estimators': [16, 32], 'batch_size': [8, 16]},
                   {'use_amp': True, 'checkpoint_version': 'tabicl-classifier-v1.1-0506.ckpt', 'random_state': 42}),
    }


def _tabicl_checkpoint(version):
    """Path and SHA-256 of the cached TabICL checkpoint file actually used (Hugging Face cache)."""
    import glob
    paths = glob.glob(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub', '*TabICL*', 'snapshots',
                                   '*', version))
    if not paths:
        return {'version': version, 'path': None, 'sha256': None}
    digest = hashlib.sha256(open(paths[0], 'rb').read()).hexdigest()
    rel = paths[0].split(os.path.join('huggingface', 'hub') + os.sep)[-1].replace(os.sep, '/')
    return {'version': version, 'path': f'huggingface/hub/{rel}', 'sha256': digest}


def load_cohorts(horizon):
    dev = pd.read_csv(os.path.join(ROOT, HORIZONS[horizon]['dev']))
    ext = pd.read_csv(os.path.join(ROOT, HORIZONS[horizon]['ext']))
    predictors = [c for c in dev.columns if c not in NON_PREDICTORS]
    assert list(ext[predictors].columns) == predictors
    for d in (dev, ext):
        assert d['y'].notna().all() and ((d['ageym'] > 50) & (d['demcls'] == 0)).all()
    return dev.reset_index(drop=True), ext.reset_index(drop=True), predictors


def split_indices(y, repeat):
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=repeat)
    train, val = train_test_split(trval, test_size=0.25, stratify=y[trval], random_state=repeat)
    return np.sort(train), np.sort(val), np.sort(test)


NOMINAL_PREFIXES = ('edstg', 'cendiv', 'mstat', 'raceeth', 'lbrf')


def _category_groups(columns):
    """Dummy columns that encode one categorical variable, so they are imputed jointly.

    Nominal variables (reference category = all dummies 0) and each functional item with its "don't do" dummy
    (no difficulty / difficulty / don't do) are each imputed as a single categorical variable, which guarantees
    exactly one category per imputed participant.
    """
    groups = {pre: [c for c in columns if c.startswith(pre)] for pre in NOMINAL_PREFIXES}
    for c in columns:
        if c.endswith('dontdo'):
            groups[f'{c[:-len("dontdo")]}_item'] = [c[:-len('dontdo')], c]
    return {name: cols for name, cols in groups.items() if cols}


def _collapse(X, groups):
    X = X.copy()
    for name, cols in groups.items():
        D = X[cols].to_numpy(dtype=float)
        assert (np.nansum(D, axis=1) <= 1).all(), f'{name}: more than one category set'
        code = np.where(D.sum(axis=1) == 0, 0, np.argmax(D, axis=1) + 1).astype(float)
        code[np.isnan(D).any(axis=1)] = np.nan
        X = X.drop(columns=cols)
        X[name] = pd.Categorical(code, categories=[float(k) for k in range(len(cols) + 1)])
    return X


def _expand(X, groups, columns):
    X = X.copy()
    for name, cols in groups.items():
        code = X.pop(name).astype(float).to_numpy()
        for j, c in enumerate(cols):
            X[c] = (code == j + 1).astype(float)
    return X[columns]


def _impute(fit_X, apply_to, seed):
    """Fit miceforest on fit_X only; return (imputed fit_X, [imputed frames for apply_to]).

    Every predictor gets an imputation model (variable_schema = all columns), so a predictor that is complete in
    fit_X but missing in a later data set is still imputed. Dummy groups are imputed as one categorical variable.
    """
    import miceforest as mf
    columns = list(fit_X.columns)
    groups = _category_groups(columns)
    fit_c = _collapse(fit_X.reset_index(drop=True), groups)
    # Categorical variables use 'fast' mean matching (draw from the predicted class probabilities): rare levels
    # (e.g. "don't do" for picking up a dime) give probabilities of exactly 0 or 1, which break the default
    # log-odds nearest-neighbor matching.
    kernel = mf.ImputationKernel(fit_c, variable_schema=list(fit_c.columns),
                                 mean_match_strategy={name: 'fast' for name in groups}, random_state=seed)
    kernel.mice(MICE_ITERATIONS)
    fitted = _expand(kernel.complete_data(0), groups, columns)
    applied = [_expand(kernel.impute_new_data(_collapse(X.reset_index(drop=True), groups), random_state=seed)
                       .complete_data(0), groups, columns) for X in apply_to]
    for out, src in zip([fitted] + applied, [fit_X] + list(apply_to)):
        assert not out.isna().any().any(), 'missing values left after imputation'
        observed = src.reset_index(drop=True).notna().to_numpy()
        assert np.allclose(out.to_numpy()[observed], src.to_numpy(dtype=float)[observed]), 'observed values changed'
    return fitted, applied


def _cache_path(horizon, name):
    return os.path.join(CACHE_DIR, f'h{horizon}', f'{name}.pkl')


def build_imputations(horizon, repeats=range(N_REPEATS)):
    """Compute and cache the train-only imputations for every repeat and for temporal validation."""
    dev, ext, P = load_cohorts(horizon)
    y = dev['y'].to_numpy()
    os.makedirs(os.path.dirname(_cache_path(horizon, 'x')), exist_ok=True)
    stamp = {'imputation': 'joint-categorical-v2', 'n_dev': len(dev), 'n_ext': len(ext), 'predictors': P, 'dev_y_sum': int(y.sum())}
    for r in repeats:
        path = _cache_path(horizon, f'rep{r}')
        if os.path.exists(path) and pickle.load(open(path, 'rb'))['stamp'] == stamp:
            continue
        t0 = time.time()
        tr, va, te = split_indices(y, r)
        X = dev[P].astype(float)
        X_tr, (X_va, X_te_a) = _impute(X.iloc[tr], [X.iloc[va], X.iloc[te]], seed=42 + r)
        X_trva, (X_te_b,) = _impute(X.iloc[np.r_[tr, va]], [X.iloc[te]], seed=142 + r)
        pickle.dump({'stamp': stamp, 'train': tr, 'val': va, 'test': te, 'X_train': X_tr, 'X_val': X_va,
                     'X_test_from_train': X_te_a, 'X_trainval': X_trva, 'X_test': X_te_b,
                     'seconds': time.time() - t0}, open(path, 'wb'))
        print(f'h{horizon} repeat {r}: imputed in {time.time() - t0:.0f}s', flush=True)
    path = _cache_path(horizon, 'external')
    if not (os.path.exists(path) and pickle.load(open(path, 'rb'))['stamp'] == stamp):
        t0 = time.time()
        X_dev, (X_ext,) = _impute(dev[P].astype(float), [ext[P].astype(float)], seed=999)
        pickle.dump({'stamp': stamp, 'X_dev': X_dev, 'X_ext': X_ext, 'seconds': time.time() - t0}, open(path, 'wb'))
        print(f'h{horizon} external: imputed in {time.time() - t0:.0f}s', flush=True)


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {'n': int(len(y)), 'cases': int(y.sum()), 'AUROC': roc_auc_score(y, p), 'AUPRC': average_precision_score(y, p),
            'Accuracy': accuracy_score(y, pred), 'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred), 'F1': f1_score(y, pred), 'Brier': brier_score_loss(y, p),
            'threshold': threshold}


def _fit_predict(cls, params, X_fit, y_fit, X_pred):
    t0 = time.time()
    model = cls(**params)
    model.fit(X_fit.astype(np.float32), pd.Series(y_fit))
    p = np.asarray(model.predict_proba(X_pred.astype(np.float32)))[:, 1]
    _fit_predict.restarts = getattr(model, 'n_restarts_', 0)  # GAT restart rule (0 for other models)
    return p, time.time() - t0


POPULATIONS = ('all', 'complete')


def run(model_name, horizon, repeats=range(N_REPEATS), out_dir=None):
    cls, grid, fixed = model_specs()[model_name]
    configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    dev, ext, P = load_cohorts(horizon)
    y = dev['y'].to_numpy()
    complete = dev[P].notna().all(axis=1).to_numpy()
    out_dir = out_dir or os.path.join(RESULTS_DIR, f'h{horizon}', model_name)
    os.makedirs(out_dir, exist_ok=True)

    def score(setting, repeat, config, y_true, p, is_complete, threshold, secs):
        for pop in POPULATIONS:
            mask = is_complete if pop == 'complete' else np.ones(len(y_true), bool)
            metric_rows.append({'setting': setting, 'population': pop, 'repeat': repeat, 'config': config,
                                **metrics(y_true[mask], p[mask], threshold), 'fit_seconds': secs,
                                'restarts': _fit_predict.restarts})

    grid_rows, metric_rows, pred_rows, selected = [], [], [], []
    val_store = {}  # (repeat, config index) -> validation predictions, for the temporal-validation threshold
    for r in repeats:
        c = pickle.load(open(_cache_path(horizon, f'rep{r}'), 'rb'))
        tr, va, te = c['train'], c['val'], c['test']
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        best = (-1, None)
        for k, cfg in enumerate(configs):
            p_va, secs = _fit_predict(cls, {**fixed, **cfg}, c['X_train'], y[tr], c['X_val'])
            auprc = average_precision_score(y[va], p_va)
            val_store[(r, k)] = p_va
            grid_rows.append({'repeat': r, 'config': k, **{f'param_{n}': str(v) for n, v in cfg.items()},
                              'val_AUPRC': auprc, 'fit_seconds': secs, 'restarts': _fit_predict.restarts})
            if auprc > best[0]:
                best = (auprc, k)
        k = best[1]
        threshold = best_f1_threshold(y[va], val_store[(r, k)])
        p_te, secs = _fit_predict(cls, {**fixed, **configs[k]}, c['X_trainval'], y[np.r_[tr, va]], c['X_test'])
        score('internal', r, k, y[te], p_te, complete[te], threshold, secs)
        selected.append({'repeat': r, 'config': k, 'params': configs[k], 'val_AUPRC': best[0], 'threshold': threshold})
        pred_rows.append(pd.DataFrame({'repeat': r, 'hhidpn': dev['hhidpn'].to_numpy()[te], 'y': y[te],
                                       'p': p_te, 'complete': complete[te]}))
        m = metric_rows[-2]
        print(f'{model_name} h{horizon} repeat {r}: config {k} {configs[k]} val AUPRC {best[0]:.3f} '
              f'thr {threshold:.3f} test AUROC {m["AUROC"]:.3f} AUPRC {m["AUPRC"]:.3f}', flush=True)

    # temporal validation: every choice from development data only
    grid_df = pd.DataFrame(grid_rows)
    k_ext = int(grid_df.groupby('config')['val_AUPRC'].mean().idxmax())
    thr_ext = float(np.median([best_f1_threshold(y[pickle.load(open(_cache_path(horizon, f'rep{r}'), 'rb'))['val']],
                                                 val_store[(r, k_ext)]) for r in repeats]))
    e = pickle.load(open(_cache_path(horizon, 'external'), 'rb'))
    p_ext, secs = _fit_predict(cls, {**fixed, **configs[k_ext]}, e['X_dev'], y, e['X_ext'])
    y_ext = ext['y'].to_numpy()
    ext_complete = ext[P].notna().all(axis=1).to_numpy()
    score('temporal', -1, k_ext, y_ext, p_ext, ext_complete, thr_ext, secs)
    m = metric_rows[-2]
    print(f'{model_name} h{horizon} temporal: config {k_ext} {configs[k_ext]} thr {thr_ext:.3f} '
          f'AUROC {m["AUROC"]:.3f} AUPRC {m["AUPRC"]:.3f} (n={m["n"]})', flush=True)

    grid_df.to_csv(os.path.join(out_dir, 'validation_grid.csv'), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    pd.concat(pred_rows).to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)
    pd.DataFrame({'hhidpn': ext['hhidpn'], 'y': y_ext, 'p': p_ext, 'complete': ext_complete}).to_csv(
        os.path.join(out_dir, 'temporal_predictions.csv'), index=False)
    versions = {}
    for pkg in ['numpy', 'pandas', 'scikit-learn', 'miceforest', 'lightgbm', 'catboost', 'torch', 'torch_geometric',
                'imbalanced-learn', 'pytabkit', 'tabicl']:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    versions['python'] = platform.python_version()
    extra = {}
    if model_name == 'TabICL':
        extra['checkpoint'] = _tabicl_checkpoint(fixed['checkpoint_version'])
    json.dump({'model': model_name, 'horizon': horizon, 'fixed_params': {k: str(v) for k, v in fixed.items()}, **extra,
               'grid': {k: [str(x) for x in v] for k, v in grid.items()}, 'per_repeat': selected,
               'temporal': {'config': k_ext, 'params': configs[k_ext], 'threshold': thr_ext},
               'mice_iterations': MICE_ITERATIONS, 'versions': versions},
              open(os.path.join(out_dir, 'selection.json'), 'w'), indent=1, default=str)
    return pd.DataFrame(metric_rows)
