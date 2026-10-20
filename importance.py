"""Feature importance on held-out data: the 2010 temporal-validation cohort.

For each model, the temporal-validation model of benchmark.run is refitted (configuration selected on development
data, fitted on the full imputed development cohort) and evaluated on the imputed 2010 cohort, which was not used
for fitting or selection.

- Permutation importance: for each of the 60 predictors, the drop in AUPRC when that predictor is randomly permuted
  in the 2010 data (one-hot groups and functional items with their "don't do" indicator are permuted jointly),
  averaged over N_PERMUTATIONS permutations. The model is not refitted. 95% CIs: household cluster bootstrap of the
  importance, recomputed on each bootstrap sample from the stored predictions.
- SHAP: exact SHAP values for the models where they can be computed exactly: LightGBM (pred_contrib) and CatBoost
  (ShapValues) on the log-odds scale; the decision tree (tree SHAP, shap.TreeExplainer, predicted probability);
  logistic regression (linear SHAP: coefficient x (value - development mean)); and Gaussian naive Bayes, whose
  log-odds are a sum of one term per input, g_j(x_j) = log N(x_j; class 1) - log N(x_j; class 0), so that its exact
  SHAP value is g_j(x_j) minus the development mean of g_j. Values are summed within each predictor group; importance
  is the mean absolute SHAP value. Rank agreement with permutation importance is reported (Spearman).

Importance describes how much a model's discrimination depends on a predictor given the other predictors; it does
not indicate a causal or modifiable effect, and correlated predictors share importance.

Usage: python importance.py [--horizon 6] [--models LR LightGBM ...]
"""
import argparse
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

import benchmark
import evaluation

N_PERMUTATIONS = 5
N_BOOT = 500
SHAP_MODELS = ('LR', 'GNB', 'DT', 'LightGBM', 'CatBoost')


def predictor_groups(predictors):
    """Variable name -> model-input columns (60 groups for the 94 inputs)."""
    groups = {}
    for name, cols in benchmark._category_groups(predictors).items():
        groups[name[:-len('_item')] if name.endswith('_item') else name] = cols
    grouped = {c for cols in groups.values() for c in cols}
    for c in predictors:
        if c not in grouped:
            groups[c] = [c]
    return groups


def fit_temporal_model(model_name, horizon):
    cls, grid, fixed = benchmark.model_specs()[model_name]
    sel = evaluation.load(horizon, model_name, 'selection')['temporal']
    dev, ext, P = benchmark.load_cohorts(horizon)
    e = pickle.load(open(benchmark._cache_path(horizon, 'external'), 'rb'))
    model = cls(**{**fixed, **sel['params']})
    model.fit(e['X_dev'].astype(np.float32), pd.Series(dev['y'].to_numpy()))
    return model, e['X_dev'], e['X_ext'].astype(np.float32), ext['y'].to_numpy(), ext['hhidpn'].to_numpy(), P


def _predict(model, X):
    return np.asarray(model.predict_proba(X))[:, 1]


def permutation_importance(model_name, horizon, n_perm=N_PERMUTATIONS, n_boot=N_BOOT, seed=2026):
    model, X_dev, X_ext, y, hhidpn, P = fit_temporal_model(model_name, horizon)
    groups = predictor_groups(P)
    base = _predict(model, X_ext)
    saved = evaluation.load(horizon, model_name, 'temporal')['p'].to_numpy()
    rng = np.random.default_rng(seed)
    permuted = {}
    t0 = time.time()
    for name, cols in groups.items():
        permuted[name] = []
        for _ in range(n_perm):
            Xp = X_ext.copy()
            Xp[cols] = X_ext[cols].to_numpy()[rng.permutation(len(Xp))]
            permuted[name].append(_predict(model, Xp))
    seconds = time.time() - t0
    base_auprc = average_precision_score(y, base)
    point = {g: base_auprc - np.mean([average_precision_score(y, p) for p in ps]) for g, ps in permuted.items()}
    boot = {g: [] for g in groups}
    for idx in evaluation.cluster_bootstrap_indices(hhidpn, n_boot, seed=seed):
        yb = y[idx]
        b0 = average_precision_score(yb, base[idx])
        for g, ps in permuted.items():
            boot[g].append(b0 - np.mean([average_precision_score(yb, p[idx]) for p in ps]))
    rows = []
    for g in groups:
        lo, hi = np.percentile(boot[g], [2.5, 97.5])
        rows.append({'horizon': horizon, 'model': model_name, 'variable': g, 'importance_AUPRC_drop': point[g],
                     'ci_low': lo, 'ci_high': hi, 'n_inputs': len(groups[g])})
    df = pd.DataFrame(rows)
    df['rank'] = df['importance_AUPRC_drop'].rank(ascending=False, method='min').astype(int)
    meta = {'horizon': horizon, 'model': model_name, 'base_AUPRC': base_auprc, 'permutation_seconds': seconds,
            'max_abs_diff_vs_saved_predictions': float(np.max(np.abs(base - saved)))}
    return df.sort_values('rank'), meta, model, X_dev, X_ext, groups


def shap_importance(model_name, model, X_dev, X_ext, groups, horizon):
    if model_name == 'LightGBM':
        phi = model.predict(X_ext, pred_contrib=True)[:, :-1]
    elif model_name == 'CatBoost':
        from catboost import Pool
        phi = model.get_feature_importance(Pool(X_ext), type='ShapValues')[:, :-1]
    elif model_name == 'LR':
        phi = (X_ext.to_numpy() - X_dev.to_numpy(dtype=np.float32).mean(axis=0)) * model.coef_[0]
    elif model_name == 'GNB':
        g_ext, g_dev = _gnb_terms(model, X_ext.to_numpy()), _gnb_terms(model, X_dev.to_numpy(dtype=np.float32))
        phi = g_ext - g_dev.mean(axis=0)
    elif model_name == 'DT':
        import shap
        phi = np.asarray(shap.TreeExplainer(model).shap_values(X_ext.to_numpy(), check_additivity=True))
        phi = phi[..., 1] if phi.ndim == 3 else phi[1]  # class-1 probability
    else:
        raise ValueError(model_name)
    phi = pd.DataFrame(phi, columns=X_ext.columns)
    rows = [{'horizon': horizon, 'model': model_name, 'variable': g,
             'mean_abs_SHAP': float(np.abs(phi[cols].sum(axis=1)).mean())} for g, cols in groups.items()]
    df = pd.DataFrame(rows)
    df['rank'] = df['mean_abs_SHAP'].rank(ascending=False, method='min').astype(int)
    return df.sort_values('rank')


def _gnb_terms(model, X):
    """Per-input log-likelihood ratio terms of Gaussian naive Bayes; their sum plus the log prior ratio is the
    log-odds of class 1."""
    X = np.asarray(X, float)
    mu, var = model.theta_, model.var_
    logpdf = lambda c: -0.5 * (np.log(2 * np.pi * var[c]) + (X - mu[c]) ** 2 / var[c])
    return logpdf(1) - logpdf(0)


def run(horizon, models):
    warnings.filterwarnings('ignore')
    os.makedirs(evaluation.TABLES, exist_ok=True)
    perm_rows, shap_rows, agree_rows, meta_rows = [], [], [], []
    for m in models:
        perm, meta, model, X_dev, X_ext, groups = permutation_importance(m, horizon)
        perm_rows.append(perm); meta_rows.append(meta)
        print(f"{m} h{horizon}: permutation done in {meta['permutation_seconds']:.0f}s; "
              f"top 5: {list(perm['variable'][:5])}", flush=True)
        if m in SHAP_MODELS:
            sh = shap_importance(m, model, X_dev, X_ext, groups, horizon)
            shap_rows.append(sh)
            merged = perm.merge(sh, on=['horizon', 'model', 'variable'], suffixes=('_perm', '_shap'))
            rho = stats.spearmanr(merged['importance_AUPRC_drop'], merged['mean_abs_SHAP']).correlation
            top10 = len(set(perm['variable'][:10]) & set(sh['variable'][:10]))
            agree_rows.append({'horizon': horizon, 'model': m, 'spearman_rank_correlation': rho,
                               'shared_top10': top10})
    suffix = f'h{horizon}'
    _save_merged(pd.concat(perm_rows), f'importance_permutation_{suffix}.csv', models)
    _save_merged(pd.DataFrame(meta_rows), f'importance_meta_{suffix}.csv', models)
    if shap_rows:
        _save_merged(pd.concat(shap_rows), f'importance_shap_{suffix}.csv', models)
        _save_merged(pd.DataFrame(agree_rows), f'importance_rank_agreement_{suffix}.csv', models)


def _save_merged(new, name, models):
    """Write a table, replacing the rows of the models just run and keeping rows of other models."""
    path = os.path.join(evaluation.TABLES, name)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old['model'].isin(models)], new], ignore_index=True)
    order = {m: i for i, m in enumerate(evaluation.MODELS)}
    new = new.sort_values(['model'] + (['rank'] if 'rank' in new else []), key=lambda c: c.map(order) if c.name == 'model' else c)
    new.to_csv(path, index=False)


if __name__ == '__main__':
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count()))
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=6, choices=[6, 8])
    ap.add_argument('--models', nargs='*', default=evaluation.MODELS)
    args = ap.parse_args()
    run(args.horizon, args.models)
