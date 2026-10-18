"""Descriptive tables for the analytic cohorts: participant flow, complete vs incomplete records, and the overlap
between the 2000 development and 2010 temporal-validation cohorts. Writes aggregate tables to results/tables/.

Usage: python cohort_tables.py
"""
import os

import numpy as np
import pandas as pd

import benchmark

ROOT = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(ROOT, 'data', 'preprocessed data')
OUT = os.path.join(ROOT, 'results', 'tables')
WINDOWS = {6: ('demcls00-06', 'demcls10-16'), 8: ('demcls00-08', 'demcls10-18')}


def household(hhidpn):
    return np.asarray(hhidpn) // 1000  # hhidpn = HHID * 1000 + PN


def overlap_status(temporal_ids, development_ids):
    """Per temporal-validation participant: returning participant / new participant in a development household / new."""
    t = np.asarray(temporal_ids)
    dev = set(development_ids)
    dev_hh = set(household(list(dev)))
    return np.where(np.isin(t, list(dev)), 'returning participant',
                    np.where(np.isin(household(t), list(dev_hh)), 'new participant, development household',
                             'new participant and household'))


def participant_flow():
    status = pd.read_csv(os.path.join(PRE, 'outcome_status.csv')).set_index('window')
    flow00 = pd.read_csv(os.path.join(PRE, 'flow_2000.csv')).iloc[0]
    base10 = pd.read_csv(os.path.join(PRE, '2010.csv'), usecols=['hhidpn', 'agey_m', 'demcls'])
    base00 = pd.read_csv(os.path.join(PRE, '2000.csv'), usecols=['hhidpn', 'agey_m', 'demcls'])
    interviewed10 = base10['agey_m'].notna()
    age10 = base10['agey_m'] > 50
    first_steps = {
        2000: {'interviewed at baseline': flow00['interviewed in 2000'], 'excluded: age <= 50': flow00['excluded: age <= 50'],
               'excluded: dementia at baseline': flow00['excluded: dementia at baseline'],
               'excluded: baseline dementia status unknown': int(base00['demcls'].isna().sum())},
        2010: {'interviewed at baseline': int(interviewed10.sum()), 'excluded: age <= 50': int((interviewed10 & ~age10).sum()),
               'excluded: dementia at baseline': int((age10 & (base10['demcls'] == 1)).sum()),
               'excluded: baseline dementia status unknown': int((age10 & base10['demcls'].isna()).sum())},
    }
    rows = []
    for horizon, (w_dev, w_ext) in WINDOWS.items():
        dev, ext, P = benchmark.load_cohorts(horizon)
        for year, window, cohort in [(2000, w_dev, dev), (2010, w_ext, ext)]:
            s = status.loc[window]
            complete = cohort[P].notna().all(axis=1)
            row = {'cohort': f'{year} baseline', 'horizon (years)': horizon, **first_steps[year],
                   'eligible': int(s['eligible']),
                   'excluded: died before final wave, no dementia observed': int(s['died before final wave, no dementia observed']),
                   'excluded: alive or unknown, not assessed at final wave': int(s['alive or unknown, not assessed at final wave']),
                   'analytic cohort': len(cohort), 'incident dementia': int(cohort['y'].sum()),
                   'complete predictors': int(complete.sum()), 'incident dementia, complete predictors': int(cohort.loc[complete, 'y'].sum())}
            steps = [row['interviewed at baseline'], -row['excluded: age <= 50'], -row['excluded: dementia at baseline'],
                     -row['excluded: baseline dementia status unknown']]
            assert sum(steps) == row['eligible'], (year, horizon, sum(steps), row['eligible'])
            assert row['eligible'] - row['excluded: died before final wave, no dementia observed'] \
                - row['excluded: alive or unknown, not assessed at final wave'] == row['analytic cohort']
            rows.append(row)
    return pd.DataFrame(rows)


def _smd(a, b):
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def complete_vs_incomplete():
    characteristics = {
        'incident dementia (%)': lambda d: d['y'] * 100,
        'age (years)': lambda d: d['ageym'],
        'female (%)': lambda d: d['gender'] * 100,
        'proxy interview (%)': lambda d: d['proxy'] * 100,
        'non-Hispanic Black (%)': lambda d: d['raceeth1 '] * 100,
        'Hispanic (%)': lambda d: d['raceeth2 '] * 100,
        'education 13+ years (%)': lambda d: d['edstg4 13above'] * 100,
        'fair or poor self-rated health (%)': lambda d: (d['shlt'] >= 3).where(d['shlt'].notna()) * 100,
        'any ADL difficulty (%)': lambda d: (d['adl5a'] > 0).where(d['adl5a'].notna()) * 100,
        'number of mobility difficulties (0-5)': lambda d: d['mobila'],
        'ever stroke (%)': lambda d: d['stroke'] * 100,
        'ever diabetes (%)': lambda d: d['diabe'] * 100,
        'hospital stay, previous 2 years (%)': lambda d: d['hosp'] * 100,
    }
    rows = []
    for horizon in WINDOWS:
        dev, ext, P = benchmark.load_cohorts(horizon)
        for name, cohort in [('2000 baseline', dev), ('2010 baseline', ext)]:
            complete = cohort[P].notna().all(axis=1)
            for label, f in characteristics.items():
                v = f(cohort)
                rows.append({'cohort': name, 'horizon (years)': horizon, 'characteristic': label,
                             'complete predictors': round(v[complete].mean(), 2),
                             'incomplete predictors': round(v[~complete].mean(), 2),
                             'standardized difference': round(_smd(v[~complete], v[complete]), 3),
                             'n complete': int(complete.sum()), 'n incomplete': int((~complete).sum())})
    return pd.DataFrame(rows)


def cohort_overlap():
    rows = []
    for horizon in WINDOWS:
        dev, ext, P = benchmark.load_cohorts(horizon)
        status = pd.Series(overlap_status(ext['hhidpn'], dev['hhidpn']))
        counts = status.value_counts()
        rows.append({'horizon (years)': horizon, '2010 analytic cohort': len(ext),
                     **{k: int(counts.get(k, 0)) for k in ['returning participant', 'new participant, development household',
                                                           'new participant and household']},
                     'households in 2010 cohort': len(set(household(ext['hhidpn']))),
                     'of which also in 2000 cohort': len(set(household(ext['hhidpn'])) & set(household(dev['hhidpn'])))})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, f in [('participant_flow', participant_flow), ('complete_vs_incomplete', complete_vs_incomplete),
                    ('cohort_overlap', cohort_overlap)]:
        df = f()
        df.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
        print(f'\n== {name}'); print(df.to_string(index=False))
