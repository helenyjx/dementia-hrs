"""Command-line runner for the benchmark protocol in benchmark.py.

Examples:
    python run_benchmark.py impute --horizon 6            # cache train-only imputations (shared by all models)
    python run_benchmark.py fit --horizon 6 --model LightGBM
    python run_benchmark.py fit --horizon 8 --model all
"""
import argparse
import os
import warnings

os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count()))  # silence joblib's core-count probe on Windows

import benchmark

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except ImportError:
    pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['impute', 'fit'])
    ap.add_argument('--horizon', type=int, choices=[6, 8], required=True)
    ap.add_argument('--model', default='all')
    ap.add_argument('--outcome', default='final', choices=['final', 'any'],
                    help="'final' (primary) or 'any' (sensitivity analysis of the outcome definition)")
    args = ap.parse_args()
    benchmark.OUTCOME = args.outcome
    if args.step == 'impute':
        benchmark.build_imputations(args.horizon)
    else:
        names = list(benchmark.model_specs()) if args.model == 'all' else [args.model]
        for name in names:
            benchmark.run(name, args.horizon)
