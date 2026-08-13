import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import os
from imputation import *
def external_test(year,model,param_grid,imputation_methods,grid_search_dir,train_val_path,test_base_path,X_columns, y_column):
    # ---- Run title/banner ----
    print("\n" + "="*100)
    print(f"{int(year[-4:])-int(year[:4])}-Year Exterval External Test ({year[:2]+'1'+year[3:7]+'1'+year[8]}) — Grid Search via Parameter Tuning")
    print(f"Train/Val: {year} • 10-fold CV")
    print("Metrics shown as percentages with 2 decimals")
    # print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*100 + "\n")
    # --------------------------

    n_folds = 10

    final_test_results = []  # summary rows (stds only + best candidate)

    for imp_method in imputation_methods:
        print(f"\nProcessing imputation method: {imp_method}")
        # Dictionary to hold lists of validation AUPR per candidate across folds        candidate_results = {}
        candidate_results = {}
        for fold in range(1, n_folds + 1):
            train_file = os.path.join(grid_search_dir, f"{year}train_{fold}.csv")
            val_file   = os.path.join(grid_search_dir, f"{year}val_{fold}.csv")

            train_df = pd.read_csv(train_file)
            val_df   = pd.read_csv(val_file)
            train_df.columns = train_df.columns.astype(str)
            val_df.columns   = val_df.columns.astype(str)

            # Impute training set per method
            X_train, y_train = impute_data_and_y(train_df, imp_method, X_columns, y_column)
            X_val, y_val = val_df[X_columns], val_df[y_column]

            # Grid search
            combinations = itertools.product(*param_grid.values())
            for parameter in combinations:
                candidate = list(param_grid.keys())
                try:
                    m = model(**dict(zip(candidate, parameter)))
                    m.fit(X_train, y_train)
                    y_val_prob = m.predict_proba(X_val)[:, 1]
                    aupr = average_precision_score(y_val, y_val_prob)
                    candidate_results.setdefault(parameter, []).append(aupr)
                except Exception:
                    continue

        if not candidate_results:
            print(f"[{imp_method}] No valid candidates found; skipping.")
            continue

        avg_val_aupr = {cand: np.mean(scores) for cand, scores in candidate_results.items()}
        best_candidate = max(avg_val_aupr, key=avg_val_aupr.get)
        best_val_aupr = avg_val_aupr[best_candidate]

        print(
            f"For imputation method '{imp_method}', best candidate: {best_candidate} "
            f"with average validation AUPR = {pct(best_val_aupr)}"
        )

        # Retrain on Full Data and Evaluate on External Test Data
        # Load full training+validation data        per_fold = []
        full_train_df = pd.read_csv(train_val_path)
        full_train_df.columns = full_train_df.columns.astype(str)
        # Impute full training data using the current method
        X_full_train, y_full_train = impute_data_and_y(full_train_df, imp_method, X_columns,y_column)    
        final_model = model(**dict(zip(candidate, best_candidate)))
        final_model.fit(X_full_train, y_full_train)
        # Load external test data
        test_df = pd.read_csv(test_base_path)
        test_df.columns = test_df.columns.astype(str)
        X_test = test_df[X_columns]
        y_test = test_df[y_column]

        y_test_pred = final_model.predict(X_test)
        y_test_prob = final_model.predict_proba(X_test)[:, 1]

        test_auc   = roc_auc_score(y_test, y_test_prob)
        test_acc   = accuracy_score(y_test, y_test_pred)
        test_aupr  = average_precision_score(y_test, y_test_prob)

        # Store both numeric and formatted percentage strings
        final_test_results.append({
            'Imputation Method': imp_method,
            'Model': 'Model',
            'AUC_num': test_auc,    # numeric (0-1)
            'ACC_num': test_acc,    # numeric (0-1)
            'AUPR_num': test_aupr,  # numeric (0-1)
            'AUC': pct(test_auc),   # formatted percentage
            'ACC': pct(test_acc),
            'AUPR': pct(test_aupr)
        })

        print(
            f"Final test metrics for imputation method '{imp_method}': "
            f"AUC = {pct(test_auc)}, ACC = {pct(test_acc)}, AUPR = {pct(test_aupr)}"
        )

    return final_test_results, X_test, y_test, final_model

def pct(x: float) -> str:
    """Format a fraction as a percentage with two decimals."""
    return f"{x * 100:.2f}%"