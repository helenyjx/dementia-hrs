import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import os
from imputation import *
def internal_test(year,model,param_grid,imputation_methods,grid_search_dir,X_columns, y_column):
    # ---------------- Header ----------------
    print("\n" + "=" * 100)
    print(f"Internal Test ({year})")
    print(f"Model: {model}")
    print("Protocol: 10-Fold Cross-Validation with 8:1:1 Train : Validation : Test splits")
    print("Metrics are reported as percentages with two decimal places.")
    print("=" * 100 + "\n")

    n_folds = 10

    final_rows = []  # summary rows (stds only + best candidate)

    for imp_method in imputation_methods:
        print(f"\nProcessing imputation method: {imp_method}")
        # ----- Step 1: Grid search using Validation AUPR across folds -----
        candidate_results = {}

        for fold in range(1, n_folds + 1):
            train_file = os.path.join(grid_search_dir, f"{year}int_train_{fold}.csv")
            val_file   = os.path.join(grid_search_dir, f"{year}int_val_{fold}.csv")

            train_df = pd.read_csv(train_file)
            val_df   = pd.read_csv(val_file)
            
            train_df.columns = train_df.columns.astype(str)
            val_df.columns   = val_df.columns.astype(str)

            # Impute training set per method
            X_train, y_train = impute_data_and_y(train_df, imp_method, X_columns, y_column)
            X_val, y_val = val_df[X_columns], val_df[y_column]
            
            X_train   = X_train.to_numpy(dtype=np.float32)
            X_val   = X_val.to_numpy(dtype=np.float32)
            

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
                except Exception as e:
                    print(e)
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

        # ----- Step 2: Refit on Train+Validation and evaluate on Test -----
        per_fold = []

        for fold in range(1, n_folds + 1):
            train_file = os.path.join(grid_search_dir, f"{year}int_train_{fold}.csv")
            val_file   = os.path.join(grid_search_dir, f"{year}int_val_{fold}.csv")
            test_file  = os.path.join(grid_search_dir, f"{year}int_test_{fold}.csv")

            train_df = pd.read_csv(train_file)
            val_df   = pd.read_csv(val_file)
            test_df  = pd.read_csv(test_file)
            
            for df_ in (train_df, val_df, test_df):
                df_.columns = df_.columns.astype(str)

            tv_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
            X_tv, y_tv = impute_data_and_y(tv_df, imp_method, X_columns, y_column)
            
            X_tv   = X_tv.to_numpy(dtype=np.float32)

            final_model = model(**dict(zip(candidate, best_candidate)))
            final_model.fit(X_tv, y_tv)

            X_test, y_test = test_df[X_columns], test_df[y_column]
            X_test   = X_test.to_numpy(dtype=np.float32)
            
            y_prob = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            per_fold.append({
                "AUC":  roc_auc_score(y_test, y_prob),
                "ACC":  accuracy_score(y_test, y_pred),
                "AUPR": average_precision_score(y_test, y_prob),
            })

        # Means and stds across folds
        aucs  = np.array([m["AUC"]  for m in per_fold])
        accs  = np.array([m["ACC"]  for m in per_fold])
        auprs = np.array([m["AUPR"] for m in per_fold])

        mean_auc,  std_auc  = aucs.mean(),  aucs.std()
        mean_acc,  std_acc  = accs.mean(),  accs.std()
        mean_aupr, std_aupr = auprs.mean(), auprs.std()

        print(
            f"Final test metrics for imputation method '{imp_method}': "
            f"AUC = {pct(mean_auc)}, ACC = {pct(mean_acc)}, AUPR = {pct(mean_aupr)}"
        )
        print(
            f"Standard deviations across folds (for error bars): "
            f"AUC_std = {pct(std_auc)}, ACC_std = {pct(std_acc)}, AUPR_std = {pct(std_aupr)}"
        )

        # Keep only stds + best candidate in summary
        final_rows.append({
            "Imputation": imp_method,
            "BestCandidate": best_candidate,
            "AUC_std":  std_auc,
            "ACC_std":  std_acc,
            "AUPR_std": std_aupr,
        })
    return final_rows, X_test, y_test, final_model

def pct(x: float) -> str:
    """Format a fraction as a percentage with two decimals."""
    return f"{x * 100:.2f}%"
