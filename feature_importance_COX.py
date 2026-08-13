import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
# --- Top-5 Grouped Feature Importance (CoxPHFitter, MICE only, AUPR-based) ---
def permutation_importance_cox_aupr(final_model_cox, X, y, n_repeats=25, random_state=42):
    """
    Custom permutation importance for CoxPHFitter using AUPR.
    Returns mean/std ΔAUPR per *column* in X.
    """
    rng = np.random.RandomState(random_state)

    # Binary outcome = event indicator
    y_true = y["event"].astype(int).values

    # Baseline predictions = partial hazard (risk scores)
    baseline_pred = final_model_cox.predict_partial_hazard(X).values.ravel()
    baseline_aupr = average_precision_score(y_true, baseline_pred)

    importances = np.zeros((X.shape[1], n_repeats))

    for col in range(X.shape[1]):
        for n in range(n_repeats):
            X_perm = X.copy()
            X_perm.iloc[:, col] = rng.permutation(X_perm.iloc[:, col].values)
            perm_pred = final_model_cox.predict_partial_hazard(X_perm).values.ravel()
            aupr = average_precision_score(y_true, perm_pred)
            importances[col, n] = baseline_aupr - aupr

    return {
        "importances_mean": importances.mean(axis=1),
        "importances_std": importances.std(axis=1)
    }

def _cox_group_name(col: str) -> str:
    """
    Map one-hot/dummy columns back to their original categorical variable group.
    """
    if col.startswith("edstg"):
        return "edstg (education)"
    elif col.startswith("cendiv"):
        return "cendiv (region)"
    elif col.startswith("mstat"):
        return "mstat (marital status)"
    elif col.startswith("raceeth"):
        return "raceeth (ethnicity)"

    else:
        return col  # keep continuous or non-dummy features as-is

def plot_top5_grouped_perm_importance_mice(final_model_cox, X, y, feat_names):
    """
    Group one-hot encoded columns to their original variable, sum ΔAUPR (pp),
    and plot Top-5 groups with labels outside the bars.
    """
    r = permutation_importance_cox_aupr(final_model_cox, X, y, n_repeats=25)

    # Column-level importance (ΔAUPR absolute, then convert to % of total)
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "mean": r["importances_mean"],
        "std":  r["importances_std"],
    })

    # Convert to percentage points relative to total ΔAUPR (so it sums to 100%)
    total = imp_df["mean"].sum()
    if total <= 0:
        imp_df["mean_pp"] = 0.0
    else:
        imp_df["mean_pp"] = imp_df["mean"] / total * 100.0

    # Grouping
    imp_df["group"] = imp_df["feature"].apply(_cox_group_name)
    group_imp = (
        imp_df.groupby("group", as_index=False)["mean_pp"]
              .sum()
              .sort_values("mean_pp", ascending=False)
    )

    group_imp['normalized'] = group_imp['mean_pp'] / group_imp['mean_pp'].sum()

    # Select Top-5, reverse for barh
    top5 = group_imp.head(5).iloc[::-1].copy()

    # --- Plot ---
    FONT_SIZE = 11
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(
        top5["group"],
        top5["normalized"],
        color="#9ECAE1",
        edgecolor="#2171B5",
        linewidth=0.6
    )

    ax.set_title(
        "Top-5 Grouped Permutation-Based Feature Importances (ΔAUPR)\n"
        "Elastic Net Cox Model with MICE Imputation",
        fontsize=FONT_SIZE
    )
    ax.set_xlabel("Mean Decrease in AUPR (%)", fontsize=FONT_SIZE)
    ax.set_ylabel("Feature Group", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    # Extend x-axis for outside labels
    xmax = max(1e-6, float(top5["normalized"].max()))
    ax.set_xlim(0, xmax * 1.15)

    # Labels OUTSIDE the bars
    for bar, val in zip(bars, top5["normalized"].values):
        y_mid = bar.get_y() + bar.get_height() / 2
        ax.text(val + xmax * 0.01, y_mid, f"{val*100:.2f}%", va="center", ha="left",
                fontsize=FONT_SIZE, color="black")

    plt.tight_layout()
    plt.show()

    return top5