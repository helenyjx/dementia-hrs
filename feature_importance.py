# --- Top-5 Grouped Feature Importance (Logistic Regression, MICE only) ---
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

# Define grouping rule
def get_group_name(col):
    if col.startswith("edstg"):
        return "edstg (education)"
    elif col.startswith("cendiv"):
        return "cendiv (region)"
    elif col.startswith("mstat"):
        return "mstat (marital status)"
    elif col.startswith("raceeth"):
        return "raceeth (ethnicity)"
    else:
        return col  # keep as-is if not part of dummy set

def plot_top5_grouped_importance_mice(final_model, X, y, feat_names):
    """
    Permutation importance on the external test set with AUPR scoring.
    Groups one-hot encoded variables into their original categorical group,
    then plots the Top-5 groups with percentage-point labels OUTSIDE the bars.
    """

    # Compute permutation importance
    r = permutation_importance(
        final_model, X, y,
        scoring="average_precision",   # AUPR
        n_repeats=25,
        random_state=42,
        n_jobs=-1
    )

    # Raw importances at dummy-variable level
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "mean": r.importances_mean,
        "std":  r.importances_std
    })
    imp_df["mean_pp"] = imp_df["mean"] * 100.0  # percentage points

    imp_df["group"] = imp_df["feature"].apply(get_group_name)

    # Aggregate to group-level
    group_imp = (
        imp_df.groupby("group")["mean_pp"]
        .sum()
        .reset_index()
        .sort_values("mean_pp", ascending=False)
    )

    # Select Top-5 groups, reverse order for barh plot
    top5 = group_imp.head(5).iloc[::-1].copy()

    # --- Plot ---
    FONT_SIZE = 11
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(
        top5["group"],
        top5["mean_pp"],
        color="#9ECAE1",    # lighter blue
        edgecolor="#2171B5",
        linewidth=0.6
    )

    # Academic title
    ax.set_title(
        "Top-5 Grouped Permutation-Based Feature Importances (Δ↓AUPR)\n",
        fontsize=FONT_SIZE
    )
    ax.set_xlabel("Mean Decrease in AUPR (%)", fontsize=FONT_SIZE)
    ax.set_ylabel("Feature Group", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    xmax = max(1e-6, top5["mean_pp"].max())
    ax.set_xlim(0, xmax * 1.15)  # more space for labels outside
 # Add labels OUTSIDE bars
    for bar, val in zip(bars, top5["mean_pp"].values):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(val + xmax * 0.01, y, f"{val:.2f}%", va="center", ha="left",
                fontsize=FONT_SIZE, color="black")

    plt.tight_layout()
    plt.show()

    return top5

def feature_importance(model, X_test, y_test, scoring="average_precision", n_repeats=10):
    result = permutation_importance(
        model, X_test, y_test, scoring="average_precision", n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_importances_idx = result.importances_mean.argsort()[-5:]
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X_test.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title(
        "Top-5 Grouped Permutation-Based Feature Importances (Δ↓AUPR)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in AUPR score")
    ax.figure.tight_layout()
    top_5 = pd.DataFrame(
        result.importances[result.importances_mean.argsort()][::-1].T,
        columns=X_test.columns[result.importances_mean.argsort()[::-1]],
        ).mean()[:5].round(3)
    return top_5