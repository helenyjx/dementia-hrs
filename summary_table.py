import pandas as pd

def pct(x: float) -> str:
    """Format a fraction as a percentage with two decimals."""
    return f"{x * 100:.2f}%"

def summary_table(final_rows):
    df_sum = pd.DataFrame(final_rows)
    cols = ["Imputation", "BestCandidate", "AUC_std", "ACC_std", "AUPR_std"]
    print("\n" + "=" * 100)
    print("Internal Cross-Validation — Standard Deviations Summary (for error bars)")
    print("=" * 100)
    display_df = df_sum.copy()
    for c in ["AUC_std", "ACC_std", "AUPR_std"]:
        display_df[c] = display_df[c].apply(pct)
    print(display_df[cols].to_string(index=False))