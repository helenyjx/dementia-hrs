# Helper Function: Data Imputation
import pandas as pd
import miceforest as mf
from sklearn.impute import SimpleImputer

def impute_data_and_y(df, method, X_columns, y_column):
    """
    Returns (X, y) using the specified imputation method.
    For method 'none', rows with missing values are dropped.
    """
    if method == 'mice':
        kernel = mf.ImputationKernel(df[X_columns].copy(), random_state=42)
        kernel.mice(20)
        X_imp = kernel.complete_data(0)
        return X_imp, df[y_column]
    elif method == 'mean':
        imp = SimpleImputer(strategy='mean')
        X_imp = pd.DataFrame(imp.fit_transform(df[X_columns]), columns=X_columns)
        return X_imp, df[y_column]
    elif method == 'none':
        if type(y_column) is list:
            df_clean = df.dropna(subset=X_columns + y_column)
        else:
            df_clean = df.dropna(subset=X_columns + [y_column])
        return df_clean[X_columns], df_clean[y_column]
    else:
        raise ValueError("Unknown imputation method: " + method)
    