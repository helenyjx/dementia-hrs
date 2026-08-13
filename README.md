# Benchmarking Performance of Long-Term Prognostic Models for Dementia Risk Using the Health and Retirement Study

## Overview

This repository implements a population-based benchmarking framework for **long-term dementia risk prediction** using longitudinal data from the **Health and Retirement Study (HRS, 2000–2018)**. Nine predictive models spanning four methodological families — regression, Bayesian, tree-based, and deep learning — are trained on baseline (2000) HRS predictors to estimate incident dementia risk over **6-year** and **8-year** horizons, and are evaluated using both **internal** (2000→2006/2008) and **temporal external** (2010→2016/2018) validation.

The evaluated models are:

* **Regression-based**: Penalized Logistic Regression, Elastic Net Cox Regression
* **Bayesian probabilistic**: Gaussian Naïve Bayes (GNB)
* **Tree-based**: Decision Tree, LightGBM, CatBoost
* **Deep learning**: RealMLP, Graph Attention Network (GAT), TabICL

The pipeline also includes permutation-based feature importance analysis (measured as mean decrease in AUPRC) to identify the predictors most strongly associated with dementia risk across models.

## Repository Structure

```
dementia/
├── config.py                              # Global paths and column definitions
├── add_time_event(2000-2006).py           # Builds (time, event) survival columns for the 6-year Cox model
├── imputation.py                          # Shared imputation utility (none / mean / miceforest)
├── internal_test.py                       # Internal validation harness (10-fold grid search + evaluation)
├── external_test.py                       # External (temporal) validation harness
├── summary_table.py                       # Formats cross-fold standard-deviation summary tables
├── feature_importance.py                  # Permutation-based feature importance (sklearn-style classifiers)
├── feature_importance_COX.py              # Permutation-based feature importance for CoxPHFitter
├── neuralclassifier_fixed.py              # PyTorch classifier wrappers (MLP/GNN/GCN/GAT) with focal loss + SMOTE
├── nnblocks.py                            # GraphSAGE / GCN / GAT building blocks (PyTorch Geometric)
├── readHRS.R                              # Converts raw fixed-width HRS wave files (.DA/.DCT) to CSV
├── preprocess_2000.ipynb                  # Parses the 2000 baseline wave into structured predictors
├── preprocess_2002-2016.ipynb             # Parses the 2002–2016 waves
├── preprocess_2018-2020.ipynb             # Parses the 2018–2020 waves
├── all_y.ipynb                            # Aggregates per-wave cognitive/dementia classification into outcome labels
├── data_management (2000-2006).ipynb      # Builds the 6-year analytic dataset and internal/external splits
├── data_management (2000-2008).ipynb      # Builds the 8-year analytic dataset and internal/external splits
├── internal test/                         # One notebook per model — internal validation (2000→2006/2008)
│   ├── LR.ipynb, COX.ipynb, GNB.ipynb, DT.ipynb
│   ├── LightGBM.ipynb, CatBoost.ipynb
│   └── GAT.ipynb, RealMLP.ipynb, TabICL-transformer.ipynb
├── external test/                         # One notebook per model — temporal external validation (2010→2016/2018)
│   ├── LR.ipynb, COX.ipynb, GNB.ipynb, DT.ipynb
│   ├── LightGBM.ipynb, CatBoost.ipynb
│   └── GAT.ipynb, RealMLP.ipynb, TabICL-transformer.ipynb
└── data/
    ├── preprocessed data/                 # Per-year cleaned/encoded CSVs produced by preprocess_*.ipynb
    ├── 2000-2006/                         # Model-ready folds for the 6-year prediction horizon
    └── 2000-2008/                         # Model-ready folds for the 8-year prediction horizon
```

## Files Description

* **`config.py`** — Defines global resource paths (raw/processed data directories) and the HRS variable sets used for self-respondent and proxy-respondent cognitive assessment (`SELF_DEM_COLS`, `PROXY_DEM_COLS`).
* **`add_time_event(2000-2006).py`** — Constructs the survival-analysis targets required by the Elastic Net Cox model: for each participant, `first_1_else_last_0_with_value()` scans the per-wave binary dementia labels and returns `time` (years from the 2000 baseline to dementia onset, or to the last observed wave if censored) and `event` (1 = incident dementia, 0 = censored). These `(time, event)` columns are merged onto the fold-wise train/validation files and the encoded full dataset to produce the `*_COX_*.csv` inputs consumed by `internal test/COX.ipynb` / `external test/COX.ipynb`. **Note:** `base_dir` and the input/output filenames in this script currently point to a sibling directory (`dementia_benchmark/interval/...`) rather than this repo's `data/2000-2006/` folder — update these paths before re-running it; the `*_COX_*.csv` files already present under `data/2000-2006/` appear to be its prior output.
* **`imputation.py`** — `impute_data_and_y()`: applies one of three missing-data strategies — complete-case (`none`), mean imputation (`mean`), or `miceforest` multiple imputation by chained equations with LightGBM (`mice`, T = 20 iterations).
* **`internal_test.py`** — `internal_test()`: performs 10-fold grid search over a model's hyperparameter space (selecting the configuration with the highest mean validation AUPRC), refits on train+validation, and evaluates on held-out internal test folds (8:1:1 split protocol).
* **`external_test.py`** — `external_test()`: performs the analogous grid search on the 2000 HRS training/validation folds, then refits on the full 2000 sample and evaluates on the temporally held-out 2010 HRS external test set.
* **`summary_table.py`** — Formats the internal cross-validation standard-deviation summary (AUROC/Accuracy/AUPRC) used for reporting error bars across folds.
* **`feature_importance.py`** — Permutation-based feature importance (scored via AUPRC, 10–25 repeats) for scikit-learn–style classifiers, with one-hot dummy variables (education, region, marital status, race/ethnicity) grouped back to their original categorical variable for plotting.
* **`feature_importance_COX.py`** — Analogous permutation importance implementation for the `CoxPHFitter` survival model (permutes partial hazard predictions and scores the resulting change in AUPRC).
* **`neuralclassifier_fixed.py`** — `NeuralClassifier` base class plus `MLPClassifier` / `GNNClassifier` / `GCNClassifier` / `GATClassifier` wrappers implementing training loops, focal loss, class-weighted BCE loss, and optional SMOTE oversampling for the imbalanced dementia outcome.
* **`nnblocks.py`** — Underlying PyTorch / PyTorch Geometric network blocks: `MLP`, `GNN` (GraphSAGE), `GCN`, and `GAT`, each with residual and batch-normalization variants.
* **`readHRS.R`** — Template script that reads a raw fixed-width HRS data file (`.DA`) together with its data dictionary (`.DCT`) and writes a labeled CSV.
## Requirements

Core dependencies (Python 3.10.10 for classical models; Python 3.12.7 for GAT/TabICL, per the paper's Methods section 2.4.2):

```
numpy
pandas
scikit-learn
lightgbm
catboost
lifelines
miceforest
imbalanced-learn
torch
torch_geometric
pytabkit          # RealMLP
tabicl            # TabICL
matplotlib
tqdm
```

R dependencies (for `readHRS.R`): `readr`

## Data

### Data Availability

This study uses restricted-access survey data from the **Health and Retirement Study (HRS)**, sponsored by the National Institute on Aging (NIA U01AG009740) and conducted by the University of Michigan. Raw HRS files are **not redistributed in this repository** and must be obtained directly by registered researchers at [https://hrs.isr.umich.edu](https://hrs.isr.umich.edu). 
### Predictors and Outcome

The final predictor set comprises **77 variables** across seven domains: demographic/family characteristics, socioeconomic/geographic factors, self-reported health and behaviors, chronic conditions, healthcare utilization, functional limitations (ADL/IADL), and mobility/physical functioning. The binary outcome (`y` / `demcls`) is derived from HRS's validated self-respondent (TICS-m) and proxy-respondent cognitive assessment protocols, using thresholds validated against the Aging, Demographics, and Memory Study (ADAMS).

### Data Format

Model-ready CSVs under `data/2000-2006/` and `data/2000-2008/` follow the naming convention:

* `{year}int_train_{fold}.csv`, `{year}int_val_{fold}.csv`, `{year}int_test_{fold}.csv` — per-fold internal 10-fold CV splits (folds 1–10)
* `{year}internal_train.csv`, `{year}internal_val.csv`, `{year}internal_test.csv`, `{year}internal_trainval.csv` — pooled internal train/validation/test sets
* `*_COX_*` variants — the same splits with `time`-to-event and `event` columns for the Elastic Net Cox model

## Usage

Run the pipeline in the following order:

**1. Raw data extraction (per wave, optional if starting from RAND HRS files)**
```bash
Rscript readHRS.R
```

**2. Preprocessing** — parse raw wave data into structured, encoded predictors
```
preprocess_2000.ipynb
preprocess_2002-2016.ipynb
preprocess_2018-2020.ipynb
```

**3. Outcome construction** — aggregate per-wave cognitive classification into the dementia outcome
```
all_y.ipynb
```

**4. Dataset assembly** — build the internal/external, fold-wise train/val/test splits for each prediction horizon
```
data_management (2000-2006).ipynb
data_management (2000-2008).ipynb
```

**5. Survival targets for the Cox model** — compute the `(time, event)` columns for the 6-year horizon (update `base_dir` and file paths to point at `data/2000-2006/` first):
```bash
python "add_time_event(2000-2006).py"
```

**6. Model training and evaluation** — run the corresponding notebook for each model under `internal test/` (2000→2006/2008 internal validation) and `external test/` (2010→2016/2018 temporal external validation):
```
internal test/LR.ipynb
internal test/COX.ipynb
internal test/GNB.ipynb
internal test/DT.ipynb
internal test/LightGBM.ipynb
internal test/CatBoost.ipynb
internal test/GAT.ipynb
internal test/RealMLP.ipynb
internal test/TabICL-transformer.ipynb
```
Each notebook imports the shared modules (`imputation`, `internal_test`/`external_test`, `summary_table`, `feature_importance`), defines the model-specific hyperparameter grid, and reports AUROC, AUPRC, Accuracy, Precision, Recall, and F1.

## Model Architecture

1. **Regression-based models**: Penalized logistic regression (L1/L2) for binary classification; Elastic Net Cox regression for time-to-event modeling with combined L1–L2 penalty.
2. **Bayesian probabilistic model**: Gaussian Naïve Bayes, modeling class-conditional feature distributions under a conditional independence assumption.
3. **Tree-based models**: Decision tree (recursive partitioning); LightGBM (histogram-based gradient boosting); CatBoost (ordered boosting with native categorical encoding).
4. **Deep learning models**: RealMLP (tuned multilayer perceptron for tabular data); GAT (graph attention network over GraphSAGE/GCN/GAT message-passing blocks in `nnblocks.py`, trained via `neuralclassifier_fixed.py` with focal loss and optional SMOTE oversampling); TabICL (transformer-based tabular foundation model using in-context learning).

## Evaluation

Model performance is assessed using **AUROC**, **AUPRC**, **Accuracy**, **Precision**, **Recall**, and **F1**. Because dementia incidence is a minority outcome, **AUPRC** is emphasized as the primary metric for hyperparameter selection and model comparison, consistent with the paper's evaluation strategy (section 2.4.3).

## Feature Importance

Predictor importance is computed using permutation-based importance on the external test set: each predictor is permuted 10–25 times, and importance is defined as the mean decrease in AUPRC (Breiman's permutation paradigm). One-hot encoded categorical variables are aggregated back to their parent construct (e.g., education, census division, marital status, race/ethnicity) before ranking.
