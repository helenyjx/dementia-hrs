# Benchmarking Performance of Long-Term Prognostic Models for Dementia Risk Using the Health and Retirement Study

## Overview

This repository implements a population-based benchmarking framework for **long-term dementia risk prediction** using longitudinal data from the **Health and Retirement Study (HRS, 2000–2018)**. Eight predictive models spanning four methodological families — regression, Bayesian, tree-based, and deep learning — are trained on baseline (2000) HRS predictors to estimate incident dementia risk over **6-year** and **8-year** horizons, and are evaluated using both **internal** (2000→2006/2008) and **temporal external** (2010→2016/2018) validation.

The evaluated models are:

* **Regression-based**: Penalized Logistic Regression
* **Bayesian probabilistic**: Gaussian Naïve Bayes (GNB)
* **Tree-based**: Decision Tree, LightGBM, CatBoost
* **Deep learning**: RealMLP, Graph Attention Network (GAT), TabICL

The pipeline also includes permutation-based feature importance analysis (measured as mean decrease in AUPRC) to identify the predictors most strongly associated with dementia risk across models.

## Repository Structure

```
dementia/
├── config.py                              # Global paths and column definitions
├── benchmark.py                           # Evaluation protocol: splits, training-only imputation, tuning, thresholds, metrics
├── run_benchmark.py                       # Command-line runner for benchmark.py
├── results.ipynb                          # Tables of the benchmark results
├── evaluation.py                          # Confidence intervals, paired differences, calibration, decision curves, runtime
├── importance.py                          # Permutation and SHAP importance on the 2010 cohort
├── cohort_tables.py                       # Participant flow and cohort-comparison tables
├── make_figures.py                        # Manuscript figures from results/tables (written to figures/)
├── feature_importance.py                  # Permutation-based feature importance (sklearn-style classifiers)
├── neuralclassifier_fixed.py              # PyTorch classifier wrappers (MLP/GNN/GCN/GAT) with focal loss + SMOTE
├── nnblocks.py                            # GraphSAGE / GCN / GAT building blocks (PyTorch Geometric)
├── readHRS.R                              # Converts raw fixed-width HRS wave files (.DA/.DCT) to CSV
├── preprocess_2000.ipynb                  # Parses the 2000 baseline wave into structured predictors
├── preprocess_2002-2016.ipynb             # Parses the 2002–2016 waves
├── preprocess_2018-2020.ipynb             # Parses the 2018–2020 waves
├── all_y.ipynb                            # Builds incident-dementia outcome labels and the outcome-status table
├── dont_do.py                             # Reads HRS core "don't do" answers for functional items
├── PREDICTOR_DICTIONARY.md                # Definitions, sources, and coding of all predictors
├── data_management (2000-2006).ipynb      # Builds the 6-year analytic dataset and internal/external splits
├── data_management (2000-2008).ipynb      # Builds the 8-year analytic dataset and internal/external splits
├── results/                               # Benchmark outputs per horizon and model (created by run_benchmark.py)
├── figures/                               # Manuscript figures (PDF and PNG, created by make_figures.py)
└── data/
    ├── preprocessed data/                 # Per-year cleaned/encoded CSVs produced by preprocess_*.ipynb
    ├── 2000-2006/                         # Model-ready folds for the 6-year prediction horizon
    └── 2000-2008/                         # Model-ready folds for the 8-year prediction horizon
```

## Files Description

* **`config.py`** — Defines global resource paths (raw/processed data directories) and the HRS variable sets used for self-respondent and proxy-respondent cognitive assessment (`SELF_DEM_COLS`, `PROXY_DEM_COLS`).
* **`benchmark.py`** — The evaluation protocol shared by all models (see Evaluation below). `build_imputations()` caches the training-only miceforest imputations for every repeat; `run()` tunes, thresholds, refits, and scores one model and writes its outputs to `results/h{6,8}/{model}/`.
* **`run_benchmark.py`** — Command-line entry point: `impute` and `fit` steps per horizon and model.
* **`results.ipynb`** — Reads `results/` and builds the internal (mean and SD over repeats) and temporal-validation tables and the selected hyperparameters.
* **`dont_do.py`** — Maps RAND functional-limitation items to the HRS core questions and reads the "don't do" answers, which are kept as a separate category.
* **`feature_importance.py`** — Permutation-based feature importance (scored via AUPRC, 10–25 repeats) for scikit-learn–style classifiers, with one-hot dummy variables (education, region, marital status, race/ethnicity) grouped back to their original categorical variable for plotting.
* **`neuralclassifier_fixed.py`** — `NeuralClassifier` base class plus `MLPClassifier` / `GNNClassifier` / `GCNClassifier` / `GATClassifier` wrappers implementing training loops, focal loss, class-weighted BCE loss, and optional SMOTE oversampling for the imbalanced dementia outcome.
* **`nnblocks.py`** — Underlying PyTorch / PyTorch Geometric network blocks: `MLP`, `GNN` (GraphSAGE), `GCN`, and `GAT`, each with residual and batch-normalization variants.
* **`readHRS.R`** — Template script that reads a raw fixed-width HRS data file (`.DA`) together with its data dictionary (`.DCT`) and writes a labeled CSV.
## Requirements

Core dependencies (exact versions used are recorded in each `results/h*/*/selection.json`):

```
numpy
pandas
scikit-learn
lightgbm
catboost
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

The final predictor set comprises **60 variables (94 model inputs)** across seven domains: demographic/family characteristics, socioeconomic/geographic factors, self-reported health and behaviors, chronic conditions, healthcare utilization, functional limitations (ADL/IADL), and mobility/physical functioning. Definitions, RAND/HRS sources, and coding are in [PREDICTOR_DICTIONARY.md](PREDICTOR_DICTIONARY.md).

Dementia status at each wave (`demcls`) is derived from HRS's validated self-respondent (TICS-m) and proxy-respondent cognitive assessment protocols, using thresholds validated against the Aging, Demographics, and Memory Study (ADAMS). The outcome `y` is **incident dementia**: among participants aged > 50 and assessed dementia-free at baseline, `y = 1` if dementia is observed at any follow-up wave in the window, `y = 0` if the participant is assessed dementia-free at the final wave of the window. Participants not assessed at the final wave (deceased, lost to follow-up, or nonresponse) are excluded; their numbers are in `data/preprocessed data/outcome_status.csv`.

### Data Format

The benchmark reads the development cohorts `data/preprocessed data/2000(00-06y)encode.csv` / `2000(00-08y)encode.csv` and the temporal-validation cohorts `data/2000-2006/2010(10-16y).csv` / `data/2000-2008/2010(10-18y).csv` (all eligible participants with a known outcome; splits and imputations are made inside `benchmark.py`). The data-management notebooks also write fixed split files under `data/2000-2006/` and `data/2000-2008/`:

* `{year}int_train_{fold}.csv`, `{year}int_val_{fold}.csv`, `{year}int_test_{fold}.csv` — per-fold internal 10-fold CV splits (folds 1–10)
* `{year}internal_train.csv`, `{year}internal_val.csv`, `{year}internal_test.csv`, `{year}internal_trainval.csv` — pooled internal train/validation/test sets

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

**5. Model training and evaluation** — cache the training-only imputations, then fit all models, for each horizon:
```bash
python run_benchmark.py impute --horizon 6
python run_benchmark.py fit --horizon 6 --model all
python run_benchmark.py impute --horizon 8
python run_benchmark.py fit --horizon 8 --model all
```
A single model can be run with `--model LightGBM` (names: LR, GNB, DT, LightGBM, CatBoost, RealMLP, TabICL, GAT). Then open `results.ipynb` for the tables.

**6. Tables and figures** — aggregate tables (written to `results/tables/`) and manuscript figures (written to `figures/`):
```bash
python evaluation.py
python cohort_tables.py
python importance.py --horizon 6
python make_figures.py
```

## Model Architecture

1. **Regression-based models**: Penalized logistic regression (L1/L2) for binary classification.
2. **Bayesian probabilistic model**: Gaussian Naïve Bayes, modeling class-conditional feature distributions under a conditional independence assumption.
3. **Tree-based models**: Decision tree (recursive partitioning); LightGBM (histogram-based gradient boosting); CatBoost (ordered boosting with native categorical encoding).
4. **Deep learning models**: RealMLP (tuned multilayer perceptron for tabular data); GAT (graph attention network over GraphSAGE/GCN/GAT message-passing blocks in `nnblocks.py`, trained via `neuralclassifier_fixed.py` with focal loss and optional SMOTE oversampling); TabICL (transformer-based tabular foundation model using in-context learning).

## Evaluation

For each horizon and each of 10 repeats, the development cohort is split into training, validation, and test sets (60/20/20, stratified). The miceforest imputation model (20 iterations) is fitted on the training set only and applied to the validation and test sets. Hyperparameters are selected within each repeat by validation AUPRC, and the decision threshold is the F1-maximizing threshold on the validation set. The selected model is refitted on training plus validation data and evaluated once on the test set. For temporal validation, the configuration, threshold, imputation model, and final model are all obtained from the 2000 development cohort, and the 2010 cohort is used only for evaluation.

Metrics: **AUROC**, **AUPRC**, **Accuracy**, **Precision**, **Recall**, **F1**, and the **Brier score**, reported for all participants (primary; missing predictors imputed by the training-only imputer) and for participants with complete predictors. Because dementia incidence is a minority outcome, **AUPRC** is used for hyperparameter selection.

## Feature Importance

Predictor importance is computed using permutation-based importance: each predictor is permuted 10–25 times, and importance is defined as the mean decrease in AUPRC (Breiman's permutation paradigm). One-hot encoded categorical variables are aggregated back to their parent construct (e.g., education, census division, marital status, race/ethnicity) before ranking.

## Key Results

> These are the results of the originally submitted manuscript; they are being updated for the revised manuscript.

Across nine benchmarked models spanning regression, Bayesian, tree-based, and deep learning families, **TabICL, LightGBM, and CatBoost consistently achieved the strongest predictive performance** in both internal and external validation, with comparable discrimination (AUROC differences under 1% between the three at external validation) but larger separation on precision-oriented metrics — TabICL held a modest, consistent edge in AUPRC and precision, while LightGBM and CatBoost traded that for higher recall. Penalized logistic regression and RealMLP formed a second performance tier with moderate but stable results, while GAT and Gaussian Naïve Bayes ranked lowest across all metrics and horizons. Model rankings were stable across the 6- and 8-year prediction windows, indicating good temporal generalizability. Permutation-based feature importance identified **age and educational attainment as the most consistently influential predictors** across all nine models.
