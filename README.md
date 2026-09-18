# CredFraud — Fraud Detection ML Pipeline

An end-to-end machine learning pipeline for credit card transaction fraud detection, built on the IEEE-CIS Fraud Detection dataset (Kaggle). Covers EDA, feature engineering, model benchmarking, explainability, and threshold tuning — with experiment tracking in MLflow.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle), consisting of:

- `train_transaction.csv`, `train_identity.csv`
- `test_transaction.csv`, `test_identity.csv`
- `sample.csv`

Training data was merged (`train_transaction` + `train_identity`) on `TransactionID`.

**Class imbalance:** ~20,663 fraud cases out of the full training set (~3.5% fraud rate). Because raw accuracy is a poor metric for this class distribution (a model predicting "not fraud" for everything would score highly), evaluation focuses on **recall, precision, PR-AUC, and ROC-AUC** instead.

## Exploratory Data Analysis

- Dropped columns with a missing-value rate above 90% (e.g. `id_24`, `id_25`).
- Fraud vs. non-fraud count and percentage distributions.
- Transaction amount histogram (clipped at 1000, 60 bins) and boxplot of fraud vs. non-fraud amounts — fraud transactions skew toward slightly higher amounts.
- Top 15 numerical features by correlation with `isFraud`.
- Categorical feature audit: unique value counts and missing rate per feature.
- Transaction amount bucketed into 10 bins, with fraud rate computed per bucket.
- Missing value imputation: median for numerical columns, mode for categorical columns (median chosen for robustness to outliers/skew).

## Feature Engineering

**Time-based** (derived from `TransactionDT`, a seconds-based timestamp):
- `txn_hour`, `txn_day`, `txn_weekday`, `is_weekend`, `is_night`

**Amount-based:**
- `amt_log` — log-transformed transaction amount
- `amt_bin` — amount bucketed into 10 bins
- `amt_percent` — percentile rank of amount
- `amt_zscore` — standardized amount (0 mean, 1 std)

**Email domain features:**
- Binary match flag: `1` if purchaser and recipient email domains match, else `0`
- `email_rarity` — frequency of the email domain across the dataset
- `email_is_rare` — `1` if `email_rarity < 0.01`, else `0`

## Preprocessing

1. Categorical features encoded with `LabelEncoder`.
2. Train/test split, then all features cast to `float32`.
3. `StandardScaler` fit on the training split and applied to both splits.
4. Artifacts saved for reuse: `X_train`, `X_test`, `y_train`, `y_test`, `feature_names.json`.
5. **SMOTE** applied to the training split only (after the split, before model training) to oversample the minority (fraud) class.

## Model Training & Comparison

Three models were benchmarked, each tuned with `RandomizedSearchCV` (10 iterations, 3-fold CV, AUC as the scoring metric), with runs logged to **MLflow**.

### XGBoost
Search space: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`.

**Test AUC: 0.9539**

### LightGBM — best performing model
Search space: `n_estimators`, `max_depth`, `num_leaves`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`.

**Test AUC: 0.9612 | Test F1: 0.7235**

### PyTorch MLP
Feature set reduced from 432 to the **top 200 features** (selected via LightGBM + SHAP, covering 96% of feature importance).

- Architecture: `200 → 256 → 128 → 64 → 1`, ReLU activations, BatchNorm after each layer, dropout (0.3 / 0.3 / 0.2)
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam (`lr=1e-3`, `weight_decay=1e-4`)
- Scheduler: `ReduceLROnPlateau` (patience 3, factor 0.5, mode `min`, tracking loss)
- 50 epochs, batch gradient descent

**Test AUC: 0.9394 | Test F1: 0.6019**

### Comparison
All three models were evaluated on the same held-out test set (MLP restricted to its 200-feature subset). Per-model confusion matrices, classification reports, and a combined ROC-AUC curve were plotted to compare performance directly.

**LightGBM was selected as the winning model.**

## Explainability

SHAP was used to explain the tree-based models (XGBoost, LightGBM), plotting the top 15 most important features for each. The engineered features consistently ranked among the most influential: `txn_weekday`, `txn_hour`, `amt_bin`, `amt_percent`, `email_rarity`, `is_weekend`.

## Threshold Tuning

The default 0.5 classification threshold is suboptimal here since recall (catching as much fraud as possible) matters more than a balanced threshold would give. The threshold was tuned by scanning F1 scores across thresholds, landing on **~0.277** — this improved recall meaningfully at a moderate cost to precision. The tuned threshold was logged alongside the model in MLflow.

## Final Validation

The tuned LightGBM model was run against the untouched Kaggle test set, with the full preprocessing pipeline (imputation, feature engineering, high-missing-column drops, label encoding, scaling — using the artifacts saved during training) replicated exactly to keep the test data compatible with the trained model. This produced a predicted fraud rate of **~2.9%** on the test set, consistent with the training distribution.

## Why This Isn't Deployed Live

This model is deliberately **not deployed as a live/production service**. The IEEE-CIS dataset includes a large number of masked, anonymized features (`id_*`, `V_*`, `C_*` columns — more than half the total feature set) that come from the specific data collection pipeline behind this dataset. Real-world fraud systems built on typical customer/transaction data wouldn't have access to this same feature set, so a model trained on it wouldn't generalize to a live setting. This project focuses on demonstrating the full ML lifecycle (EDA → feature engineering → model selection → tuning → explainability → experiment tracking) end-to-end rather than shipping a production endpoint.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, PyTorch, imbalanced-learn (SMOTE), SHAP, MLflow

## Repository Structure

```
CredFraud/
├── notebooks/          # EDA, feature engineering, model training/comparison
├── fraud-test/         # testing code for real dataset
└── README.md
```
