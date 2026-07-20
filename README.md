# Fraud Detection Model Bake-off

Benchmarking XGBoost, LightGBM, and a PyTorch MLP on the IEEE-CIS Fraud Detection dataset (Kaggle).

## Notebooks
1. `01_eda_feature_engineering.ipynb` — EDA, cleaning, feature engineering, scaling
2. `02_xgboost_training.ipynb` — XGBoost tuning via RandomizedSearchCV
3. `03_neural_net_training.ipynb` — PyTorch MLP training
4. `04_model_comparison.ipynb` — Metrics, ROC curves, feature importance comparison
5. `05_lightgbm_training.ipynb` — LightGBM tuning (run on Kaggle)

## Results
LightGBM won: Test AUC 0.961, F1 0.72 (0.75 after threshold tuning) — beating XGBoost (0.954/0.68) and the MLP (0.939/0.60).

## Dataset
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — not included here, download separately.
