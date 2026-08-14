# EV-Battery-Failure-Prediction
A machine learning model designed to predict electric vehicle battery failures from a highly imbalanced dataset with a 9.96% failure rate. The workflow features leakage-safe median and mode imputation , domain-driven feature engineering , and decision threshold optimization that prioritizes PR-AUC over ROC-AUC for reliable real-world performance.

This repository contains a complete, 14-stage machine learning workflow designed to predict battery failure in electric vehicles. Built to handle a highly imbalanced dataset of 200,000 telemetry records (9.96% failure rate) across 70 features, the codebase explicitly optimizes for Precision-Recall AUC (PR-AUC) over standard accuracy to ensure practical reliability.

Key Highlights:
  1. Domain-Driven Feature Engineering: Extracts physical degradation signals by engineering 9 custom features, including soh_per_cycle, degradation_rate, and thermal_spread.
  2. Leakage-Safe Preprocessing: Utilizes scikit-learn's ColumnTransformers to ensure median imputation and scaling statistics are strictly learned only on training folds.
  3. Robust Model Selection: Evaluates Logistic Regression, Random Forest, XGBoost, and LightGBM. Notably, Logistic Regression achieves a highly competitive CV PR-AUC of 0.9227, demonstrating that the decision boundary for degradation is largely linear.
  4. Threshold & Business Optimization: Sweeps the PR curve to select an F1-maximizing threshold, and translates the resulting confusion matrix into concrete business metrics (e.g., $8,000 per unplanned failure vs. $250 per inspection).
  5. Comprehensive Output: Automatically generates exploratory data analysis (EDA) visualizations, multicollinearity heatmaps, permutation importance charts, and serializes the final tuned model via joblib.
