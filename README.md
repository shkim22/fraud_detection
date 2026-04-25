# Credit Card Fraud Detection

Machine learning pipeline for detecting fraudulent credit card transactions.

## Exploratory Data Analysis

**Dataset:** 283,726 rows × 31 columns (after removing 1,081 duplicates from the raw 284,807-row source).

**Features:** `Time` (seconds since first transaction), `V1`–`V28` (PCA-transformed, anonymous), `Amount` — all numeric. `Class` is the binary target (0 = legitimate, 1 = fraud).

**Key findings:**

- **Severe class imbalance:** 99.83% legitimate vs 0.17% fraud (~577:1 ratio). Accuracy is a misleading metric — use ROC-AUC or precision-recall. Plan for SMOTE, undersampling, or class-weighted loss.
- **V14, V17, and V12 are the strongest fraud signals**, showing the largest distributional shift between classes (highest absolute correlation with `Class`).
- **V-features are mutually uncorrelated** (already PCA-transformed) — multicollinearity is not a concern; all 28 can be retained.
- **Amount and Time carry weak signal.** `Time` is near-zero correlated with fraud, indicating no time-of-day pattern. `Amount` has modest signal but heavy outliers.
- **Heavy-tailed distributions** on V27, V6, and V20 (>1.5% of rows exceed ±3σ) — use `RobustScaler` before feeding to distance-based or regularised models.
