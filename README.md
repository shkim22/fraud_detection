# Credit Card Fraud Detection

> End-to-end ML pipeline that catches fraudulent transactions with **0.9821 ROC-AUC** on a 284k-row, 599:1 imbalanced dataset — with an interactive Streamlit portfolio app and full CI/CD.

🔴 **[Live Demo →](https://frauddetection-cmhlhdlis68swc83btd7sm.streamlit.app/)** &nbsp;|&nbsp; 📓 **[EDA Notebook](notebooks/eda.ipynb)** &nbsp;|&nbsp; ⚙️ **[CI Status](https://github.com/shkim22/fraud_detection/actions)**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Results](#3-results)
4. [Tech Stack](#4-tech-stack)
5. [Setup & Installation](#5-setup--installation)
6. [How to Run](#6-how-to-run)
7. [Feature Engineering](#7-feature-engineering)
8. [Key Decisions & Lessons](#8-key-decisions--lessons)
9. [File Structure](#9-file-structure)

---

## 1. Project Overview

### The Problem

Credit card fraud costs the global economy **$32 billion per year**. A card network processes thousands of transactions per second — a fraud model must make a binary decision (approve / flag) in under 100 ms, with no human in the loop for most transactions. Getting it wrong in either direction is expensive: missed fraud costs money; too many false positives erode cardholder trust.

### Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 (283,726 after quality cleaning) |
| Fraud rate | 0.17% — 473 fraud in 283,253 legitimate |
| Imbalance ratio | ~599 : 1 |
| Features | 28 PCA-anonymised components (V1–V28) + `Amount` + `Time` |
| Collection window | 48 hours, European cardholders, September 2013 |

The V-features are the result of a PCA transformation applied by the original researchers to protect cardholder privacy. Their real-world meaning is unknown, but their statistical relationship with fraud is exploitable.

### What the Model Outputs

A probability score **P(fraud | transaction)** in [0, 1]. A downstream threshold (tuned separately from model training) converts this to an approve/flag decision based on the ops team's daily review capacity.

### Who Is the End User

An automated transaction decisioning system at a card network or issuing bank. Every flagged transaction is reviewed by a fraud analyst; the model's recall governs how many real frauds reach that queue, and its precision governs the analyst's workload.

### Key Design Decision

**ROC-AUC is the training metric, not F1.** F1 bakes in a threshold choice that belongs to the business. AUC is threshold-free and measures ranking quality — how well the model separates fraud from legit across all operating points. The optimal threshold is set separately, after training, by the ops team.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                         │
│                                                                     │
│  creditcard.csv  ──►  Quality Gate  ──►  cleaned.csv               │
│  (Kaggle raw)         (great-exp)        (283k rows)               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  FEATURE LAYER                                                      │
│                                                                     │
│  cleaned.csv  ──►  Feature Engineering  ──►  Feature Selection      │
│                    (+12 new features)        (corr + var filter)    │
│                         │                                           │
│                         ▼                                           │
│                    features.csv  (42 cols)                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  MODELLING LAYER                                                    │
│                                                                     │
│  features.csv                                                       │
│       │                                                             │
│       ├──► Train / Test Split (80/20, stratified)                   │
│       │                                                             │
│       ├──► StandardScaler + SMOTE (inside CV folds only)           │
│       │                                                             │
│       ├──► Model Comparison ──► XGBoost · RandomForest · LightGBM  │
│       │         (5-fold CV)                                         │
│       │                                                             │
│       └──► Optuna Tuning (30 TPE trials) ──► tuned_model.pkl       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  SERVING LAYER                                                      │
│                                                                     │
│  tuned_model.pkl ──►  Streamlit App  (portfolio + live prediction)  │
│                   └►  MLflow         (experiment tracking)          │
│                   └►  Docker         (containerised deployment)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Results

### Model Comparison

All models trained on the same 80/20 stratified split with `StandardScaler → SMOTE` inside each CV fold to prevent leakage.

| Model | CV AUC (5-fold) | Test AUC | F1 | Precision | Recall | Train (s) |
|---|---|---|---|---|---|---|
| Logistic Regression *(baseline)* | 0.9622 | 0.9672 | 0.7811 | 0.8919 | 0.6947 | 4 |
| Random Forest | 0.9698 | 0.9728 | 0.8101 | 0.9312 | 0.7183 | 52 |
| LightGBM | 0.9748 | 0.9788 | 0.8453 | 0.9198 | 0.7823 | 13 |
| XGBoost | 0.9768 | 0.9808 | 0.8427 | 0.9036 | 0.7895 | 38 |
| **XGBoost (Tuned) ✓** | **0.9845** | **0.9821** | **0.8065** | **0.8242** | **0.7895** | 61 |

**Improvement over baseline: +0.0149 AUC (+1.49 percentage points)**

### Confusion Matrix (Tuned XGBoost — held-out test set, 56,746 transactions)

```
                  Predicted Legit   Predicted Fraud
Actual Legit          56,635              16         ← 16 false alarms
Actual Fraud              20              75         ← 75 of 95 frauds caught
```

- **Recall 0.79** — catches 75 of the 95 fraud cases in the test set
- **Precision 0.82** — 82% of flagged transactions are genuine fraud
- **False positive rate 0.028%** — ops team reviews 16 legitimate transactions per 56k

### Why XGBoost Won

LightGBM matched AUC (0.979) and trained 3× faster. XGBoost was chosen as the winner for two reasons:

1. **SHAP ecosystem** — richer tooling for explaining individual decisions to fraud analysts, a regulatory requirement in many jurisdictions.
2. **Deployment precedent** — XGBoost has wider adoption in financial-services ML infrastructure and more battle-tested serving libraries.

The 30-trial Optuna search pushed CV AUC from 0.9768 → **0.9845** by reducing `learning_rate` (0.1 → 0.019), increasing `max_depth` (6 → 7), and applying heavier regularisation (`gamma=3.98`, `reg_lambda=0.093`).

---

## 4. Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.9** | Core language |
| **pandas** | Data loading, manipulation, EDA |
| **scikit-learn** | Pipelines, scalers, metrics, baseline models |
| **XGBoost** | Winner model — gradient boosting with L1/L2 regularisation |
| **LightGBM** | Candidate model — leaf-wise boosting, 3-5× faster training |
| **imbalanced-learn** | SMOTE oversampling inside CV pipelines |
| **Optuna** | Bayesian hyperparameter search (TPE sampler, 30 trials) |
| **MLflow** | Experiment tracking, parameter and metric logging |
| **great-expectations** | Data quality gate — schema, null rates, value ranges |
| **Streamlit** | Interactive portfolio app — EDA, model comparison, live prediction |
| **Plotly** | Interactive charts in the Streamlit app |
| **pytest** | 52-test suite across data quality, features, and model |
| **ruff** | Fast Python linter (E, F, W rules) |
| **Docker** | Containerised deployment |
| **GitHub Actions** | CI/CD — test + lint on every push and PR |

---

## 5. Setup & Installation

### Prerequisites

- Python 3.9 or higher
- The [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) downloaded to `data/creditcard.csv`

### macOS (Apple Silicon)

```bash
# 1. Clone the repo
git clone https://github.com/shkim22/fraud_detection.git
cd my-ml-project

# 2. Create and activate a conda environment
conda create -n fraud-detection python=3.9
conda activate fraud-detection

# 3. Install OpenMP (required by XGBoost on macOS)
#    Use conda to get the arm64 build and avoid Homebrew architecture mismatches
conda install -c conda-forge libomp

# 4. Install Python dependencies
pip install -r requirements.txt
pip install -e .
```

### Linux / CI

```bash
git clone https://github.com/shkim22/fraud_detection.git
cd my-ml-project

# Install OpenMP runtime (required by XGBoost and LightGBM on Linux)
sudo apt-get install -y libgomp1

pip install -r requirements.txt
pip install -e .
```

### Windows

```bash
git clone https://github.com/shkim22/fraud_detection.git
cd my-ml-project
pip install -r requirements.txt
pip install -e .
# XGBoost on Windows requires the Visual C++ Redistributable:
# https://support.microsoft.com/en-us/help/2977003
```

---

## 6. How to Run

### Full Training Pipeline

Run the steps in order from the project root. Each step reads from and writes to `data/` and `models/`.

```bash
# Step 1 — Validate and clean raw data
python src/data/cleaner.py

# Step 2 — Engineer and select features
python src/features/engineering.py

# Step 3 — Train baseline model (Logistic Regression)
python src/models/baseline.py

# Step 4 — Compare XGBoost, LightGBM, Random Forest
python src/models/train_models.py

# Step 5 — Hyperparameter tuning (30 Optuna trials, ~5 min)
python src/models/tuning.py
```

MLflow logs every run automatically. View the experiment UI:

```bash
mlflow ui
# Open http://localhost:5000
```

### Streamlit Portfolio App

```bash
# Generate app data from the trained models (run once after training)
python app/generate_app_data.py

# Launch the app
streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

### Docker

```bash
# Build the image
docker build -t my-ml-project .

# Run the container
docker run -p 8501:8501 my-ml-project

# Or use Compose (mounts data/ and models/ as live volumes)
docker compose up --build
```

### Tests

```bash
# Run the full test suite (52 tests)
pytest tests/ -v

# Run a specific file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

> **CI note:** `data/` and `models/` are gitignored. When the real artifacts are absent (e.g. in CI), `tests/conftest.py` automatically generates synthetic data and trains a minimal model so all 52 tests still pass.

---

## 7. Feature Engineering

13 features were added on top of the 29 original columns (`Time`, `Amount`, `V1`–`V28`). A correlation filter (threshold 0.95) and variance filter were then applied, leaving **42 features** in the final model.

### Engineered Features

| Feature | Category | Rationale | XGB Importance |
|---|---|---|---|
| `v14_x_v12` | Interaction | V14 × V12 product: large only when *both* are simultaneously extreme — a specific fraud fingerprint | **30.2%** |
| `V14` | Raw PCA | Strongest single-feature correlation with fraud (r = −0.30 from EDA) | **21.5%** |
| `V10` | Raw PCA | Secondary PCA fraud signal, orthogonal to V14 | **11.2%** |
| `top_fraud_signal` | Statistical | Mean of V14, V17, V12 — averages noise while retaining the three strongest directional indicators | **9.3%** |
| `v17_x_v12` | Interaction | Covers cases where V14 is near-zero but V17 and V12 are jointly extreme | **1.9%** |
| `v_abs_max` | Statistical | Max absolute PCA component per row — captures localised anomalies invisible to the row mean | **1.6%** |
| `amount_small` | Domain | Flag: Amount < $5 — classic card-testing pattern (stolen card probed with micro-transaction) | **1.5%** |
| `is_nighttime` | Domain | Flag: 10 PM – 6 AM — fraudsters exploit reduced overnight monitoring | **1.3%** |
| `amount_rounded` | Domain | Flag: Amount ends in .00 — fraud scripts often use round dollar amounts | **1.1%** |
| `amount_large` | Domain | Flag: Amount > $500 — common manual-review threshold at card networks | **0.9%** |
| `amount_log_x_v14` | Interaction | High-value transaction *and* primary PCA fraud signal — compound high-risk pattern | **0.9%** |
| `amount_log` | Domain | log(1 + Amount) — compresses right-skewed distribution for distance-based models | — |
| `v_mean` | Statistical | Row-wise mean across V1–V28 — large non-zero mean indicates distance from PCA centroid | — |

---

## 8. Key Decisions & Lessons

**✅ SMOTE inside CV folds, not before splitting.**
Applying SMOTE to the full training set before cross-validation leaks synthetic fraud samples into the validation folds, inflating recall by 8–12 points. Using `ImbPipeline` ensures oversampling happens exclusively inside each training fold. This is the single most common mistake in imbalanced-learning tutorials.

**✅ Optuna TPE over grid/random search.**
A 5×5 random grid (25 evaluations) returned a best CV AUC of 0.9803. Thirty TPE trials reached 0.9845 — a larger gain — because TPE builds a probabilistic model of the search space and concentrates sampling in promising regions. The winning configuration (`learning_rate=0.019`, `max_depth=7`, `gamma=3.98`) would not have appeared in a coarse manual grid.

**✅ Feature interactions outperformed raw features.**
`v14_x_v12` (30.2% importance) ranked above all 28 raw PCA components. Domain-guided feature construction — exploiting the EDA finding that V14, V17, and V12 are the three strongest fraud signals — paid off more than feature selection alone would have.

**❌ First attempt: SMOTE before splitting caused silent leakage.**
In the initial prototype, SMOTE was applied to the entire dataset before the train/test split. Test recall appeared to be 0.91. Re-running with the correct `ImbPipeline` approach dropped it to 0.79 — still good, but the earlier number was 15% inflated. This is easy to miss because no error is raised and the metrics look *better*, not worse.

**✅ LightGBM is the better production choice (even though XGBoost won here).**
LightGBM matched XGBoost at 0.979 AUC and trained 3× faster. For a system that retrains daily on new transaction batches, that training-time difference compounds significantly. XGBoost was chosen here for SHAP explainability and deployment familiarity, but in a real production system the decision would lean toward LightGBM unless explainability is a hard regulatory requirement.

---

## 9. File Structure

```
my-ml-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Test (pytest) + Lint (ruff) on push/PR
│
├── app/
│   ├── generate_app_data.py    # One-time script: builds predictions.csv + model_results.json
│   └── streamlit_app.py        # 4-page portfolio app (Overview, EDA, Models, Process)
│
├── data/                       # gitignored — download Kaggle CSV, rest is generated
│   ├── creditcard.csv          # Raw Kaggle dataset (not committed)
│   ├── cleaned.csv             # Output of cleaner.py
│   ├── features.csv            # Output of engineering.py
│   ├── predictions.csv         # Output of generate_app_data.py
│   └── model_results.json      # Output of generate_app_data.py
│
├── models/                     # gitignored — generated by training scripts
│   ├── baseline.pkl            # Logistic Regression pipeline
│   ├── xgboost.pkl             # Default XGBoost pipeline
│   ├── best_params.json        # Optuna best hyperparameters
│   └── tuned_model.pkl         # Final tuned XGBoost pipeline
│
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis (7 sections, plotted outputs)
│
├── src/
│   ├── data/
│   │   ├── loader.py           # CSV loader + schema reporter
│   │   ├── quality.py          # great-expectations quality gate
│   │   └── cleaner.py          # Cleaning pipeline (dedup, type coercion)
│   ├── features/
│   │   └── engineering.py      # create_features() + select_features()
│   └── models/
│       ├── baseline.py         # Logistic Regression baseline
│       ├── train_models.py     # Multi-model comparison (XGB, RF, LGBM)
│       └── tuning.py           # Optuna hyperparameter search
│
├── tests/
│   ├── conftest.py             # Synthetic artifact generator for CI
│   ├── test_data_quality.py    # 12 tests — quality gate pass/fail cases
│   ├── test_features.py        # 23 tests — feature correctness and ranges
│   └── test_model.py           # 17 tests — predictions, AUC, directional sanity
│
├── Dockerfile                  # python:3.9-slim, port 8501
├── docker-compose.yml          # Mounts data/ and models/ as volumes
├── requirements.txt            # Python dependencies
├── setup.py                    # Editable install for src/ imports
└── README.md                   # This file
```

---

## Reproducing the Results

```bash
# 1. Clone and install
git clone https://github.com/shkim22/fraud_detection.git
cd my-ml-project
pip install -r requirements.txt && pip install -e .

# 2. Download the dataset
#    Place creditcard.csv in data/ from:
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Run the full pipeline
python src/data/cleaner.py
python src/features/engineering.py
python src/models/baseline.py
python src/models/train_models.py
python src/models/tuning.py

# 4. Launch the app
python app/generate_app_data.py
streamlit run app/streamlit_app.py

# 5. Run tests
pytest tests/ -v
```

Expected final output from `tuning.py`:
```
Test-set metrics:
  ROC-AUC      0.9821
  F1           0.8065
  Precision    0.8242
  Recall       0.7895
```

---

*Built by [Hyun Gil Kim](https://github.com/stephenhgkim) · Data: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group)*
