# Tweedie Loss for Cost Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](notebooks/)

> **A practical, code-first guide to modeling zero-inflated, right-skewed cost data using the Tweedie distribution — from scikit-learn prototypes to distributed PySpark pipelines.**

---

## The Problem

Cost prediction in healthcare, insurance, and retail shares a common data challenge:

- **30–80% of observations are exactly zero** (no claim, no visit, no sale)
- **Non-zero values are continuous and heavily right-skewed** (routine $50 visits vs. $200K hospitalizations)
- **Standard loss functions fail**: MSE treats zeros and outliers equally; log-transforms introduce systematic under-forecasting

The **Tweedie distribution** solves this in a single model. It naturally generates zero-inflated, right-skewed data via a compound Poisson-Gamma process — no two-part models, no transformation bias, no hacks.

## What's in This Repository

```
tweedie-loss-cost-prediction/
├── notebooks/
│   ├── Notebook1_Healthcare_Cost_Tweedie.ipynb   # Healthcare cost prediction
│   ├── Notebook2_Sales_Prediction_Tweedie.ipynb   # Retail sales prediction
│   └── Notebook3_LargeScale_Tweedie_Comparison.ipynb  # 10M-row benchmark
├── src/
│   ├── __init__.py
│   └── tweedie_loss.py          # Reusable Tweedie loss (NumPy + PyTorch)
├── data/
│   └── README.md                # Data sources and download instructions
├── docs/
│   └── blog_post.md             # Companion Medium blog post
├── images/                      # Generated plots and figures
├── requirements.txt             # Python dependencies
├── requirements-full.txt        # All dependencies including PySpark & Dask
├── .gitignore
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Notebooks Overview

### Notebook 1: Healthcare Cost Prediction

**Goal**: Predict total healthcare expenditure per patient using the Medical Cost Personal Dataset.

| Section | What You'll Learn |
|---------|-------------------|
| **Part A** — Standard | EDA of zero-inflated medical costs, scikit-learn `TweedieRegressor` with power tuning, XGBoost with `reg:tweedie`, model comparison |
| **Part B** — Scalable | PyTorch custom `TweedieLoss` + feedforward NN (500K rows), PySpark `GeneralizedLinearRegression` with `family="tweedie"` (1M rows), Dask `ParallelPostFit` (2M rows) |

**Key finding**: XGBoost with Tweedie loss captures the non-linear interaction between smoking status, BMI, and age that linear GLMs miss, reducing MAE by 15–25%.

### Notebook 2: Sales Prediction

**Goal**: Predict daily revenue per product-store combination for a retail business.

| Section | What You'll Learn |
|---------|-------------------|
| **Part A** — Standard | Synthetic retail data with seasonal patterns, feature engineering (lag features, cyclical encodings), XGBoost + LightGBM with Tweedie objective, variance power tuning |
| **Part B** — Scalable | PyTorch with **embedding layers** for high-cardinality store/product IDs, PySpark distributed GLM, Dask parallel prediction |

**Key finding**: LightGBM with tuned power parameter ($p \approx 1.5$) and 7-day lag features provides the best accuracy-speed trade-off for tabular sales data.

### Notebook 3: Large-Scale Framework Comparison

**Goal**: Head-to-head benchmark of PyTorch, PySpark, and Dask on a single 10M-row synthetic Tweedie-distributed dataset.

| Metric | Scikit-learn | PyTorch | PySpark | Dask |
|--------|-------------|---------|---------|------|
| Data scale | 500K (subsample) | 8M (full) | 8M (full) | 2M (predict) |
| Training | Single-core GLM | Mini-batch NN | Distributed GLM | Subsample + parallel predict |
| Best for | Prototyping | Complex patterns + GPU | Cluster-scale | Out-of-core single machine |

**What's measured**: Training time, peak memory, prediction throughput (rows/sec), MAE, RMSE, Tweedie deviance, convergence curves.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tweedie-loss-cost-prediction.git
cd tweedie-loss-cost-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
# Core dependencies (Notebooks 1A, 2A)
pip install -r requirements.txt

# Full dependencies including PySpark and Dask (Notebooks 1B, 2B, 3)
pip install -r requirements-full.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook notebooks/
```

### 5. Use the Tweedie loss module in your own code

```python
from src.tweedie_loss import tweedie_deviance, get_pytorch_tweedie_loss, evaluate_predictions

# NumPy
score = tweedie_deviance(y_true, y_pred, power=1.5)

# PyTorch
criterion = get_pytorch_tweedie_loss(power=1.5)
loss = criterion(model(X), y)

# Evaluation
evaluate_predictions(y_true, y_pred, power=1.5, label="My Model")
```

## Theoretical Background

The Tweedie distribution with power parameter $p \in (1, 2)$ is a **compound Poisson-Gamma** process:

$$Y = \sum_{i=1}^{N} X_i, \quad N \sim \text{Poisson}(\lambda), \quad X_i \sim \text{Gamma}(\alpha, \gamma)$$

- When $N = 0$: $Y = 0$ (no event occurred)
- When $N > 0$: $Y$ is a sum of Gamma-distributed values (right-skewed positive)

The **Tweedie deviance** (loss function) is:

$$d(y, \mu) = 2 \left[ \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \cdot \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right]$$

Special cases: $p=0$ (Normal), $p=1$ (Poisson), $p=2$ (Gamma), $p=3$ (Inverse Gaussian).

## Framework Cheat Sheet

| Your Situation | Recommended Approach |
|---|---|
| Prototyping, < 1M rows | `sklearn.linear_model.TweedieRegressor` |
| Tabular data, < 10M rows | XGBoost / LightGBM with `objective='tweedie'` |
| Non-linear patterns, GPU available | PyTorch with custom `TweedieLoss` |
| High-cardinality categoricals | PyTorch with embedding layers |
| Cluster-scale (> 50M rows) | PySpark `GeneralizedLinearRegression(family="tweedie")` |
| Out-of-core, single machine | Dask with `ParallelPostFit` wrapper |
| Sequential cost data (time series) | LSTM / Transformer with `TweedieLoss` |

## References

- Jorgensen, B. (1987). Exponential dispersion models. *Journal of the Royal Statistical Society, Series B*, 49(2), 127–162.
- Delong, L., Lindholm, M., & Wuthrich, M. V. (2021). Making Tweedie's compound Poisson model more accessible. *European Actuarial Journal*, 11, 185–226.
- Denuit, M., Charpentier, A., & Trufin, J. (2021). Autocalibration and Tweedie-dominance for insurance pricing with machine learning. arXiv:2103.03635.
- More, S. (2022). Identifying and overcoming transformation bias in forecasting models. arXiv:2208.12264.
- So, B., & Valdez, E. A. (2025). Zero-inflated Tweedie boosted trees with CatBoost for insurance loss analytics. *Applied Soft Computing*, 169, 113226.
- Manna, A. et al. (2025). Distribution-free inference for LightGBM and GLM with Tweedie loss. arXiv:2507.06921.

## Blog Post

The companion blog post is available at on [Medium](https://medium.com/@juankehoe).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


