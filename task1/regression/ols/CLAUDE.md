# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
python3 ols.py            # Simple (single-feature) OLS regression
python3 ols_multiple.py   # Multiple OLS regression on weather dataset
```

Dependencies: `numpy`, `pandas`, `matplotlib`

## Architecture

This repo implements Ordinary Least Squares (OLS) linear regression **from scratch** (no scikit-learn). All math is done manually with NumPy.

- **`ols.py`** — Simple linear regression (one feature). Computes slope/intercept using the closed-form mean-based formula. Uses a synthetic dataset (`y = 3x + 5` with noise).
- **`ols_multiple.py`** — Multiple linear regression (n features). Computes coefficients using the **normal equation** `β = (XᵀX)⁻¹Xᵀy`. Loads `weatherHistory.csv` and predicts Apparent Temperature from Temperature, Humidity, Wind Speed, and Pressure.
- **`weatherHistory.csv`** — ~96k rows of hourly weather data. Numeric columns: Temperature (C), Apparent Temperature (C), Humidity, Wind Speed (km/h), Wind Bearing (degrees), Visibility (km), Pressure (millibars).
- **`.ipynb` files** — Jupyter notebook versions of the same scripts.

## Key patterns

- The intercept is handled by prepending a column of ones to the feature matrix (`X_b = [1 | X]`), so the coefficient vector is `[intercept, w1, w2, ..., wn]`.
- Evaluation uses MSE and RMSE computed manually.
- No train/test split is used — the model is fit and evaluated on the full dataset.
