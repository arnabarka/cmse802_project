"""
regression_model.py
----------------------------------------
Polynomial regression module for modeling the nonlinear
relationship between Reynolds number (Re) and flow quantities
(Cd, St) in the vortex-shedding dataset.

Produces:
- Regression fits for Cd–Re and St–Re
- RMSE and R² evaluation metrics
- Residual plots for model validation

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_and_prepare_data

def polynomial_regression(df: pd.DataFrame, save_dir="results", degree=3):
    """
    Fits polynomial regression models to Cd–Re and St–Re data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned vortex-shedding dataset.
    save_dir : str
        Directory to save output plots.
    degree : int
        Degree of the polynomial for fitting.

    Returns
    -------
    metrics : dict
        Dictionary of RMSE and R² for Cd and St.
    """
    os.makedirs(save_dir, exist_ok=True)
    metrics = {}

    # Aggregate mean for smoother fit
    agg = df.groupby(pd.cut(df.Re, bins=60)).mean(numeric_only=True)

    for ycol in ["St", "Cd"]:
        y = agg[ycol].values
        X = agg[["Re"]].values
        Xp = PolynomialFeatures(degree, include_bias=False).fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        yhat = model.predict(Xp)

        rmse = np.sqrt(mean_squared_error(y, yhat))
        r2 = r2_score(y, yhat)
        metrics[ycol] = {"RMSE": rmse, "R2": r2}

        # ---- Plot fitted curve ----
        plt.figure(figsize=(6, 4))
        plt.scatter(agg["Re"], y, s=10, label="True", alpha=0.6)
        plt.plot(agg["Re"], yhat, color="r", lw=2, label="Fitted")
        plt.xlabel("Re"); plt.ylabel(ycol)
        plt.title(f"{ycol} vs Re Fit (R²={r2:.3f}, RMSE={rmse:.4f})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{ycol.lower()}_fit.png"))
        plt.close()

        # ---- Plot residuals ----
        plt.figure(figsize=(6, 3))
        plt.scatter(agg["Re"], y - yhat, s=10)
        plt.axhline(0, color="r", ls="--")
        plt.xlabel("Re"); plt.ylabel(f"{ycol} Residuals")
        plt.title(f"Residual Plot for {ycol}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{ycol.lower()}_residuals.png"))
        plt.close()

    print("✅ Polynomial regression complete. Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: RMSE={v['RMSE']:.4f}, R²={v['R2']:.3f}")

    return metrics


if __name__ == "__main__":
    # Run quick test if executed directly
    df = load_and_prepare_data()
    metrics = polynomial_regression(df)
    print(metrics)
