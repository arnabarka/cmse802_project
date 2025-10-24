"""
metrics.py
----------------------------------------
Utility functions for computing evaluation metrics
used across regression and classification modules.

Includes:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- ΔRMSE (Iterative Stability Check)

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def compute_r2(y_true, y_pred):
    """Compute R² score."""
    return r2_score(y_true, y_pred)

def delta_rmse(prev_rmse, curr_rmse, tol=1e-4):
    """
    Compute RMSE stability difference to ensure numerical consistency.
    Returns True if ΔRMSE < tol.
    """
    delta = abs(curr_rmse - prev_rmse)
    return delta < tol, delta

def summarize_metrics(y_true, y_pred):
    """
    Convenience wrapper returning all metrics as a dictionary.
    """
    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    r2 = compute_r2(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}
