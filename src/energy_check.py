"""
energy_check.py
----------------------------------------
Numerical stability and energy-consistency validation module.

Ensures that:
- RMSE between consecutive model fits remains stable (ΔRMSE < 10⁻⁴)
- No artificial amplification (energy gain) occurs between iterations

This acts as a diagnostic tool confirming that regression and
classification stages behave consistently over multiple runs.

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import numpy as np
from src.metrics import compute_rmse, delta_rmse

def check_rmse_stability(y_true, y_pred_prev, y_pred_curr, tol=1e-4):
    """
    Verify ΔRMSE stability between two prediction sets.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred_prev : array-like
        Predictions from previous iteration.
    y_pred_curr : array-like
        Predictions from current iteration.
    tol : float
        Tolerance for RMSE stability (default: 1e-4).

    Returns
    -------
    passed : bool
        True if ΔRMSE < tol, else False.
    delta : float
        Absolute difference in RMSE.
    """
    rmse_prev = compute_rmse(y_true, y_pred_prev)
    rmse_curr = compute_rmse(y_true, y_pred_curr)
    passed, delta = delta_rmse(rmse_prev, rmse_curr, tol)
    if passed:
        print(f"✅ Energy check passed (ΔRMSE = {delta:.6e} < {tol})")
    else:
        print(f"⚠️ Energy check failed (ΔRMSE = {delta:.6e} ≥ {tol})")
    return passed, delta

def check_energy_amplification(y_series, threshold=1.05):
    """
    Detects potential 'energy amplification'—cases where model
    predictions show growing oscillations or unphysical deviations.

    Parameters
    ----------
    y_series : array-like
        Sequence of model-predicted output values.
    threshold : float
        Max allowed relative standard deviation ratio (default: 1.05).

    Returns
    -------
    stable : bool
        True if system is numerically stable.
    """
    std_ratio = np.std(y_series[1:]) / (np.std(y_series[:-1]) + 1e-8)
    stable = std_ratio < threshold
    if stable:
        print(f"✅ No energy amplification detected (ratio={std_ratio:.3f})")
    else:
        print(f"⚠️ Potential instability detected (ratio={std_ratio:.3f})")
    return stable
