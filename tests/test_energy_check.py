"""
test_energy_check.py
----------------------------------------
Ensures ΔRMSE stability and no energy amplification.
"""

import numpy as np
from src.energy_check import check_rmse_stability, check_energy_amplification

def test_rmse_stability_and_energy():
    y_true = np.linspace(0, 1, 100)
    y_pred_prev = y_true + np.random.normal(0, 0.005, 100)
    y_pred_curr = y_true + np.random.normal(0, 0.004, 100)
    passed, delta = check_rmse_stability(y_true, y_pred_prev, y_pred_curr)
    assert passed, f"ΔRMSE stability failed (Δ={delta:.6e})"
    stable = check_energy_amplification(y_true)
    assert stable, "Energy amplification detected"
