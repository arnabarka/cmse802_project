"""
test_regression_model.py
----------------------------------------
Validates polynomial regression fitting accuracy.
"""

from src.data_loader import load_and_prepare_data
from src.regression_model import polynomial_regression

def test_regression_rmse_thresholds():
    df = load_and_prepare_data()
    metrics = polynomial_regression(df)
    assert metrics["Cd"]["RMSE"] < 0.05, "Cd RMSE above threshold"
    assert metrics["St"]["RMSE"] < 0.005, "St RMSE above threshold"
