"""
test_data_loader.py
----------------------------------------
Unit test for data loading and preprocessing module.
"""

from src.data_loader import load_and_prepare_data
import pandas as pd

def test_data_shape_and_normalization():
    df = load_and_prepare_data()
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ["Re", "Cd", "St", "Regime"])
    # Normalized columns must exist
    assert "Re_norm" in df.columns and "Cd_norm" in df.columns
    # Check for no NaN values
    assert not df.isnull().values.any()
