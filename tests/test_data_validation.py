# tests/test_data_validity.py
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_and_prepare_data


class TestDataValidity(unittest.TestCase):
    """Unit tests for data loading and preprocessing."""

    def setUp(self):
        # Load small representative dataset
        self.df = load_and_prepare_data()

    def test_dataframe_structure(self):
        """Check expected columns and no missing values."""
        expected_cols = {"Re", "Cd", "St"}
        self.assertTrue(expected_cols.issubset(self.df.columns))
        self.assertFalse(self.df.isnull().any().any())

    def test_physical_ranges(self):
        """Ensure Re, Cd, St values fall in realistic physical ranges."""
        self.assertTrue(self.df["Re"].between(1e2, 1e6).all())
        self.assertTrue(self.df["Cd"].between(0.8, 2.0).all())
        self.assertTrue(self.df["St"].between(0.15, 0.25).all())


if __name__ == "__main__":
    unittest.main()
