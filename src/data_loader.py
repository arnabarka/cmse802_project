"""
data_loader.py

This file loads the vortex cylinder dataset.

It performs the following steps:
1. Check if the dataset file exists.
2. Read the CSV file.
3. Verify that the required columns Re, St and Cd are present.
4. Remove rows with missing values.
5. Return a clean dataframe.
"""

import pandas as pd
import os


def load_data(path):
    """
    Load the dataset from the given file path.

    Parameters
    path : str
        Path to the vortex data file.

    Returns
    df : pandas DataFrame
        Dataframe containing Re, St and Cd after cleaning.

    Raises
    FileNotFoundError
        When the file path is not found.
    ValueError
        When required columns are missing.
    """

    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    # Load CSV into dataframe
    df = pd.read_csv(path)

    # Ensure required columns exist
    required_cols = {"Re", "St", "Cd"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    # Remove missing values and clean index
    df = df.dropna().reset_index(drop=True)

    return df
