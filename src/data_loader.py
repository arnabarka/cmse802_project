"""
data_loader.py
----------------------------------------
Module for data loading, cleaning, and normalization
for the vortex-shedding simulation dataset.

It can either generate a synthetic dataset (if not found)
or load the existing CSV, remove outliers, and normalize
the key variables (Re, Cd, St) for modeling.

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_dataset(file_path="data/cylinder_vortex_full.csv", N=40000, random_state=42):
    """Generate synthetic Re–Cd–St dataset and save as CSV."""
    np.random.seed(random_state)
    Re = np.linspace(10, 40000, N)
    Cd_true = 1.2 + 24/Re + 6/(1 + np.sqrt(Re))
    St_true = 0.18 + 0.02 * np.tanh((Re - 200)/4000)
    Cd = Cd_true + np.random.normal(0, 0.03, N)
    St = St_true + np.random.normal(0, 0.005, N)
    regime = np.where(Re < 200, "Laminar",
              np.where(Re < 3000, "Transitional", "Turbulent"))
    df = pd.DataFrame({"Re": Re, "Cd": Cd, "St": St, "Regime": regime})

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"✅ Dataset generated: {file_path} ({len(df)} samples)")
    return df

def load_and_prepare_data(file_path="data/cylinder_vortex_full.csv",
                          normalize=True, remove_outliers=True):
    """
    Load, clean, and preprocess vortex-shedding data.
    - Removes NaNs
    - Optionally removes outliers via z-score
    - Optionally normalizes Re, Cd, St
    """
    if not os.path.exists(file_path):
        print("⚠️ Dataset not found — generating synthetic data...")
        df = generate_dataset(file_path)
    else:
        df = pd.read_csv(file_path)

    df = df.dropna()

    # Outlier removal (z-score > 3)
    if remove_outliers:
        z = np.abs((df[["Re", "Cd", "St"]] - df[["Re", "Cd", "St"]].mean()) /
                   df[["Re", "Cd", "St"]].std())
        df = df[(z < 3).all(axis=1)]

    # Normalization
    if normalize:
        scaler = StandardScaler()
        df[["Re_norm", "Cd_norm", "St_norm"]] = scaler.fit_transform(df[["Re", "Cd", "St"]])

    print(f"✅ Data loaded and preprocessed: {df.shape[0]} valid samples.")
    return df

if __name__ == "__main__":
    # Quick self-test
    df = load_and_prepare_data()
    print(df.head())
