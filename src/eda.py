"""
eda.py
----------------------------------------
Exploratory Data Analysis (EDA) module for the vortex-shedding dataset.

Performs correlation heatmaps, scatter plots (Re–Cd and Re–St),
and regime-based distributions to visually assess trends.

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import load_and_prepare_data

def run_eda(df: pd.DataFrame, save_dir="results"):
    """
    Conducts exploratory analysis and saves plots.
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned vortex-shedding dataset.
    save_dir : str
        Directory where figures will be stored.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid", context="talk")

    # 1️⃣ Correlation heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[["Re", "Cd", "St"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Flow Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "corr_heatmap.png"))
    plt.close()

    # 2️⃣ Re vs Cd scatter
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df.sample(2000), x="Re", y="Cd",
                    hue="Regime", s=10, alpha=0.7)
    plt.xscale("log")
    plt.title("Cd Variation with Reynolds Number")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "re_cd.png"))
    plt.close()

    # 3️⃣ Re vs St scatter
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df.sample(2000), x="Re", y="St",
                    hue="Regime", s=10, alpha=0.7)
    plt.xscale("log")
    plt.title("St Variation with Reynolds Number")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "re_st.png"))
    plt.close()

    print("✅ EDA completed — figures saved in", save_dir)


if __name__ == "__main__":
    # Run quick demo if executed directly
    df = load_and_prepare_data()
    run_eda(df)
