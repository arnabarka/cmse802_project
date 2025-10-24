"""
visualization.py
----------------------------------------
Visualization utilities for 2D and 3D data exploration,
model diagnostics, and feature analysis in vortex-shedding studies.

Includes:
- 3D scatter and surface plots (Re–Cd–St)
- Residual visualization
- Correlation matrix and feature-importance plots

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_3d_scatter(df, save_dir="results"):
    """3D scatter visualization of Re–Cd–St relationships."""
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    sample = df.sample(3000)
    p = ax.scatter(sample["Re"], sample["Cd"], sample["St"],
                   c=sample["Re"], cmap="viridis", s=5)
    fig.colorbar(p, ax=ax, shrink=0.6, label="Reynolds Number")
    ax.set_xlabel("Re"); ax.set_ylabel("Cd"); ax.set_zlabel("St")
    plt.title("3D Scatter: Re–Cd–St")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3d_surface1.png"))
    plt.close()
    print("✅ 3D scatter saved.")

def plot_3d_surface(df, save_dir="results"):
    """Interpolated 3D surface plot of flow field."""
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    sample = df.sample(3000)
    ax.plot_trisurf(sample["Re"], sample["Cd"], sample["St"],
                    cmap="plasma", linewidth=0)
    ax.set_xlabel("Re"); ax.set_ylabel("Cd"); ax.set_zlabel("St")
    plt.title("3D Surface Plot (Interpolated Flow Field)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3d_surface2.png"))
    plt.close()
    print("✅ 3D surface saved.")

def plot_residuals(Re, y_true, y_pred, ycol, save_dir="results"):
    """Residual plot for polynomial regression validation."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.scatter(Re, y_true - y_pred, s=10)
    plt.axhline(0, color="r", ls="--")
    plt.xlabel("Re"); plt.ylabel(f"{ycol} Residuals")
    plt.title(f"Residual Plot for {ycol}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{ycol.lower()}_residuals.png"))
    plt.close()

def correlation_matrix(df, save_dir="results"):
    """Correlation matrix for Re–Cd–St."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[["Re", "Cd", "St"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Flow Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "corr_heatmap.png"))
    plt.close()

def plot_feature_importance(importances, feature_names, save_dir="results"):
    """Bar plot of Random Forest feature importances."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 3))
    sns.barplot(x=importances, y=feature_names, palette="mako")
    plt.title("Feature Importance for Regime Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"))
    plt.close()
