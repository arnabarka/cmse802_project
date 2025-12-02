"""
visualization.py

This file creates all plots for the vortex cylinder analysis.

It contains four groups of plots.
1. Basic scatter plots for Re, St and Cd.
2. Predicted versus actual plots for checking model accuracy.
3. A combined comparison plot showing theory, experiment and all machine learning models.
4. A bar plot of best GBR hyperparameters.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



# 1. Scatter plots

def plot_cd_vs_re(df, save_path):
    """
    Plot Cd versus Re using a log scale for Re.
    Save the figure at the specified path.
    """
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df["Re"], y=df["Cd"], s=18, alpha=0.35)
    plt.xscale("log")
    plt.xlabel("Re log scale")
    plt.ylabel("Cd")
    plt.title("Cd vs Re")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cd_vs_st(df, save_path):
    """
    Plot Cd versus St.
    Save the figure at the specified path.
    """
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df["St"], y=df["Cd"], s=18, alpha=0.35)
    plt.xlabel("St")
    plt.ylabel("Cd")
    plt.title("Cd vs St")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_re_vs_st(df, save_path):
    """
    Plot Re versus St using log scale for Re.
    Save the figure at the specified path.
    """
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df["Re"], y=df["St"], s=18, alpha=0.35)
    plt.xscale("log")
    plt.xlabel("Re log scale")
    plt.ylabel("St")
    plt.title("Re vs St")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# 2. Predicted versus actual plots

def plot_pred_vs_actual(df, pred_col, save_path):
    """
    Create a scatter plot comparing actual Cd and predicted Cd.
    A diagonal reference line is drawn to indicate perfect prediction.
    """
    plt.figure(figsize=(5.5, 5.5))
    sns.scatterplot(x=df["Cd"], y=df[pred_col], s=18, alpha=0.35)

    mn = min(df["Cd"].min(), df[pred_col].min())
    mx = max(df["Cd"].max(), df[pred_col].max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.2)

    plt.xlabel("Actual Cd")
    plt.ylabel(f"Predicted Cd {pred_col}")
    plt.title(f"Predicted vs Actual {pred_col}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# 3. Combined comparison plot

def plot_combined_cd_re(df, poly_results, gbr_re_model, gbr_st_model, save_path):
    """
    Create a combined Cd versus Re plot including the following items.
    1. Experimental data from the dataset.
    2. A simple theoretical Cd curve for reference.
    3. Polynomial regression curves for all fitted degrees.
    4. A GBR model that uses only Re.
    5. A GBR model that uses both Re and St. The St values are smoothed using
       moving median smoothing,applied through binning followed by linear interpolation.
       This smoothing reduces noise and produces a stable St versus Re trend.

    All curves are plotted on the same figure to show how each method compares.
    """

    plt.figure(figsize=(10, 6))

    # 1. Experimental data
    plt.scatter(df["Re"], df["Cd"], color="black", s=10, alpha=0.30, label="Experimental Data")

    # 2. Theoretical Cd curve
    Re_grid = np.linspace(20, 5000, 500)
    Cd_theoretical = 1.2 - 0.1 * np.log10(Re_grid)
    Cd_theoretical = np.clip(Cd_theoretical, 0.3, 1.2)
    plt.plot(Re_grid, Cd_theoretical, color="blue", linewidth=2.2, label="Theoretical Cd")

    # 3. Polynomial regression curves
    colors = ["red", "green", "orange", "purple", "brown"]
    for i, deg in enumerate(poly_results.keys()):
        poly = PolynomialFeatures(degree=deg)
        model = LinearRegression()
        model.fit(poly.fit_transform(df[["Re"]]), df["Cd"])
        Cd_poly_curve = model.predict(poly.transform(Re_grid.reshape(-1, 1)))
        plt.plot(Re_grid, Cd_poly_curve, linestyle="--", linewidth=1.3,
                 color=colors[i], alpha=0.8, label=f"Polynomial degree {deg}")

    # 4. GBR using Re only
    Cd_gbr_re_curve = gbr_re_model.predict(Re_grid.reshape(-1, 1))
    plt.plot(Re_grid, Cd_gbr_re_curve, color="magenta", linewidth=2,
             label="GBR using Re")

    # 5. Smoothing St for the GBR model using Re and St

    # Step 1. Define number of bins
    bin_count = 60
    bins = np.linspace(df["Re"].min(), df["Re"].max(), bin_count)

    # Step 2. Assign each Re to a bin index
    df["Re_bin"] = np.digitize(df["Re"], bins) - 1

    # Step 3. Keep only valid bins
    df_valid = df[df["Re_bin"].between(0, bin_count - 2)]

    # Step 4. Compute median St in each bin
    med_st = df_valid.groupby("Re_bin")["St"].median()

    # Step 5. Compute bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Step 6. Interpolate median St values to create smooth curve
    median_st = med_st.reindex(range(bin_count - 1)).interpolate().bfill().ffill().values

    # Step 7. Create smooth St curve for Re_grid
    St_curve = np.interp(Re_grid, bin_centers, median_st)

    # Step 8. Predict using GBR that uses both Re and St
    X_grid_st = np.column_stack([Re_grid, St_curve])
    Cd_gbr_st_curve = gbr_st_model.predict(X_grid_st)
    # This code snippet is generated by Open AI chatgpt version 5.1 

    plt.plot(Re_grid, Cd_gbr_st_curve, color="cyan", linewidth=2,
             label="GBR using Re and St")

    plt.xscale("log")
    plt.xlabel("Re log scale")
    plt.ylabel("Cd")
    plt.title("Combined Cd vs Re comparison")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# 4. Bar plot for best parameters

def plot_best_params(best_params, save_path):
    """
    Create a bar plot showing the best GBR hyperparameters.
    Save the plot at the specified location.
    """
    keys = list(best_params.keys())
    values = list(best_params.values())

    plt.figure(figsize=(6, 4))
    sns.barplot(x=keys, y=values, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Best Hyperparameters for GBR")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
