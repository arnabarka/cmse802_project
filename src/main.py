"""
main.py

This file runs the full analysis pipeline for the vortex cylinder dataset.

The pipeline contains five major stages.
1. Load the dataset and prepare folders.
2. Run exploratory data analysis and save plots and summary statistics.
3. Train polynomial regression models and select the best degree.
4. Train two Gradient Boosting Regressor models,
   one using only Re and one using both Re and St.
5. Generate all visual plots and launch an interactive predictor
   for live Cd prediction from user input values.

All results are saved inside the results folder.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Import modules
from src.data_loader import load_data
from src.eda import run_eda
from src.polynomial_regression import polynomial_regression
from src.gbr_model import train_gbr
from src.visualization import (
    plot_cd_vs_re,
    plot_cd_vs_st,
    plot_re_vs_st,
    plot_combined_cd_re,
    plot_pred_vs_actual,
)



# Print a formatted table
def print_table(title, headers, rows):
    """
    Print a table with title, column headers and rows.
    """
    print("\n" + title)
    print("-" * len(title))

    header_line = "  ".join(f"{h:>12}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        print("  ".join(f"{str(v):>12}" for v in row))
    print()



# Print hyperparameters
def print_params(title, params):
    """
    Print hyperparameters in a readable format.
    """
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in params.items():
        print(f"{k:<15}  {v}")
    print()



# Main execution pipeline
def run():

    # 1. Setup folders and file paths
    PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA = os.path.join(PROJECT, "data", "vortex_data.csv")
    RESULTS = os.path.join(PROJECT, "results")
    os.makedirs(RESULTS, exist_ok=True)

    # 2. Load dataset
    df = load_data(DATA)
    print("\nLoaded dataset with", len(df), "samples")

    # 3. Run exploratory data analysis
    run_eda(df, os.path.join(RESULTS, "eda"))
    print("\nEDA completed.\n")

    # 4. Polynomial regression
    best_model, best_poly_obj, poly_metrics = polynomial_regression(df)

    rows = []
    for deg, m in poly_metrics.items():
        rows.append([
            deg,
            round(m["train_r2"], 4),
            round(m["test_r2"], 4),
            round(m["train_rmse"], 4),
            round(m["test_rmse"], 4),
            round(m["train_mae"], 4),
            round(m["test_mae"], 4),
        ])

    print_table(
        "Polynomial Regression Comparison Re to Cd",
        ["Degree", "Train R2", "Test R2", "Train RMSE", "Test RMSE", "Train MAE", "Test MAE"],
        rows
    )

    # Best degree
    best_deg = max(poly_metrics, key=lambda d: poly_metrics[d]["test_r2"])
    print(f"\nBest Polynomial Degree = {best_deg}")

    df["Cd_poly"] = best_model.predict(best_poly_obj.transform(df[["Re"]]))

    # 5. GBR model using only Re
    gbr_re_model, gbr_re_metrics, gbr_re_params = train_gbr(df, use_st=False)
    print_params("Best GBR Re Hyperparameters", gbr_re_params)

    df["Cd_gbr_re"] = gbr_re_model.predict(df[["Re"]])

    print_table(
        "GBR Re Performance Metrics",
        ["Metric", "Value"],
        [
            ["Train R2", round(gbr_re_metrics["train_r2"], 4)],
            ["Test R2", round(gbr_re_metrics["test_r2"], 4)],
            ["Train RMSE", round(gbr_re_metrics["train_rmse"], 4)],
            ["Test RMSE", round(gbr_re_metrics["test_rmse"], 4)],
            ["Train MAE", round(gbr_re_metrics["train_mae"], 4)],
            ["Test MAE", round(gbr_re_metrics["test_mae"], 4)],
        ]
    )

    # 6. GBR model using both Re and St
    gbr_st_model, gbr_st_metrics, gbr_st_params = train_gbr(df, use_st=True)
    print_params("Best GBR Re St Hyperparameters", gbr_st_params)

    df["Cd_gbr_rest"] = gbr_st_model.predict(df[["Re", "St"]])

    print_table(
        "GBR Re St Performance Metrics",
        ["Metric", "Value"],
        [
            ["Train R2", round(gbr_st_metrics["train_r2"], 4)],
            ["Test R2", round(gbr_st_metrics["test_r2"], 4)],
            ["Train RMSE", round(gbr_st_metrics["train_rmse"], 4)],
            ["Test RMSE", round(gbr_st_metrics["test_rmse"], 4)],
            ["Train MAE", round(gbr_st_metrics["train_mae"], 4)],
            ["Test MAE", round(gbr_st_metrics["test_mae"], 4)],
        ]
    )

    # 7. Generate all visual plots
    print("\nGenerating plots...")

    plot_cd_vs_re(df, os.path.join(RESULTS, "cd_vs_re.png"))
    plot_cd_vs_st(df, os.path.join(RESULTS, "cd_vs_st.png"))
    plot_re_vs_st(df, os.path.join(RESULTS, "re_vs_st.png"))

    plot_combined_cd_re(
        df,
        poly_metrics,
        gbr_re_model,
        gbr_st_model,
        os.path.join(RESULTS, "combined_cd_re.png")
    )

    plot_pred_vs_actual(df, "Cd_gbr_re", os.path.join(RESULTS, "pred_vs_actual_gbr_re.png"))
    plot_pred_vs_actual(df, "Cd_gbr_rest", os.path.join(RESULTS, "pred_vs_actual_gbr_rest.png"))

    print("\nPlots saved in:", RESULTS)

    # 8. Live predictor
    print("\nLive Cd Predictor")
    print("Best model uses Re and St")
    print("Valid input ranges")
    print("Re between 20 and 5000")
    print("St between 0.12 and 0.3")
    print()

    while True:
        try:
            Re_val = float(input("Enter Re: "))
            St_val = float(input("Enter St: "))

            Cd_val = gbr_st_model.predict([[Re_val, St_val]])[0]
            print(f"Predicted Cd = {Cd_val:.6f}\n")

            if input("Another prediction y or n: ").lower() != "y":
                break

        except Exception:
            print("Invalid input. Try again.\n")



if __name__ == "__main__":
    run()

# Portions of this code, including debugging assistance were developed with help from OpenAI ChatGPT 5.1.
