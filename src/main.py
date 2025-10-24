"""
main.py
----------------------------------------
Central execution script for the CMSE802 Final Project:
"Python-Based Simulation and Prediction of Vortex Shedding
Behind a Circular Cylinder at Moderate Reynolds Numbers."

This script orchestrates:
- Data loading & preprocessing
- Exploratory data analysis (EDA)
- Polynomial regression modeling
- Random Forest classification
- Visualization of flow trends
- Numerical energy-consistency checks

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import os
import numpy as np
from src.data_loader import load_and_prepare_data
from src.eda import run_eda
from src.regression_model import polynomial_regression
from src.random_forest_model import train_random_forest
from src.visualization import plot_3d_scatter, plot_3d_surface
from src.energy_check import check_rmse_stability, check_energy_amplification

def main():
    print("\n🚀 Starting CMSE802 vortex-shedding analysis pipeline...\n")
    os.makedirs("results", exist_ok=True)

    # 1️⃣ Load data
    df = load_and_prepare_data("data/cylinder_vortex_full.csv")

    # 2️⃣ Exploratory analysis
    print("🔍 Running exploratory data analysis...")
    run_eda(df)

    # 3️⃣ Regression modeling
    print("\n📈 Performing polynomial regression on Cd and St...")
    metrics = polynomial_regression(df)

    # 4️⃣ Random Forest regime prediction
    print("\n🌲 Training Random Forest classifier...")
    model, acc = train_random_forest(df)

    # 5️⃣ 3D Visualization
    print("\n🎨 Generating 3D flow visualizations...")
    plot_3d_scatter(df)
    plot_3d_surface(df)

    # 6️⃣ Energy & stability checks
    print("\n⚙️ Performing numerical stability checks...")
    # Generate pseudo predictions for test
    y_true = np.linspace(0.1, 1.0, 100)
    y_pred_prev = y_true + np.random.normal(0, 0.005, 100)
    y_pred_curr = y_true + np.random.normal(0, 0.004, 100)
    check_rmse_stability(y_true, y_pred_prev, y_pred_curr)
    check_energy_amplification(y_true)

    # 7️⃣ Summary
    print("\n✅ Pipeline complete.")
    print("--------------------------------------")
    print(f"Polynomial Regression RMSEs: Cd={metrics['Cd']['RMSE']:.4f}, St={metrics['St']['RMSE']:.4f}")
    print(f"Random Forest Accuracy: {acc:.3f}")
    print("All plots and data stored in /results and /data folders.\n")

if __name__ == "__main__":
    main()
