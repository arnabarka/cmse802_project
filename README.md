# 🌀 CMSE802 Final Project  
**Flow Characterization and Data-Driven Modeling of Cylinder Vortex Shedding**

Author: **Arnab Mustafi Arka**  
Course: **CMSE 802 – Computational Modeling for Engineers**  
Semester: **Fall 2025**  
Advisor: *Dr. Ricardo Mejia-Alvarez*  

---

## 📘 Abstract
This project develops a **Python-based computational framework** for analyzing vortex shedding behind a circular cylinder at moderate Reynolds numbers.  
It combines **physics-informed data generation**, **statistical regression**, and **machine learning classification** to study aerodynamic behavior across laminar, transitional, and turbulent flow regimes.

The workflow covers:
- Synthetic data generation with noise-driven realism  
- Exploratory data analysis (EDA) and correlation studies  
- Polynomial regression for drag coefficient (*Cd*) and Strouhal number (*St*) prediction  
- Random Forest classification of flow regimes  
- Numerical stability and ΔRMSE-based energy checks  
- Automated testing framework for reliability and reproducibility

---

## 🗂️ Project Structure
```
cmse802_project/
│
├── src/ # Core Python source code
│ ├── init.py
│ ├── data_loader.py # Data generation, cleaning, normalization
│ ├── eda.py # Correlation, scatter, and regime visualization
│ ├── regression_model.py # Polynomial regression (Cd–Re, St–Re)
│ ├── random_forest_model.py # Flow-regime classification using Random Forest
│ ├── metrics.py # RMSE, MAE, R², ΔRMSE utilities
│ ├── visualization.py # 3D & residual plots, correlation matrices
│ ├── energy_check.py # Stability and energy amplification checks
│ └── main.py # Central execution pipeline
│
├── data/
│ ├── cylinder_vortex_full.csv # Generated dataset (auto-created)
│ └── README.md # Notes on data origin and preprocessing
│
├── results/ # Auto-saved figures and outputs
│ ├── corr_heatmap.png
│ ├── re_cd.png
│ ├── re_st.png
│ ├── cd_fit.png
│ ├── st_fit.png
│ ├── cd_residuals.png
│ ├── st_residuals.png
│ ├── 3d_surface1.png
│ ├── 3d_surface2.png
│ ├── confusion_matrix.png
│ ├── feature_importance.png
│ └── predicted_regions.png
│
├── tests/ # Pytest-based validation suite
│ ├── test_data_validation.py
│ ├── test_environment.py
│ └── init.py
│
├── docs/ # Reports and documentation
│ ├── Flow Characterization and Data.docx
│ ├── HW01-ProjectPlan-ARNAB.ipynb
│ └── README.md
│
├── notebooks/ # Interactive exploration
│ ├── project_cmse_802.ipynb
│ └── demo_results.ipynb
│
├── .gitignore
└── README.md
