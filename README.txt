# Vortex Induced Drag Prediction Using Machine Learning  
CMSE 802 Final Project — Arnab Mustafi Arka

## Project Overview
This project models the drag coefficient Cd of flow over a circular cylinder using polynomial regression and Gradient Boosting Regression GBR.

Polynomial models of degree 1 to 5 are evaluated. 
Two GBR models are used: GBR using only Re, and GBR using both Re and St.

The goal is to evaluate whether machine learning can predict Cd more accurately than polynomial regression and whether adding St improves prediction accuracy.

## Research Question
Can Strouhal number provide additional predictive value for machine-learning models of circular-cylinder drag compared to Reynolds-number-only models?

## Key Findings

Cd is strongly negatively correlated with both Re and St. 
 
Correlation with St absolute value about 0.88 is slightly stronger than with Re absolute value about 0.82.  

Polynomial regression best degree 3 captures the global trend but underfits nonlinear behavior.  

GBR Re St provides the highest accuracy, test R2 around 0.90.  

GBR Re also performs well, test R2 around 0.82, but lacks information from St.  

Even if curves look visually close, numerical metrics show clear improvement when St is included.

## Final Project Folder Structure

project_root/
│
├── data/
│   └── vortex_data.csv
│
├── results/
│   ├── eda/
│   │   ├── summary_stats.csv
│   │   ├── corr_heatmap.png
│   │   ├── hist_Cd.png
│   │   ├── hist_Re.png
│   │   ├── hist_St.png
│   │   └── pairplot.png
│   │
│   ├── cd_vs_re.png
│   ├── cd_vs_st.png
│   ├── re_vs_st.png
│   ├── combined_cd_re.png
│   ├── pred_vs_actual_gbr_re.png
│   └── pred_vs_actual_gbr_rest.png
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── polynomial_regression.py
│   ├── gbr_model.py
│   ├── visualization.py
│   └── main.py
│
├── README.md
├── requirements.txt

Installation and Requirements

Install required packages
pip install -r requirements.txt

How to Run the Project

From the project root directory run:
python -m src.main

This will:

load the dataset
run EDA
train polynomial regression models
train GBR models
print metrics in tables
generate all plots
launch the interactive Cd predictor

Running Unit Tests

To run the full unittest suite:
python -m unit_test.unittest_final

Tests included:

Dataset loading
Polynomial regression output format
GBR(Re) model output
GBR(Re, St) model output

All tests use a 50-row subset for speed.

Generated Plots

results/eda/

summary_stats.csv
corr_heatmap.png
hist_Re.png
hist_St.png
hist_Cd.png
pairplot.png

results/

cd_vs_re.png
cd_vs_st.png
re_vs_st.png
combined_cd_re.png
pred_vs_actual_gbr_re.png
pred_vs_actual_gbr_rest.png

Description of Key Files

src/main.py

Runs the complete pipeline including EDA, polynomial regression, GBR models, metric tables, plot generation, and the interactive Cd predictor.

src/eda.py

Creates histograms, pairplot and correlation heatmap.

src/polynomial_regression.py

Trains polynomial regression models degree 1 to 5 and returns best model and metrics.

src/gbr_model.py

Trains Gradient Boosting Regression models using Re or Re plus St with grid search. Returns best model, parameters and metrics.

src/visualization.py

Generates scatter plots, combined Cd vs Re comparison plot, and predicted versus actual plots. Includes median smoothing of St for stable GBR Re St predictions.

Interactive Cd Predictor

After running main.py you will see:
LIVE Cd PREDICTOR
Using the best model GBR Re St
Valid ranges:
Re 20 to 5000
St 0.12 to 0.3
Example:
Enter Re: 2000
Enter St: 0.3
Predicted Cd = 0.655415

Summary of Results

Model Performance Ranking

1.GBR Re St — best performance, test R2 around 0.90
2.GBR Re — strong baseline, test R2 around 0.82
3.Polynomial Regression — test R2 around 0.64 to 0.79

Why Re plus St Works Best

Re describes viscous and inertial flow effects.
St describes vortex shedding behavior.
Using both provides a more complete physical description of Cd.

CMSE 802 Rubric Alignment

This project meets rubric requirements including
clear research question
exploratory data analysis
modeling and comparison
organized multi file codebase
meaningful visualizations
reproducible results
interactive demonstration model



