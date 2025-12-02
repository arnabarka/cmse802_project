"""
gbr_model.py

This file trains a Gradient Boosting Regressor to predict Cd.
It supports two modes.
1. Use only Re as input.
2. Use both Re and St as inputs.

It performs the following actions.
1. Select input features.
2. Split data into train and test.
3. Perform grid search to find best hyperparameters.
4. Train the best model.
5. Evaluate model on train and test sets.
6. Return model, metrics and best parameters.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_gbr(df, use_st=False):
    """
    Train a Gradient Boosting Regressor using grid search.

    Parameters
    df : pandas DataFrame
        Input dataset with Re, St and Cd.
    use_st : bool
        If True, the model uses Re and St.
        If False, the model uses only Re.

    Returns
    best_model : trained GradientBoostingRegressor
    metrics : dictionary containing train and test metrics
    best_params : dictionary of best hyperparameters
    """

    # 1. Select input columns
    if use_st:
        X = df[["Re", "St"]].values
    else:
        X = df[["Re"]].values

    y = df["Cd"].values

    # 2. Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Grid search for hyperparameters
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.9, 1.0],
    }

    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    # 4. Fit grid search and get best model
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    # 5. Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # 6. Compute performance metrics
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
    }

    return best_model, metrics, best_params

# Portions of this code, including debugging assistance were developed with help from OpenAI ChatGPT 5.1.
