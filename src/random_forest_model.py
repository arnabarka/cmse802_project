"""
random_forest_model.py
----------------------------------------
Random Forest classifier for predicting flow regimes
(Laminar, Transitional, Turbulent) from Reynolds number,
drag coefficient, and Strouhal number.

Generates:
- Model accuracy report
- Confusion matrix visualization
- Feature importance ranking
- Regime prediction scatter plot

Author: Arnab Mustafi Arka
Course: CMSE802 Final Project (Fall 2025)
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.data_loader import load_and_prepare_data

def train_random_forest(df: pd.DataFrame, save_dir="results",
                        n_estimators=150, random_state=42):
    """
    Trains a Random Forest classifier to predict flow regime.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing Re, Cd, St, Regime.
    save_dir : str
        Directory for saving result plots.
    n_estimators : int
        Number of trees in the Random Forest.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : RandomForestClassifier
        Trained classifier object.
    acc : float
        Accuracy on the test set.
    """
    os.makedirs(save_dir, exist_ok=True)

    X = df[["Re", "Cd", "St"]]
    y = df["Regime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap="coolwarm")
    plt.title("Flow Regime Classification Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # ---- Feature importance ----
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
    plt.figure(figsize=(5, 3))
    sns.barplot(x=feat_imp, y=feat_imp.index, palette="mako")
    plt.title("Feature Importance for Regime Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"))
    plt.close()

    # ---- Predicted regime scatter ----
    df_pred = df.copy()
    df_pred["Predicted"] = clf.predict(X)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df_pred.sample(3000), x="Re", y="Cd", hue="Predicted", s=10)
    plt.xscale("log")
    plt.title("Predicted Flow Regime Regions")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_regions.png"))
    plt.close()

    print(f"✅ Random Forest training complete. Accuracy: {acc:.3f}")
    return clf, acc


if __name__ == "__main__":
    # Standalone demo
    df = load_and_prepare_data()
    model, acc = train_random_forest(df)
    print(f"Model accuracy: {acc:.3f}")
