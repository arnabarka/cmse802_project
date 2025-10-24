"""
test_random_forest_model.py
----------------------------------------
Checks classification accuracy and regime consistency.
"""

from src.data_loader import load_and_prepare_data
from src.random_forest_model import train_random_forest

def test_random_forest_accuracy():
    df = load_and_prepare_data()
    _, acc = train_random_forest(df)
    assert acc > 0.90, f"Classifier accuracy too low: {acc:.3f}"
