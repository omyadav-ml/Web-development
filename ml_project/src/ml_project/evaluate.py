"""Evaluation utilities."""
from sklearn.metrics import accuracy_score, precision_score, recall_score
from .model import load_model


def evaluate_model(model_path: str, X_test, y_test):
    model = load_model(model_path)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds))
    }
    return metrics