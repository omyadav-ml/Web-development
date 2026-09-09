"""Data generation utilities."""
from sklearn.datasets import make_classification
import pandas as pd
from pathlib import Path


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10, random_state: int = 42, out_path: str | None = None):
    """Generate a synthetic binary classification dataset and optionally save to CSV.

    Returns a pandas DataFrame with a column named 'target'.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=max(2, n_features // 2),
                               n_redundant=0, n_classes=2, random_state=random_state)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df