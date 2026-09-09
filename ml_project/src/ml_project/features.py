"""Feature preprocessing helpers."""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, random_state: int = 42):
    """Split a DataFrame into train/test arrays for X and y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test