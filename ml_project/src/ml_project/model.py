"""Model training and persistence."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib
import mlflow
import tempfile


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])


def train_and_save(X, y, save_path: str, run_name: str | None = None, params: dict | None = None):
    """Train a model pipeline, save it to save_path, and log to MLflow.

    Returns the fitted pipeline object and the path to the saved artifact.
    """
    pipeline = build_pipeline(random_state=params.get("random_state", 42) if params else 42)
    with mlflow.start_run(run_name=run_name):
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        pipeline.fit(X, y)
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_p)
        mlflow.log_artifact(str(save_p))
    return pipeline, str(save_p)


def load_model(path: str):
    return joblib.load(path)