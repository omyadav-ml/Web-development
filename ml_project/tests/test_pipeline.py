import tempfile
from ml_project.data import generate_synthetic_data
from ml_project.features import split_features_target
from ml_project.model import train_and_save, load_model
import os


def test_end_to_end_runs_and_saves_model():
    df = generate_synthetic_data(n_samples=200, n_features=5, random_state=0)
    X_train, X_test, y_train, y_test = split_features_target(df, test_size=0.2, random_state=0)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "model.joblib")
        model, path = train_and_save(X_train, y_train, save_path=out, params={"random_state":0})
        assert os.path.exists(path)
        loaded = load_model(path)
        preds = loaded.predict(X_test)
        assert len(preds) == len(y_test)
