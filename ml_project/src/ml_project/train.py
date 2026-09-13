"""Entrypoint script for training a model and logging to MLflow."""
from argparse import ArgumentParser
from ml_project.data import generate_synthetic_data
from ml_project.features import split_features_target
from ml_project.model import train_and_save
from pathlib import Path


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--model-out", type=str, default="models/model.joblib")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    df = generate_synthetic_data(n_samples=args.n_samples, random_state=args.random_state)
    X_train, X_test, y_train, y_test = split_features_target(df)

    params = {"random_state": args.random_state, "n_samples": args.n_samples}
    model, saved_path = train_and_save(X_train, y_train, save_path=args.model_out, run_name=args.run_name, params=params)
    print(f"Model trained and saved to: {saved_path}")


if __name__ == "__main__":
    main()