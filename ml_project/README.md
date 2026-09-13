# ml_project — example end-to-end ML project

This repository contains an example end-to-end Python ML project with:

- Synthetic data generator
- Modular package in src/ml_project
- Training script that logs to MLflow and saves a model artifact
- Evaluation utilities
- Tests (pytest)
- Dockerfile for reproducible runs
- GitHub Actions CI workflow

Quickstart (from repository root):

1. Create a virtualenv and activate it:
   python -m venv .venv
   .\.venv\Scripts\activate
2. Install dependencies:
   pip install -r ml_project\requirements.txt
   pip install -e ml_project
3. Run a training run (logs to mlflow locally):
   python -m ml_project.train --n-samples 500
4. Run tests:
   pytest ml_project\tests

See files under ml_project/src/ml_project for the implementation and ml_project/notebooks for notes.