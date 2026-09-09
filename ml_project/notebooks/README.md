This folder is intended for Jupyter notebooks demonstrating the pipeline.

To create and run a notebook locally:

1. Install dependencies (see requirements.txt)
2. Start a notebook server:
   python -m notebook
3. Create a new notebook and import the package, for example:

```python
from ml_project.data import generate_synthetic_data
from ml_project.features import split_features_target
from ml_project.model import train_and_save

df = generate_synthetic_data(n_samples=500)
X_train, X_test, y_train, y_test = split_features_target(df)
train_and_save(X_train, y_train, save_path='models/notebook_model.joblib')
```
