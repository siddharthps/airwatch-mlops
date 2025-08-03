import mlflow
import os
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Load .env variables
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("test_experiment")

# Dummy data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
model = LinearRegression()

with mlflow.start_run():
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_rmse", 0.123)
