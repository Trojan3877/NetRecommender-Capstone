import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from math import sqrt
import json
from pathlib import Path

def train_model():
    mlflow.set_experiment("NetRecommender")
    with mlflow.start_run(run_name="sgd_baseline"):
        df = pd.read_csv("data/processed/train.csv")
        X = df[["user_id","item_id"]].values
        y = df["rating"].values
        model = SGDRegressor(max_iter=1000, tol=1e-3).fit(X, y)

        preds = model.predict(X)
        rmse = sqrt(mean_squared_error(y, preds))
        mlflow.log_param("model_type", "SGDRegressor")
        mlflow.log_metric("rmse_train", rmse)

        Path("models").mkdir(exist_ok=True)
        mlflow.sklearn.log_model(model, "model")
        with open("models/model_metadata.json","w") as f:
            json.dump({"rmse_train": rmse}, f)

        print(f"✅ Trained model. RMSE={rmse:.3f}")
