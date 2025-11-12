import mlflow
import mlflow.sklearn

def train_model():
    with mlflow.start_run():
        # Train logic here
        model = ...
        mlflow.log_param("model_type", "Collaborative Filtering")
        mlflow.log_metric("rmse", 0.93)
        mlflow.sklearn.log_model(model, "model")
        return model
