from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_and_log

with DAG(
    "recommender_pipeline",
    start_date=datetime(2025,1,1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"owner": "Corey Leath"},
    tags=["recsys","capstone"]
) as dag:

    t1 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t2 = PythonOperator(task_id="train_model", python_callable=train_model)
    t3 = PythonOperator(task_id="evaluate_and_log", python_callable=evaluate_and_log)

    t1 >> t2 >> t3
