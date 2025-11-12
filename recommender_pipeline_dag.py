from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model

with DAG(
    'recommender_pipeline',
    default_args={'owner': 'Corey Leath', 'start_date': datetime(2025, 1, 1)},
    schedule_interval='@daily',
    catchup=False
) as dag:

    task1 = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data)
    task2 = PythonOperator(task_id='train_model', python_callable=train_model)
    task3 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)

    task1 >> task2 >> task3
