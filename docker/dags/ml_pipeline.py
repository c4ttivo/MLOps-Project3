from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator, ShortCircuitOperator,BranchOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import requests
import os
from scripts.train import train_random_forest, train_xgboost
from scripts.clean import clean_data


# Definiciones globales
DATABASE_RAW_URI = 'postgresql+psycopg2://rawusr:rawpass@rawdb/rawdb'
DATABASE_CLEAN_URI = 'postgresql+psycopg2://cleanusr:cleanpass@cleandb/cleandb'

    
def fetch_and_store_data(**kwargs):
    '''
    Fetch data from the API and store it directly into the database.
    '''
    
    # De acuerdo con la documentación de Arflow, top-level code e imports pesados
    # deben ser importados en la función llamable
    import pandas as pd
    from sqlalchemy import create_engine
    
    
    ## download the dataset
    # Directory of the raw data files
    _data_root = './data/Diabetes'
    # Path to the raw training data
    _data_filepath = os.path.join(_data_root, 'Diabetes.csv')
    # Download data
    os.makedirs(_data_root, exist_ok=True)
    if not os.path.isfile(_data_filepath):
        url = 'https://docs.google.com/uc?export= \
        download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
        r = requests.get(url, allow_redirects=True, stream=True)
        open(_data_filepath, 'wb').write(r.content)    


        # Crea un DataFrame con los datos
        df = pd.read_csv(_data_filepath)

        # Almacena los datos en la base de datos
        engine = create_engine(DATABASE_RAW_URI)
        df.to_sql('dataset_diabetes_table', con=engine, if_exists='replace', index=False)
        
        return True

    else:
        raise Exception(f"Error al obtener datos")
    

def end_of_dag():
    pass 


# Configuración del DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Fetch data from drive and store in rawdb ',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
)

fetch_and_store_data_task = ShortCircuitOperator(
    task_id='fetch_and_store_data',
    python_callable=fetch_and_store_data,
    provide_context=True,
    dag=dag,
)


process_data_task = PythonOperator(
    task_id = 'clean_data',
    python_callable = clean_data,
    dag=dag
)

train_random_forest_task = PythonOperator(
    task_id = 'train_random_forest',
    python_callable = train_random_forest,
    dag = dag,
)

train_xgb_task = PythonOperator(
    task_id = 'train_xgboost',
    python_callable = train_xgboost,
    dag = dag,
)


end = DummyOperator(
    task_id='end',
    trigger_rule='none_failed_or_skipped',
    dag=dag,
)

# Task dependencies
fetch_and_store_data_task >> process_data_task >> [train_random_forest_task, train_xgb_task] >> end
