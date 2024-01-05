# Импорт модулей
from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt


# Передача аргумента в скрипты train_model.py и test_model.py
models = ['naive', 'exp', 'theta']

# Определение базовых аргументов для DAG
args = {
    "owner": "admin",
    "start_date": dt.datetime(2024, 1, 1),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False
}

# Создание DAG 
with DAG(
    dag_id='Google_X_mention_score',
    default_args=args,
    schedule='30 * * * *',
    max_active_runs=1,
    tags=['Google', 'X', 'score']
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="python3 /home/igor/mlops_4/scripts/get_data.py",
                            dag=dag)
    process_data = BashOperator(task_id='process_data',
                            bash_command="python3 /home/igor/mlops_4/scripts/process_data.py",
                            dag=dag)
    train_test_split = BashOperator(task_id='train_test_split',
                            bash_command="python3 /home/igor/mlops_4/scripts/train_test_split.py",
                            dag=dag)
    for model in models:
        train_model = BashOperator(
            task_id=f'train_model_{model}',
            bash_command=f"python3 /home/igor/mlops_4/scripts/train_model.py {model}",
            dag=dag
        )
        test_model = BashOperator(
            task_id=f'test_model_{model}',
            bash_command=f"python3 /home/igor/mlops_4/scripts/test_model.py {model}",
            dag=dag
        )

    get_data >> process_data >> train_test_split >> train_model >> test_model
