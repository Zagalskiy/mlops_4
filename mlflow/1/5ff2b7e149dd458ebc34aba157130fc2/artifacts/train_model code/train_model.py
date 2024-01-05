# Этот скрипт принимает 1 аргумент командной строки: имя модели (naive, exp, pl)
# Импорт библиотек
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
import pickle
import pandas as pd
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient


# Имя модели передаётся как 2-й аргумент командной строки
model_name = sys.argv[1]

# Установка URI для отслеживания экспериментов в MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f'train_model_{model_name}')
SEASON = 24*7

# Чтение данных из CSV-файла в объект DataFrame
y_train = pd.read_csv('/home/igor/mlops_4/datasets/data_train.csv', index_col='timestamp', parse_dates=True)

# Выбор модели на основе аргумента командной строки
if model_name == "naive":  # Наивное сезонное предсказание с суточной сезонностью
    model = NaiveForecaster(strategy="mean", sp=SEASON)
elif model_name == "exp":  # Предсказание тройным экспоненциальным сглаживанием с учётом тренда и сезонности
    model = ExponentialSmoothing(trend="mul", seasonal="add", sp=SEASON, method='ls')
elif model_name == "pl":  # 
    holt_winter_mul_boxcox = ExponentialSmoothing(
        trend="mul", seasonal="add", use_boxcox=True, sp=SEASON, method='ls'
    )
    model = TransformedTargetForecaster(steps=[
        ("deseasonalize1", Deseasonalizer(model="multiplicative", sp=24)),
        ("deseasonalize2", Deseasonalizer(model="multiplicative", sp=SEASON)),
        ("forecaster", holt_winter_mul_boxcox)
    ])
else:
    raise ValueError("Invalid model name provided")

# Работа с MLflow
with mlflow.start_run():
    model.fit(y_train)  # Обучение модели
    mlflow.sklearn.log_model(model,  # Логирование модели
                             artifact_path=f"{model_name}_model",
                             registered_model_name=f"{model_name}_model")
    mlflow.log_artifact(local_path="/home/igor/mlops_4/scripts/train_model.py",
                        artifact_path="train_model code")  # Логирование кода
    mlflow.end_run()

# Сохранение модели в файл pickle
with open(f'/home/igor/mlops_4/models/{model_name}_model.pickle', 'wb') as f:
    pickle.dump(model, f)
