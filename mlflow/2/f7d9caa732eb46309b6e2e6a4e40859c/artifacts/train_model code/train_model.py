# Этот скрипт принимает 1 аргумент командной строки: имя модели (naive, exp, pl)
# Импорт библиотек
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
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
y_test = pd.read_csv('/home/igor/mlops_4/datasets/data_test.csv', index_col='timestamp', parse_dates=True)

# Создание объекта ForecastingHorizon с правильной частотой
fh = ForecastingHorizon(y_test.index, is_relative=False, freq='H')

# Выбор модели на основе аргумента командной строки
if model_name == "naive":  # Наивное сезонное предсказание с суточной сезонностью
    model = NaiveForecaster(strategy="mean", sp=SEASON)
elif model_name == "exp":  # Предсказание тройным экспоненциальным сглаживанием с учётом тренда и сезонности
    model = ExponentialSmoothing(trend="mul", seasonal="add", sp=SEASON, method='ls')
elif model_name == "theta":  # Предсказание двойным экспоненциальным сглаживанием с учётом тренда
    model = ThetaForecaster(sp=SEASON)
else:
    raise ValueError("Invalid model name provided")

# Работа с MLflow
with mlflow.start_run():
    if model_name == "theta":
        model.fit(y_train, fh=fh)  # Обучение модели theta
    else:
        model.fit(y_train)  # Обучение моделей naive и exp

    mlflow.sklearn.log_model(model,  # Логирование модели
                             artifact_path=f"{model_name}_model",
                             registered_model_name=f"{model_name}_model")
    mlflow.log_artifact(local_path="/home/igor/mlops_4/scripts/train_model.py",
                        artifact_path="train_model code")  # Логирование кода
    mlflow.end_run()

# Сохранение модели в файл pickle
with open(f'/home/igor/mlops_4/models/{model_name}_model.pickle', 'wb') as f:
    pickle.dump(model, f)
