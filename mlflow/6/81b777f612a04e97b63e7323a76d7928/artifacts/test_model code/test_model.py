# Этот скрипт принимает 1 аргумент командной строки: имя модели (naive, exp, theta)
# Импорт библиотек
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
import sys
import pickle
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


# Имя модели передаётся как 1-й аргумент командной строки
model_name = sys.argv[1]
SEASON = 24*7

# Установка URI для отслеживания экспериментов в MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f'test_model_{model_name}')

# Чтение данных из CSV-файла в объект DataFrame
y_test = pd.read_csv('/home/igor/mlops_4/datasets/data_test.csv', index_col='timestamp', parse_dates=True)
y_test = y_test.asfreq('H')

# Создание объекта ForecastingHorizon с правильной частотой
fh = ForecastingHorizon(y_test.index, is_relative=False, freq='H')

# Выбор и обучение модели на основе аргумента командной строки
if model_name == "naive":  # Наивное сезонное предсказание с суточной сезонностью
    model = NaiveForecaster(strategy="mean", sp=SEASON)
elif model_name == "exp":  # Предсказание тройным экспоненциальным сглаживанием с учётом тренда и сезонности
    model = ExponentialSmoothing(trend="mul", seasonal="add", sp=SEASON, method='ls')
elif model_name == "theta":  # Предсказание двойным экспоненциальным сглаживанием с учётом тренда
    model = ThetaForecaster(sp=SEASON)
else:
    raise ValueError("Invalid model name provided")

# Загрузка модели из файла pickle
with open(f'/home/igor/mlops_4/models/{model_name}_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Работа с MLflow
with mlflow.start_run():
    # Вычисление и вывод оценки
    y_pred = model.predict(fh)
    smape = MeanAbsolutePercentageError(symmetric = True)
    score = smape(y_test.values, y_pred.values)
    mlflow.log_metric("sMAPE", score)  # Логирование метрики sMAPE
    mlflow.log_artifact(local_path="/home/igor/mlops_4/scripts/test_model.py",
                        artifact_path="test_model code")  # Логирование кода
    mlflow.end_run()
