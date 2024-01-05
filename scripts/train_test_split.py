# Импорт библиотек
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split

# Чтение данных из CSV-файла в объект DataFrame 
df = pd.read_csv('/home/igor/mlops_4/datasets/data_processed.csv', index_col='timestamp', parse_dates=True)

# Отбрасываем выбросы
y_h = df.value
y = y_h.copy()
y.loc[y < 75] = 75
y.loc[y > 450] = 450

# Разбиение данных на обучающую и тестовую выборки
TEST_SIZE = 0.4
y_train, y_test = temporal_train_test_split(y, test_size=TEST_SIZE)

# Запись обучающей выборки в CSV-файл
y_train.to_csv('/home/igor/mlops_4/datasets/data_train.csv', index_label='timestamp')

# Запись тестовой выборки в CSV-файл
y_test.to_csv('/home/igor/mlops_4/datasets/data_test.csv', index_label='timestamp')
