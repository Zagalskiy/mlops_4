import pandas as pd


# Загрузка и преобразование данных
PATH_base = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_GOOG.csv'
data = pd.read_csv(PATH_base, index_col='timestamp', parse_dates=True)
data['date'] = pd.to_datetime(data.index).date
data['hour'] = pd.to_datetime(data.index).hour

# Запись преобразованных данных
data.to_csv('/home/igor/mlops_4/datasets/data.csv', index_label='timestamp')
