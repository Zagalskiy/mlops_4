# Импорт библиотеки пандас
import pandas as pd


# Чтение данных из CSV-файла в объект DataFrame
data = pd.read_csv('/home/igor/mlops_4/datasets/data.csv', index_col='timestamp', parse_dates=True)

# Обработка данных
df = data.groupby('date').sum()
df.drop(df.tail(1).index, inplace=True)
df.drop(df.head(1).index, inplace=True)
numeric_df = data.select_dtypes(include=['number'])
df_h = numeric_df.groupby(pd.Grouper(freq='1h')).sum()
df_h.drop(df_h.tail(1).index, inplace=True)
df_h.drop(df_h.head(1).index, inplace=True)
df_h.hour = pd.to_datetime(df_h.index).hour
df_h.value = df_h.value.astype('float')

# Запись обработанных данных
df_h.to_csv('/home/igor/mlops_4/datasets/data_processed.csv', index_label='timestamp')
