import pandas as pd

path = '..\\example_data\\population-time-series-data\\POP.csv'

df = pd.read_csv(path)

df.drop(['realtime_start', 'realtime_end'], axis=1, inplace=True)

df.to_csv('..\\example_data\\population-time-series.csv', index=False)

print(df)
