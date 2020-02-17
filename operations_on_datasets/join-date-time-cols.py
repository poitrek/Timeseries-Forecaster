import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 9)

path = '..\\example_data\\household_power_consumption.csv'

df = pd.read_csv(path, sep=';', low_memory=False)

df['Datetime'] = df['Date'] + ' ' + df['Time']
print(df)

df.drop(['Date', 'Time'], axis=1, inplace=True)

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]

df = df[cols]

print(df)

df.to_csv(path, sep=';', index=False)
