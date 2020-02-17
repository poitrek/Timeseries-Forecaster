import pandas as pd


path = '..\\example_data\\household_power_consumption.csv'

df = pd.read_csv(path, sep=';', low_memory=False)

print(df)
print(df.columns)

column_to_remove = 'Global_active_power'

df.drop(column_to_remove, axis=1, inplace=True)

df = df.tail(5000)

print(df)

df.to_csv('..\\example_data\\household_power_test.csv', sep=';', index=False)
