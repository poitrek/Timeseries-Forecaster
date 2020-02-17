import pandas as pd


path = '..\\example_data\\household_power_100_000.csv'

df = pd.read_csv(path, sep=';', low_memory=False)

df = df.head(40000)

print(df)

df.to_csv('..\\example_data\\household_power_40_000.csv', sep=';', index=False)

