import pandas as pd


path = '..\\example_data\\household_power_100_1000.csv'

df = pd.read_csv(path, sep=';', low_memory=False)

print(df)

print(df[''])
