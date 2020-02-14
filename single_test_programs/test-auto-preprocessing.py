from app_main.engine import ForecasterEngine
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# pd.set_option('display.width', 400)

# np.set_printoptions(precision=4)

# df = read_csv('..\\example_data\\property-sales\\raw_sales.csv')
# print(df)
# nominal_columns = {'propertyType', 'bedrooms'}
# date_column = 'datesold'

df = read_csv('..\\example_data\\household_power_consumption.csv', nrows=150, sep=';')
print(df)
df = df.drop('Time', axis=1)
datetime_columns = 'Date'
nominal_columns = {'Sub_metering_3', 'Sub_metering_2'}
print('df.columns =', df.columns)
print('Sub_metering_2 unique values:', df['Sub_metering_2'].unique())

x_power = np.array(df['Global_active_power'])
x_power = x_power.reshape((x_power.shape[0], 1))
scaler = StandardScaler()
x_power = scaler.fit_transform(x_power)
print('Global active power scaled:')
print(x_power[:20])

engine = ForecasterEngine()

processed_seq = engine.preprocess(df, datetime_columns, nominal_columns)

print('Processed sequence:')
print('Shape:', processed_seq.shape)
print(processed_seq[:40])

power_origin = scaler.inverse_transform(x_power)
print('Inversed power:')
print(power_origin[:20])
