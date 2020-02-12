from app_main.engine import ForecasterEngine
from pandas import read_csv


# df = read_csv('..\\example_data\\property-sales\\raw_sales.csv')
# print(df)
# nominal_columns = {'propertyType', 'bedrooms'}
# date_column = 'datesold'

df = read_csv('..\\example_data\\household_power_consumption.csv', nrows=150, sep=';')
print(df)
datetime_columns = {'Date', 'Time'}
nominal_columns = {}

engine = ForecasterEngine()

processed_seq = engine.preprocess(df, datetime_columns, nominal_columns)

print('Processed sequence:')
print('Shape:', processed_seq.shape)
print(processed_seq[:40])
