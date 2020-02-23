import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 14)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

n_stores_to_save = 160

# Read original training set
path = 'C:\\users\\piotr\\pycharmprojects\\timeseries-forecasting\\rossmann-store-sales\\train.csv'
df = pd.read_csv(path, sep=';', parse_dates=['Date'], low_memory=False)

print(df)
print(df.columns)
logging.info('Number of stores saving: %d', n_stores_to_save)

# Remove unwanted columns
df = df.drop(['DayOfWeek', 'PromoInterval', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1)
logging.info('Dropped unwanted columns from the frame')

# Filter records with 0 sales count and to desired number of stores
df = df.loc[(df['Store'] <= n_stores_to_save) & (df['Sales'] != 0)]
logging.info('Filtered records by number of store and sales count')

# Competition open since month/year - convert to number of days elapsed since competition opened
df_comp_open_date = df[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].rename(columns={
    'CompetitionOpenSinceYear': 'year', 'CompetitionOpenSinceMonth': 'month'
})
df_comp_open_date_merged = pd.to_datetime(df_comp_open_date.assign(day=1))

# Dataframe of numbers of days since competition opened
daysSinceCompOpen = df['Date'] - df_comp_open_date_merged
daysSinceCompOpen = [d.days for d in daysSinceCompOpen]
df['DaysSinceCompOpen'] = daysSinceCompOpen

# Remove unnecessary columns
df = df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], axis=1)
logging.info('Added new column')

df = df.sort_values(by=['Store', 'Date'])    # Ignore rows with 0 sales count
logging.info('Sorted values properly')

print(df)

file_name = 'rossmann-store-sales-train-{}stores.csv'.format(n_stores_to_save)
logging.info('Saving to \'%s\' ...', file_name)

df.to_csv('..\\example_data\\{}'.format(file_name), sep=';', index=False)
logging.info('Done.')
