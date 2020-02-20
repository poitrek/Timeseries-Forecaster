import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 14)


path = 'C:\\users\\piotr\\pycharmprojects\\timeseries-forecasting\\rossmann-store-sales\\'

df = pd.read_csv(path + 'train.csv', sep=';', parse_dates=['Date'], low_memory=False)

print(df)

print(df.columns)

df = df.drop(['DayOfWeek', 'PromoInterval', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1)

# Competition open since month/year - convert to number of days elapsed since competition opened
df_comp_open_date = df[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].rename(columns={
    'CompetitionOpenSinceYear': 'year', 'CompetitionOpenSinceMonth': 'month'
})
df_comp_open_date_merged = pd.to_datetime(df_comp_open_date.assign(day=1))

# Dataframe of numbers of days since competition opened
daysSinceCompOpen = df['Date'] - df_comp_open_date_merged
daysSinceCompOpen = [d.days for d in daysSinceCompOpen]
df['DaysSinceCompOpen'] = daysSinceCompOpen

df = df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], axis=1)

df = df.sort_values(by=['Store', 'Date'])    # Ignore rows with 0 sales count
df = df.loc[df['Store'] <= 80]
df = df.loc[df['Sales'] != 0]

print(df)

df.to_csv('..\\example_data\\rossmann-store-sales-train-80stores.csv', sep=';', index=False)
