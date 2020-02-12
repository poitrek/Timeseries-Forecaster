from numpy import array
from numpy import hstack
from numpy import nan
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import version_first.models as models


# Generates a Deep Learning model for forecasting on given data frame
def generate_model(current_df, model_type, n_steps_in, n_steps_out):
    print('starting generation')

    sales_dataset, store_dataset = preprocess(current_df)

    # A dictionary that holds all classes of models for every symbol
    models_dict = {
        'MLP': models.TimeseriesMLPModel(),
        'CNN': models.TimeseriesCNN_SimpleModel(),
        'LSTM': models.TimeseriesLSTMModel()
    }

    timeseries_model = models_dict.get(model_type)

    X, y = timeseries_model.split_sequence(sales_dataset, store_dataset, n_steps_in, n_steps_out)

    X_train, X_test, y_train, y_test = timeseries_model.train_test_split(X, y, test_size=0.1)

    timeseries_model.compile_model()

    timeseries_model.train_model(X_train, y_train, epochs=50)

    (y_forecast, test_mae, test_rmse, test_r2, test_mape) = predict_evaluate(timeseries_model, X_test, y_test)

    return timeseries_model


def preprocess(df):
    # For Rossmann dataset
    number_of_stores = 10

    # Sort data by store, preserving date order
    df = df.sort_values(by=['Store', 'Date'])
    # Ignore rows with 0 sales count
    df = df.loc[df['Store'] <= number_of_stores]
    df = df.loc[df['Sales'] != 0]

    # Choose a scaler to normalize/standardize numeric data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    # scaler = PowerTransformer()
    X_sales = array(df['Sales'])
    # Reshape to a 2D array
    X_sales = X_sales.reshape((X_sales.shape[0], 1))
    # Save original column before normalizing
    X_sales_ori = X_sales
    # Normalize
    X_sales = scaler.fit_transform(X_sales)

    # Customers / normalize
    X_customers = array(df['Customers'])
    X_customers = X_customers.reshape((X_customers.shape[0], 1))
    X_customers = scaler.fit_transform(X_customers)

    oneHotEncoder = OneHotEncoder()

    # Day of week / encode to one hot
    X_dayOfWeek = array(df['DayOfWeek'])
    X_dayOfWeek = X_dayOfWeek.reshape((X_dayOfWeek.shape[0], 1))
    X_dayOfWeek_enc = oneHotEncoder.fit_transform(X_dayOfWeek).toarray()

    # Open
    X_open = array(df['Open'])
    X_open = X_open.reshape((X_open.shape[0], 1))

    # Promo
    X_promo = array(df['Promo'])
    X_promo = X_promo.reshape((X_promo.shape[0], 1))

    # School Holiday
    X_schoolHoliday = array(df['SchoolHoliday'])
    X_schoolHoliday = X_schoolHoliday.reshape((X_schoolHoliday.shape[0], 1))

    # State holiday / encode to one hot
    X_stateHoliday = array(df['StateHoliday'])
    X_stateHoliday = X_stateHoliday.reshape((X_stateHoliday.shape[0], 1))
    X_stateHoliday_enc = oneHotEncoder.fit_transform(X_stateHoliday).toarray()

    # Store type / encode to one hot
    X_storeType = array(df['StoreType'])
    X_storeType = X_storeType.reshape((X_storeType.shape[0], 1))
    X_storeType_enc = oneHotEncoder.fit_transform(X_storeType).toarray()

    # Assortment / encode to one hot
    X_assortment = array(df['Assortment'])
    X_assortment = X_assortment.reshape((X_assortment.shape[0], 1))
    X_assortment_enc = oneHotEncoder.fit_transform(X_assortment).toarray()

    # Competition distance / impute, normalize
    X_competitionDistance = array(df['CompetitionDistance'])
    X_competitionDistance = X_competitionDistance.reshape((X_competitionDistance.shape[0], 1))
    # Impute nans with the median
    imputer_median = SimpleImputer(missing_values=nan, strategy='median', verbose=1)
    X_competitionDistance = imputer_median.fit_transform(X_competitionDistance)
    X_competitionDistance = scaler.fit_transform(X_competitionDistance)

    # Promo2
    X_promo2 = array(df['Promo2'])
    X_promo2 = X_promo2.reshape((X_promo2.shape[0], 1))

    # Competition open since month/year - convert to number of days elapsed since competition opened
    df_comp_open_date = df[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].rename(columns={
        'CompetitionOpenSinceYear': 'year', 'CompetitionOpenSinceMonth': 'month'
    })
    df_comp_open_date_merged = pd.to_datetime(df_comp_open_date.assign(day=1))

    # Dataframe of numbers of days since competition opened
    df_days_since_comp_open = df['Date'] - df_comp_open_date_merged
    # Convert to numpy array
    X_daysSinceCompOpen = array([d.days for d in df_days_since_comp_open])
    X_daysSinceCompOpen = X_daysSinceCompOpen.reshape((X_daysSinceCompOpen.shape[0], 1))
    # Impute, normalize
    X_daysSinceCompOpen = imputer_median.fit_transform(X_daysSinceCompOpen)
    X_daysSinceCompOpen = scaler.fit_transform(X_daysSinceCompOpen)

    print('Data preprocessed.')

    # Promo2 since week/year, PromoInterval - Leave for now

    # Get store ids
    X_store = array(df['Store']).reshape((len(df), 1))

    sales_dataset = hstack((X_sales, X_customers, X_dayOfWeek_enc, X_open, X_promo, X_stateHoliday_enc,
                            X_schoolHoliday, X_daysSinceCompOpen, X_sales_ori))
    store_dataset = hstack((X_store, X_storeType_enc, X_assortment_enc, X_competitionDistance, X_promo2))

    return sales_dataset, store_dataset


''' Makes prediction and evaluates trained timeseries model. Plots results of prediction'''
def predict_evaluate(ts_model, X_test, y_test):
    y_dash = ts_model.predict(X_test)
    print('\nPrediction / actual results:')
    for i in range(50):
        print(i+1, '. ', y_dash[i], ' - ', y_test[i], sep='')
    print('...')
    # accuracy = ts_model.evaluate(X_test, y_test)
    # print('Evaluated model accuracy: (mae)', accuracy)

    print('\nEvaluation metrics:')
    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_dash)
    print('Mean absolute error: %.3f' % mae)
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_test, y_dash))
    print('Root mean squared error: %.3f' % rmse)
    # R^2 Score
    r2_sc = r2_score(y_test, y_dash)
    print('R2 Score: %.4f' % r2_sc)
    # Mean Absolute Percentage Error
    # This was incorrect
    # mape = 100 * np.mean(np.abs(1 - y_test/y_dash))
    mape = 100 * np.mean(np.abs(1 - y_dash/y_test))
    print('Mean absolute percentage error: %.3f' % mape, '%', sep='')
    # print('ACTUAL MAPE: %.4f' % mape_true, ' %', sep='')
    # Root Mean Square Percentage Error
    rmspe = np.sqrt(np.mean((1 - y_dash/y_test) ** 2))
    print('Root mean square percentage error: %.5f' % rmspe)

    tests_scores = np.stack((np.arange(len(y_test)), y_dash.flatten(), y_test.flatten(), np.abs(y_test - y_dash).flatten()), axis=-1)
    tests_scores = tests_scores[tests_scores[:, 3].argsort()]
    print('\nTop 5 predictions:')
    # for i in range(5):
    #     print(tests_scores[i, 0], '. ', tests_scores[i, 1], ' - ', tests_scores[i, 2], sep='')
    for ts in tests_scores[:5]:
        print(int(ts[0]), '. %.3f' % ts[1], ' - ', ts[2], ' (error=%.3f)' % ts[3], sep='')

    return y_dash, mae, rmse, r2_sc, mape
