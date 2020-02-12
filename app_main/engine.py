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


class ForecasterEngine:

    def __init__(self):
        self.df = None

    # Generates a Deep Learning model for forecasting on given data frame
    def generate_model(self, df, model_type, nominal_features, datetime_features, n_steps_in, n_steps_out):
        print('starting generation')

        print('Generating a {} model with {} time-related feature(s), where nominal features are:'.format(model_type, datetime_features))
        print(nominal_features)

        print('Data frame:')
        print(df)



        # processed_sequence = self.preprocess(current_df)
        #
        # # A dictionary that holds all classes of models for every symbol
        # models_dict = {
        #     'MLP': models.TimeseriesMLPModel(),
        #     'CNN': models.TimeseriesCNN_SimpleModel(),
        #     'LSTM': models.TimeseriesLSTMModel()
        # }
        #
        # timeseries_model = models_dict.get(model_type)
        #
        # X, y = timeseries_model.split_sequence(sales_dataset, store_dataset, n_steps_in, n_steps_out)
        #
        # X_train, X_test, y_train, y_test = timeseries_model.train_test_split(X, y, test_size=0.1)
        #
        # timeseries_model.compile_model()
        #
        # timeseries_model.train_model(X_train, y_train, epochs=50)
        #
        # (y_forecast, test_mae, test_rmse, test_r2, test_mape) = self.predict_evaluate(timeseries_model, X_test, y_test)
        #
        # return timeseries_model

    ''' Performs preprocessing of data set - stanardizes numerical features, encodes nominal, imputes missing
    values, etc.
    df - data frame
    datetime_features - set of features related in time (date, time)
    nominal_features - set of nominal features of the data set'''
    def preprocess(self, df, datetime_features, nominal_features={}, extract_datetime_attributes=None):

        # # For Rossmann dataset
        # number_of_stores = 10
        #
        # # Sort data by store, preserving date order
        # df = df.sort_values(by=['Store', 'Date'])
        # # Ignore rows with 0 sales count
        # df = df.loc[df['Store'] <= number_of_stores]
        # df = df.loc[df['Sales'] != 0]

        # Sort dataframe by the time-related column
        df = df.sort_values(by=list(datetime_features))

        # Set of all features
        all_features = set(df.columns)
        # Numerical
        numerical_features = all_features.difference(nominal_features, datetime_features)

        processed_features = []

        # Choose a scaler to normalize/standardize numeric data
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = StandardScaler()
        # scaler = PowerTransformer()

        # For every numerical feature
        for feature_name in numerical_features:
            # Convert to numpy array
            feature = array(df[feature_name])
            # Reshape to a 2D array
            feature = feature.reshape((feature.shape[0], 1))
            # Normalize
            feature = scaler.fit_transform(feature)
            processed_features.append(feature)

        oneHotEncoder = OneHotEncoder()

        for feature_name in nominal_features:
            # Convert to numpy array
            feature = array(df[feature_name])
            # Reshape to a 2D array
            feature = feature.reshape((feature.shape[0], 1))
            # Encode to one hot
            feature = oneHotEncoder.fit_transform(feature).toarray()
            processed_features.append(feature)

        # Stack horizontally to get a 2D array
        processed_sequence = hstack(processed_features)

        return processed_sequence




    ''' Makes prediction and evaluates trained timeseries model. Plots results of prediction'''
    def predict_evaluate(self, ts_model, X_test, y_test):
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
