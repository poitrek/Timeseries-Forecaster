import copy
from numpy import array
from numpy import hstack
from numpy import nan
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import app_main.models as models
from app_main.utils import Timer


np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=8)
np.set_printoptions(linewidth=100)
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 9)


class ForecasterEngine:

    def __init__(self):
        self.df = None
        self.timeseries_model = None
        self._pred_feature = None
        self._pred_feature_scaler = None
        self._pred_feature_ori = None
        self._model_gen_time = None

    ''' Generates a Deep Learning model for forecasting based on given data frame and options
    df - dataframe
    model_type - name of the model type returned by radio items from the layout
    nominal_features - list of names of features in the dataframe that are nominal
    predicted_feature - the name of the feature we want to predict
    datetime_feature - the name of the time-related feature
    extra_datetime_feature - list of names of extra date/time features to be extracted from the
                             time-related feature (year, month, hour,...)
    n_steps_in - number of input steps of the time series for the model
    n_steps_out - number of output steps of the time series to be predicted
    n_train_epochs - number of epochs of model training
    differentiate_series - should we differentiate the target feature in order to improve its modelling'''
    def generate_model(self, df, model_type, predicted_feature, nominal_features, datetime_feature, extra_datetime_features,
                       n_steps_in, n_steps_out, n_train_epochs, differentiate_series):

        timer = Timer()
        timer.start_measure_time()

        # Parse to sets
        nominal_features = {} if nominal_features is None else set(nominal_features)
        # datetime_features = {} if datetime_features is None else set(datetime_features)
        if extra_datetime_features is None:
            extra_datetime_features = []

        if predicted_feature in nominal_features:
            raise Exception('Predicted feature cannot be nominal. The model predicts only numerical features.')
        
        print('starting generation')
        print('Generating a {} model with {} time-related feature(s), where nominal features are:'.format(model_type, datetime_feature))
        print(nominal_features)
        print('Data frame:')
        print(df)

        df_extra_datetime = self._extract_datetime_features(df, datetime_feature, extra_datetime_features)
        processed_sequence = self.preprocess(df, datetime_feature, predicted_feature, nominal_features, df_extra_datetime)

        print('Processed sequence:')
        for i in range(20):
            print(processed_sequence[i])
        print('Shape:', processed_sequence.shape)

        # A dictionary that holds all classes of models for every symbol
        models_dict = {
            'MLP': models.TimeseriesMLPModel(),
            'CNN': models.TimeseriesCNN_SimpleModel(),
            'LSTM': models.TimeseriesLSTMModel()
        }

        self.timeseries_model = models_dict.get(model_type)
        print('Class of timeseries model:', type(self.timeseries_model))

        X, y = self.timeseries_model.split_sequence(processed_sequence, n_steps_in, n_steps_out, self._pred_feature_ori)

        print('Model inputs/outputs:')
        for i in range(10):
            print(i, ':', X[i], '<--', y[i])

        # Specify test size based on the size of full sequence
        # Minimum (30% of full sequence, 1000)
        test_size = 0.3 if len(processed_sequence) < 3333 else 1000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        self.timeseries_model.compile_model()

        self.timeseries_model.train_model(X_train, y_train, epochs=n_train_epochs)

        self.predict_evaluate(self.timeseries_model, X_test, y_test)

        timer.stop_measure_time()

        self._model_gen_time = timer.time_elapsed

        # self.timeseries_model.model_name
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


    ''' Returns a new dataframe of time-related features extracted from the datetime column
    (year, quarter, month, day of month, day of week, hour, minute, second'''
    def _extract_datetime_features(self, df, datetime_feature, extra_datetime_features):
        # Parse to datetime. If fails, an exception will be raised
        parsed_feature = pd.to_datetime(df[datetime_feature])
        df_extra = pd.DataFrame()
        # Check for every label of extra feature, if is present in the list
        # If so, add an extra column to the dataframe
        if 'year' in extra_datetime_features:
            df_extra['year'] = pd.DatetimeIndex(parsed_feature).year
        if 'month' in extra_datetime_features:
            df_extra['month'] = pd.DatetimeIndex(parsed_feature).month
        if 'dayOfMonth' in extra_datetime_features:
            df_extra['dayOfMonth'] = pd.DatetimeIndex(parsed_feature).day
        if 'dayOfWeek' in extra_datetime_features:
            df_extra['dayOfWeek'] = pd.DatetimeIndex(parsed_feature).dayofweek
        if 'quarter' in extra_datetime_features:
            df_extra['quarter'] = pd.DatetimeIndex(parsed_feature).quarter
        if 'hour' in extra_datetime_features:
            df_extra['hour'] = pd.DatetimeIndex(parsed_feature).hour
        if 'minute' in extra_datetime_features:
            df_extra['minute'] = pd.DatetimeIndex(parsed_feature).minute
        if 'second' in extra_datetime_features:
            df_extra['second'] = pd.DatetimeIndex(parsed_feature.second)
        return df_extra
        

    ''' Extends the dataframe by time-related features extracted from the datetime column. Deprecated'''
    @DeprecationWarning
    def _extract_datetime_features_old(self, df, datetime_features, extra_datetime_features):
        if len(datetime_features) > 2:
            # An exception should be raised
            raise Exception('Cannot parse date-time features, because there were more than 2 passed.')

        dt_format = '%Y-%m-%d %H:%M:%S'
        date_format = '%Y-%m-%d'
        time_format = '%H:%M:%S'

        for dt_feature in datetime_features:
            # Get this series from the data frame
            dt_series = df[dt_feature]

            # Try parsing it as date-only, time-only or full datetime column
            try:
                parsed_feature = pd.to_datetime(dt_series, format=date_format)
            except ValueError:
                pass
            else:
                if 'year' in extra_datetime_features:
                    df['ex_year'] = pd.DatetimeIndex(parsed_feature).year
                if 'month' in extra_datetime_features:
                    df['ex_month'] = pd.DatetimeIndex(parsed_feature).month
                if 'dayOfWeek' in extra_datetime_features:
                    df['ex_dayOfWeek'] = pd.DatetimeIndex(parsed_feature).dayofweek
                if 'quarter' in extra_datetime_features:
                    df['ex_quarter'] = pd.DatetimeIndex(parsed_feature).quarter
            try:
                parsed_feature = pd.to_datetime(parsed_feature, format=time_format)
            except ValueError:
                pass
            else:
                if 'hour' in extra_datetime_features:
                    df['ex_hour'] = pd.DatetimeIndex(parsed_feature).hour
                if 'minute' in extra_datetime_features:
                    df['ex_minute'] = pd.DatetimeIndex(parsed_feature).minute
                if 'second' in extra_datetime_features:
                    df['ex_second'] = pd.DatetimeIndex(parsed_feature.second)
            try:
                parsed_feature = pd.to_datetime(parsed_feature, format=dt_format)
            except ValueError:
                pass
            else:
                # Extract all information from this
                if 'year' in extra_datetime_features:
                    df['ex_year'] = pd.DatetimeIndex(parsed_feature).year
                if 'month' in extra_datetime_features:
                    df['ex_month'] = pd.DatetimeIndex(parsed_feature).month
                if 'dayOfWeek' in extra_datetime_features:
                    df['ex_dayOfWeek'] = pd.DatetimeIndex(parsed_feature).dayofweek
                if 'quarter' in extra_datetime_features:
                    df['ex_quarter'] = pd.DatetimeIndex(parsed_feature).quarter
                if 'hour' in extra_datetime_features:
                    df['ex_hour'] = pd.DatetimeIndex(parsed_feature).hour
                if 'minute' in extra_datetime_features:
                    df['ex_minute'] = pd.DatetimeIndex(parsed_feature).minute
                if 'second' in extra_datetime_features:
                    df['ex_second'] = pd.DatetimeIndex(parsed_feature.second)
        return df



    ''' Performs preprocessing of data set - stanardizes numerical features, encodes nominal, imputes missing
    values, etc.
    df - data frame
    datetime_feature - the name of the time-related feature
    predicted_feature - the name of the predicted feature
    nominal_features - set of nominal features of the data set
    extra_datetime_features - list of date/time features that are extracted from the time-related feature
                              (year, month, hour, minute,...)'''
    def preprocess(self, df, datetime_feature, predicted_feature, nominal_features={}, df_extra_datetime=pd.DataFrame()):

        # Sort dataframe by the time-related column
        df = df.sort_values(by=[datetime_feature])

        # Set of all features
        all_features = set(df.columns)
        
        # Set of numerical features
        numerical_features = all_features.difference(nominal_features, {datetime_feature})

        if predicted_feature not in numerical_features:
            raise Exception('Predicted feature not found among numerical features. Please choose the feature you'
                            ' want to predict from existing numerical columns.')

        processed_features = []

        # Choose a scaler to normalize/standardize numeric data
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = StandardScaler()
        # scaler = PowerTransformer()

        imputer_median = SimpleImputer(missing_values=nan, strategy='median', verbose=1)

        # For every numerical feature
        for feature_name in numerical_features:
            # Convert to numpy array
            feature = array(df[feature_name])
            # Reshape to a 2D array
            feature = feature.reshape((feature.shape[0], 1))
            # Impute missing values if necessary
            # if np.isnan(feature).any():
            if pd.isna(feature).any():
                feature = imputer_median.fit_transform(feature)
            # Normalize
            feature_sc = scaler.fit_transform(feature)
            processed_features.append(feature_sc)
            if feature_name == predicted_feature:
                # Remember the predicted feature and its scale
                self._pred_feature = feature_sc
                self._pred_feature_scaler = copy.deepcopy(scaler)
                self._pred_feature_ori = feature

        oneHotEncoder = OneHotEncoder()
        imputer_dominant = SimpleImputer(missing_values=nan, strategy='most_frequent')

        for feature_name in nominal_features:
            # Convert to numpy array
            feature = array(df[feature_name])
            # Reshape to a 2D array
            feature = feature.reshape((feature.shape[0], 1))
            # Impute missing values if necessary
            # if np.isnan(feature).any():
            if pd.isna(feature).any():
                feature = imputer_dominant.fit_transform(feature)
            # Encode to one hot
            feature = oneHotEncoder.fit_transform(feature).toarray()
            processed_features.append(feature)

        # Process extra time-related features

        # DEPRECATED
        # # Year - standardize (as numerical)
        # if 'year' in df_extra_datetime.columns:
        #     feature = array(df_extra_datetime['year'])
        #     feature = feature.reshape((feature.shape[0], 1))
        #     feature = scaler.fit_transform(feature)
        #     processed_features.append(feature)
        #
        # # Quarter - encode (as nominal)
        # if 'quarter' in df_extra_datetime.columns:
        #     feature = array(df_extra_datetime['quarter'])
        #     feature = feature.reshape((feature.shape[0], 1))
        #     feature = oneHotEncoder.fit_transform(feature).toarray()
        #     processed_features.append(feature)
        #
        # # Month, day of week, hour, minute, second - cyclical - convert to coordinates
        # # of consequent points on a circle in the coordinate system
        # for extra_feature in ['month', 'dayOfMonth', 'dayOfWeek', 'hour', 'minute', 'second']:
        #     if extra_feature in df_extra_datetime:
        #         feature = array(df['ex_' + extra_feature])
        #         feature = feature.reshape((feature.shape[0], 1))
        #         feature = feature - feature.min()
        #         feature = feature * 2.0 * np.pi / feature.max()
        #         feature_sin = np.sin(feature)
        #         feature_cos = np.cos(feature)
        #         processed_features.append(feature_sin)
        #         processed_features.append(feature_cos)


        # For every extra datetime-related feature extracted
        for extra_feature in df_extra_datetime.columns:
            feature = array(df_extra_datetime[extra_feature])
            feature = feature.reshape((feature.shape[0], 1))
            # Year - standardize (as numerical)
            if extra_feature == 'year':
                feature = scaler.fit_transform(feature)
                processed_features.append(feature)
            # Quarter - encode (as nominal)
            elif extra_feature == 'quarter':
                feature = oneHotEncoder.fit_transform(feature).toarray()
                processed_features.append(feature)
            # Others - as cyclical - convert to coordinates of consequent
            # points on a circle in the coordinate system
            else:
                feature = feature - feature.min()
                feature = feature * 2.0 * np.pi / feature.max()
                feature_sin = np.sin(feature)
                feature_cos = np.cos(feature)
                processed_features.append(feature_sin)
                processed_features.append(feature_cos)


        # Stack horizontally to get a 2D array
        processed_sequence = hstack(processed_features)

        return processed_sequence

    ''' Makes prediction and evaluates trained timeseries model. Plots results of prediction'''
    def predict_evaluate(self, ts_model, X_test, y_test):
        # Inverse scaling
        # y_test = self._pred_feature_scaler.inverse_transform(y_test)
        y_dash = ts_model.predict(X_test)
        # y_dash = self._pred_feature_scaler.inverse_transform(y_dash)
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

    def model_generation_summary(self):
        string = ''
        string += 'Forecasting model used: {}<br>'.format(self.timeseries_model.model_name)
        string += 'Generation time: %.3f s\n' % self._model_gen_time
        return string
