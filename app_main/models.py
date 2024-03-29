from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import LSTM
import keras.regularizers
from numpy import array
from numpy import hstack
import pandas as pd
import numpy as np
import copy
import sklearn.model_selection as sk_ms
import tensorflow as tf
tf_graph = tf.get_default_graph()


class TimeseriesModel:

    def __init__(self, predicted_feature, nominal_features, datetime_feature, extra_datetime_features,
                 n_steps_in, n_steps_out, data_filename, scale_predicted_feature=True, pred_feature_on_input=False):
        self.predicted_feature = predicted_feature
        self._nominal_features = nominal_features
        self._datetime_feature = datetime_feature
        self._extra_datetime_features = extra_datetime_features
        self._n_steps_in = n_steps_in
        self._n_steps_out = n_steps_out
        self._keras_model = None
        self._pred_feature_scaler = None
        self.data_filename = data_filename
        self._scale_predicted_feature = scale_predicted_feature
        self._pred_feature_on_input = pred_feature_on_input
        self._datetime_array = None
        self._differentiate_predicted_feature = False
        # Initialize own session and graph for the keras model
        # self.__session = tf.Session()
        # self.__graph = tf.get_default_graph()

    def get_model_info(self):
        return '{} used for \'{}\' data set'.format(self.model_name, self.data_filename)

    ''' Prepares data from a data frame for training/testing'''

    def prepare_data(self, df, mode):
        df_extra_datetime = self._extract_datetime_features(df)
        if mode == 'training':
            processed_sequence, target_feature = self._preprocess(df, df_extra_datetime, mode)
            X, y = self._split_sequence(processed_sequence, target_feature, mode)
            return X, y
        elif mode == 'testing':
            processed_sequence, _ = self._preprocess(df, df_extra_datetime, mode)
            X, _ = self._split_sequence(processed_sequence, None, mode)
            return X, None
        else:
            raise Exception('Invalid mode in prepare_data() method. Given \'{}\', '
                            'but should be either \'training\' or \'testing\'.'.format(mode))

    ''' Returns a new dataframe of time-related features extracted from the datetime column
        (year, quarter, month, day of month, day of week, hour, minute, second'''

    def _extract_datetime_features(self, df):
        # Parse to datetime. If fails, an exception will be raised
        parsed_feature = pd.to_datetime(df[self._datetime_feature])
        self._datetime_array = parsed_feature
        df_extra = pd.DataFrame()
        # Check for every label of extra feature, if is present in the list
        # If so, add an extra column to the dataframe
        if 'year' in self._extra_datetime_features:
            df_extra['year'] = pd.DatetimeIndex(parsed_feature).year
        if 'month' in self._extra_datetime_features:
            df_extra['month'] = pd.DatetimeIndex(parsed_feature).month
        if 'dayOfMonth' in self._extra_datetime_features:
            df_extra['dayOfMonth'] = pd.DatetimeIndex(parsed_feature).day
        if 'dayOfWeek' in self._extra_datetime_features:
            df_extra['dayOfWeek'] = pd.DatetimeIndex(parsed_feature).dayofweek
        if 'quarter' in self._extra_datetime_features:
            df_extra['quarter'] = pd.DatetimeIndex(parsed_feature).quarter
        if 'hour' in self._extra_datetime_features:
            df_extra['hour'] = pd.DatetimeIndex(parsed_feature).hour
        if 'minute' in self._extra_datetime_features:
            df_extra['minute'] = pd.DatetimeIndex(parsed_feature).minute
        if 'second' in self._extra_datetime_features:
            df_extra['second'] = pd.DatetimeIndex(parsed_feature.second)
        return df_extra

    ''' Performs preprocessing of data set - stanardizes numerical features, encodes nominal, imputes missing
       values, etc.
       df - data frame
       datetime_feature - the name of the time-related feature
       predicted_feature - the name of the predicted feature
       nominal_features - set of nominal features of the data set
       extra_datetime_features - list of date/time features that are extracted from the time-related feature
                                 (year, month, hour, minute,...)'''
    def _preprocess(self, df, df_extra_datetime, mode):

        # Sort dataframe by the time-related column
        # df = df.sort_values(by=[self._datetime_feature])

        # Set of all features
        all_features = set(df.columns)

        # Set of numerical features = all \ (nominal U {datetime})
        numerical_features = all_features.difference(self._nominal_features, {self._datetime_feature})

        if not self._pred_feature_on_input:
            # Do not use predicted feature as input
            numerical_features = numerical_features.difference({self.predicted_feature})

        # if self.predicted_feature not in numerical_features:
        #     raise Exception('Predicted feature not found among numerical features. Please choose the feature you'
        #                     ' want to predict from existing numerical columns.')

        processed_features = []

        # Choose a scaler to normalize/standardize numeric data
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = StandardScaler()
        imputer_median = SimpleImputer(missing_values=np.nan,
                                       strategy='median')
        # For every numerical feature
        for feature_name in numerical_features:
            # Convert to numpy array
            feature = array(df[feature_name])
            # Reshape to a 2D array
            feature = feature.reshape((feature.shape[0], 1))
            # Impute missing values if necessary
            if pd.isna(feature).any():
                feature = imputer_median.fit_transform(feature)
            # Normalize
            feature_sc = scaler.fit_transform(feature)
            processed_features.append(feature_sc)

        oneHotEncoder = OneHotEncoder()
        imputer_dominant = SimpleImputer(missing_values=np.nan,
                                         strategy='most_frequent')
        for feature_name in self._nominal_features:
            # Convert to numpy array
            feature = array(df[feature_name])
            # Reshape to a 2D array
            feature = feature.reshape((feature.shape[0], 1))
            # Impute missing values if necessary
            if pd.isna(feature).any():
                feature = imputer_dominant.fit_transform(feature)
            # Encode to one hot
            feature = oneHotEncoder.fit_transform(feature).toarray()
            processed_features.append(feature)

        # Process extra time-related features
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

        if mode == 'training':
            # Preprocess the target feature
            target_feature = array(df[self.predicted_feature])

            # Make time series stationary - differentiate elements
            if self._differentiate_predicted_feature:
                target_shifted = np.roll(target_feature, 1)
                target_feature = target_feature - target_shifted

            target_feature = target_feature.reshape((target_feature.shape[0], 1))
            if pd.isna(target_feature).any():
                target_feature = imputer_median.fit_transform(target_feature)
            if self._scale_predicted_feature:
                target_feature = scaler.fit_transform(target_feature)
                # Remember the scaler in order to use it back in prediction
                self._pred_feature_scaler = copy.deepcopy(scaler)
        else:
            target_feature = None

        # Stack horizontally to get a 2D array
        processed_sequence = hstack(processed_features)

        # Return the processed sequence and separately the target feature
        return processed_sequence, target_feature

    ''' Splits prepared sequence into input and output series (X and y) for the model
    sequence - 2D array of preprocessed features
    target_feature - preprocessed array of the feature we predict
    mode - 'training' or 'testing' '''
    def _split_sequence(self, sequence, target_feature, mode):
        pass

    ''' Splits input/output sequences into training and testing data'''
    def train_test_split(self, X, y, test_size=0.1):
        pass

    ''' Builds inner network depending on parameters given on initialization'''
    def build_network(self):
        global tf_graph
        with tf_graph.as_default():
            self._build_network()

    ''' Compiles inner Keras model according to used type of neural network'''
    def _build_network(self):
        pass

    def train_model(self, X, y, epochs):
        global tf_graph
        with tf_graph.as_default():
            self._keras_model.fit(X, y, epochs=epochs, verbose=2)

    def predict(self, X):
        global tf_graph
        with tf_graph.as_default():
            y_dash = self._keras_model.predict(X, verbose=1)
        # If the target feature was scaled, then we have to transform it back
        if self._scale_predicted_feature:
            return self._pred_feature_scaler.inverse_transform(y_dash)
        else:
            # If not, return original output
            return y_dash

    def rescale_target_feature(self, feature):
        return self._pred_feature_scaler.inverse_transform(feature)

    def get_datetime_array(self):
        return self._datetime_array

    def get_train_test_size(self, X_train, X_test):
        return X_train.shape[0], X_test.shape[0]

    def get_model_name(self):
        pass


''' Multi Layer Perceptron model for time series'''


class TimeseriesMLPModel(TimeseriesModel):

    model_name = 'MLP'

    def _split_sequence(self, sequence, target_feature, mode):
        self._n_outputs = self._n_steps_out
        self._n_inputs = sequence.shape[1] * self._n_steps_in
        if mode == 'training':
            X, y = list(), list()
            for i in range(0, len(sequence) - self._n_steps_in - self._n_steps_out):
                # End index of the input series
                idx_in_end = i + self._n_steps_in
                # End index of the output series
                idx_out_end = idx_in_end + self._n_steps_out
                # Append rows from the sequence and flatten them
                X.append(sequence[i: idx_in_end, :].flatten())
                y.append(target_feature[idx_in_end: idx_out_end].flatten())
            return np.array(X), np.array(y)
        else:
            X = list()
            for i in range(0, len(sequence) - self._n_steps_in - self._n_steps_out):
                # End index of the input series
                idx_in_end = i + self._n_steps_in
                # Append rows from the sequence and flatten them
                X.append(sequence[i: idx_in_end, :].flatten())
            return array(X), None

    def train_test_split(self, X, y, test_size=0.1):
        return sk_ms.train_test_split(X, y, test_size=test_size, shuffle=True)

    def _build_network(self):

        self._keras_model = Sequential(name=self.model_name)
        self._keras_model.add(Dense(64, input_dim=self._n_inputs, activation='relu',
                                    kernel_regularizer=keras.regularizers.l1(0.01)))
        self._keras_model.add(Dense(32, activation='relu',
                                    kernel_regularizer=keras.regularizers.l1(0.01)))
        self._keras_model.add(Dense(16, activation='relu',
                                    kernel_regularizer=keras.regularizers.l1(0.01)))
        self._keras_model.add(Dense(self._n_outputs, activation='linear'))
        self._keras_model.compile(optimizer=Adadelta(), loss='mae')
        print(self._keras_model.summary())
        print('Input nodes:', self._n_inputs)


    def get_model_name(self):
        return 'MLP'


''' Convolutional Neural Network for time series model '''

class TimeseriesCNNModel(TimeseriesModel):

    model_name = 'CNN'

    def _split_sequence(self, sequence, target_feature, mode):
        self._n_steps_in = self._n_steps_in
        self._n_features_in = sequence.shape[1]
        self._n_outputs = self._n_steps_out
        if mode == 'training':
            X, y = list(), list()
            for i in range(0, len(sequence) - self._n_steps_in - self._n_steps_out):
                # End index of the input series
                idx_in_end = i + self._n_steps_in
                # End index of the output series
                idx_out_end = idx_in_end + self._n_steps_out
                # Append rows from the sequence - without flattening
                X.append(sequence[i: idx_in_end, :])
                y.append(target_feature[idx_in_end: idx_out_end].flatten())
            return array(X), array(y)
        else:
            X = list()
            for i in range(0, len(sequence) - self._n_steps_in - self._n_steps_out):
                # End index of the input series
                idx_in_end = i + self._n_steps_in
                X.append(sequence[i: idx_in_end, :])
            return array(X), None

    def train_test_split(self, X, y, test_size=0.1):
        return sk_ms.train_test_split(X, y, test_size=test_size, shuffle=True)

    def _build_network(self):
        kernel_size = min(6, self._n_steps_in - 2)
        self._keras_model = Sequential()
        self._keras_model.add(Conv1D(filters=32, kernel_size=kernel_size, activation='relu',
                       input_shape=(self._n_steps_in, self._n_features_in)))
        self._keras_model.add(MaxPooling1D())
        self._keras_model.add(Flatten())
        self._keras_model.add(Dense(units=32, activation='relu'))
        self._keras_model.add(Dense(units=16, activation='linear'))
        self._keras_model.add(Dense(units=self._n_outputs))
        self._keras_model.compile(optimizer=Adadelta(), loss='mae')
        print(self._keras_model.summary())
        print('Input shape:', self._n_steps_in, 'steps x', self._n_features_in, 'features')

    def get_model_name(self):
        return 'CNN Simple'


''' Long-Short Term Memory model for time series'''

class TimeseriesLSTMModel(TimeseriesModel):

    model_name = 'LSTM'

    def _split_sequence(self, sequence, target_feature, mode):
        self._n_steps_in = self._n_steps_in
        self._n_features_in = sequence.shape[1]
        self._n_outputs = self._n_steps_out
        if mode == 'training':
            X, y = list(), list()
            for i in range(0, len(sequence) - self._n_steps_in - self._n_steps_out):
                # End index of the input series
                idx_in_end = i + self._n_steps_in
                # End index of the output series
                idx_out_end = idx_in_end + self._n_steps_out
                # Append rows from the sequence - without flattening
                X.append(sequence[i: idx_in_end, :])
                y.append(target_feature[idx_in_end: idx_out_end].flatten())
            return array(X), array(y)
        else:
            X = list()
            for i in range(0, len(sequence) - self._n_steps_in - self._n_steps_out):
                # End index of the input series
                idx_in_end = i + self._n_steps_in
                X.append(sequence[i: idx_in_end, :])
            return array(X), None

    def train_test_split(self, X, y, test_size=0.1):
        return sk_ms.train_test_split(X, y, test_size=test_size, shuffle=True)

    def _build_network(self):
        self._keras_model = Sequential()
        self._keras_model.add(LSTM(units=24, activation='relu', input_shape=
                    (self._n_steps_in,self._n_features_in),
                             return_sequences=True))
        self._keras_model.add(LSTM(units=24, activation='relu', return_sequences=True))
        self._keras_model.add(LSTM(units=24, activation='linear'))
        # self._keras_model.add(Dense(units=24, activation='relu'))
        self._keras_model.add(Dense(units=self._n_outputs))
        self._keras_model.compile(optimizer=Adam(lr=0.002), loss='mae')
        print(self._keras_model.summary())
        print('Input shape:', self._n_steps_in, 'steps x', self._n_features_in, 'features')

    def get_model_name(self):
        return 'LSTM'


