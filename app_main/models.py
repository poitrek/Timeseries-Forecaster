from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import Nadam
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import LSTM
from numpy import array
from numpy import hstack
import numpy as np
import sklearn.model_selection as sk_ms
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True


class TimeseriesModel:

    def __init__(self):
        self._model = None
        self._n_epochs_used = None

    ''' Splits preprocessed datasets into input (X) and output (y) sequences for the model'''
    # def split_sequence(self, sales_data, store_data, n_steps_in, n_steps_out):
    #     pass

    ''' Splits prepared sequence into input and output series (X and y) for the model
    sequence - 2D array of preprocessed features
    n_steps_in - number of input steps on which prediction depends
    n_steps_out - number of output steps we want to predict
    pred_feature - preprocessed array of the feature we predict'''
    def split_sequence(self, sequence, n_steps_in, n_steps_out, pred_feature):
        X, y = list(), list()
        for i in range(0, len(sequence - n_steps_in - n_steps_out)):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            X.append(sequence[i: idx_in_end])
            y.append(pred_feature[idx_in_end: idx_out_end])
        return array(X), array(y)

    ''' Splits input/output sequences into training and testing data'''
    def train_test_split(self, X, y, test_size=0.1):
        pass

    ''' Compiles inner Keras model according to the number
        of steps and features on both input and output'''
    def compile_model(self):
        pass

    def train_model(self, X, y, epochs=300):
        self._model.fit(X, y, epochs=epochs, verbose=2)
        self._n_epochs_used = epochs

    def evaluate(self, X, y):
        return self._model.evaluate(X, y, verbose=1)

    def predict(self, X):
        return self._model.predict(X, verbose=1)

    def get_train_test_size(self, X_train, X_test):
        return X_train.shape[0], X_test.shape[0]

    def get_model_name(self):
        pass


''' MLP model for timeseries forecasting'''
class TimeseriesMLPModel(TimeseriesModel):

    model_name = 'MLP'
    model_acronym = 'MLP'

    def __init__(self):
        super().__init__()
        self._n_inputs = None
        self._n_outputs = None

    ''' Splits prepared sequence into input and output series (X and y) for the model
    sequence - 2D array of preprocessed features
    n_steps_in - number of input steps on which prediction depends
    n_steps_out - number of output steps we want to predict
    pred_feature - preprocessed array of the feature we predict'''
    def split_sequence(self, sequence, n_steps_in, n_steps_out, pred_feature):
        self._n_outputs = n_steps_out
        self._n_inputs = sequence.shape[1] * n_steps_in
        X, y = list(), list()
        for i in range(0, len(sequence) - n_steps_in - n_steps_out):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            # Append rows from the sequence and flatten them
            X.append(sequence[i: idx_in_end, :].flatten())
            # Xarr = np.array(X)
            y.append(pred_feature[idx_in_end: idx_out_end].flatten())
        return np.array(X), np.array(y)

    # @DeprecationWarning('You are calling an old (probably not working) version of the split_sequence method.')
    def split_sequence_old(self, sales_data, store_data, n_steps_in, n_steps_out):
        self._n_outputs = n_steps_out
        X, y = list(), list()
        # Number of columns
        for i in range(0, len(sales_data) - n_steps_in - n_steps_out):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            # If the first and the last element of the sample have different store ids
            if store_data[i, 0] != store_data[idx_out_end - 1, 0]:
                # Jump to the row with different (next) store id
                i = idx_out_end - 1
            else:
                # Input: all columns from sales data except for original sales count (last element)
                # And all columns from store data except for store id (first element)
                sample = sales_data[i: idx_in_end, :-1].flatten()
                sample = np.concatenate((sample, store_data[i, 1:]))
                X.append(sample)
                # X.append(sales_data[i: idx_in_end, :-1])
                # X.append(store_data[i, 1:])
                # Output: original sales count (last column)
                y.append(sales_data[idx_in_end: idx_out_end, -1])
        X = array(X)
        self._n_inputs = X.shape[1]
        return X, array(y)

    def train_test_split(self, X, y, test_size=0.1):
        return sk_ms.train_test_split(X, y, test_size=test_size, shuffle=True)

    def compile_model(self):

        self._model = Sequential(name=self.model_name)
        self._model.add(Dense(64, input_dim=self._n_inputs, activation='relu'))
        self._model.add(Dense(32, activation='relu'))
        self._model.add(Dense(16, activation='relu'))
        # self._model.add(Dense(16, activation='relu'))
        self._model.add(Dense(self._n_outputs, activation='relu'))
        self._model.compile(optimizer=Adadelta(), loss='mae')
        print(self._model.summary())
        print('Input nodes:', self._n_inputs)


    def get_model_name(self):
        return 'MLP'


''' Typical Convolutional Neural Network model '''
class TimeseriesCNN_SimpleModel(TimeseriesModel):

    model_name = 'CNN Simple'
    model_acronym = 'CNN_Simp'

    def __init__(self):
        super().__init__()
        self._n_outputs = None
        self._n_steps_in = None
        self._n_features_in = None

    def split_sequence(self, sequence, n_steps_in, n_steps_out, pred_feature):
        self._n_steps_in = n_steps_in
        self._n_features_in = sequence.shape[1]
        self._n_outputs = n_steps_out
        X, y = list(), list()
        for i in range(0, len(sequence - n_steps_in - n_steps_out)):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            # Append rows from the sequence - without flattening
            X.append(sequence[i: idx_in_end])
            y.append(pred_feature[idx_in_end: idx_out_end])
        return array(X), array(y)

    ''' Splits sequences to a simple, 3-dimensional sequence, the same way for sales data
    and store data, making the latter redundant in every sample'''
    def split_sequence_old(self, sales_data, store_data, n_steps_in, n_steps_out):
        self._n_steps_in = n_steps_in
        self._n_outputs = n_steps_out
        # Number of input features
        self._n_features_in = sales_data.shape[1] + store_data.shape[1] - 2
        X, y = list(), list()
        for i in range(0, len(sales_data) - n_steps_in - n_steps_out):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            # If the first and the last element of the sample have different store ids
            if store_data[i, 0] != store_data[idx_out_end - 1, 0]:
                # Jump to the row with different (next) store id
                i = idx_out_end - 1
            else:
                # Stack sales and store data to one sequence (input sample)
                X.append(hstack((sales_data[i: idx_in_end, :-1], store_data[i: idx_in_end, 1:])))
                y.append(sales_data[idx_in_end: idx_out_end, -1])
        return array(X), array(y)

    def train_test_split(self, X, y, test_size=0.1):
        return sk_ms.train_test_split(X, y, test_size=test_size, shuffle=True)

    def compile_model(self):
        kernel_size = min(6, self._n_steps_in - 2)
        self._model = Sequential()
        self._model.add(Conv1D(filters=32, kernel_size=kernel_size, activation='relu',
                       input_shape=(self._n_steps_in, self._n_features_in)))
        self._model.add(MaxPooling1D())
        self._model.add(Flatten())
        self._model.add(Dense(units=32, activation='relu'))
        self._model.add(Dense(units=16, activation='relu'))
        self._model.add(Dense(units=self._n_outputs))
        self._model.compile(optimizer=Adadelta(), loss='mae')
        print(self._model.summary())
        print('Input shape:', self._n_steps_in, 'steps x', self._n_features_in, 'features')

    def get_model_name(self):
        return 'CNN Simple'


''' Hybrid CNN model that applies convolution only for the sales data,
    then concatenates it with store data and processes together with fully
    connected Dense layers'''
class TimeseriesCNN_HybridModel(TimeseriesModel):

    model_name = 'CNN-MLP Hybrid'
    model_acronym = 'CNN_Hybr'

    # def __init__(self, sales_data_shape, store_data_shape):
    #     self.input_sales_length = sales_data_shape[1] - 1
    #     self.input_store_length = store_data_shape[1] - 1

    def split_sequence(self, sales_data, store_data, n_steps_in, n_steps_out):
        # Save length of sales and store input data
        self.input_sales_length = sales_data.shape[1] - 1
        self.input_store_length = store_data.shape[1] - 1
        self.n_steps_in = n_steps_in
        self._n_outputs = n_steps_out
        X_first, X_second = list(), list()
        y = list()
        for i in range(0, len(sales_data) - n_steps_in - n_steps_out):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            # If the first and the last element of the sample have different store ids
            if store_data[i, 0] != store_data[idx_out_end - 1, 0]:
                # Jump to the row with different (next) store id
                i = idx_out_end - 1
            else:
                # Input sequence consists of two "columns":
                # Data from sales (two-dimensional)
                X_first.append(sales_data[i: idx_in_end, :-1])
                # Data from store (one-dimensional)
                X_second.append(store_data[i, 1:])
                # Concatenate these parts
                # X.append([X_first, X_second])
                y.append(sales_data[idx_in_end: idx_out_end, -1])
        return [array(X_first), array(X_second)], array(y)

    def train_test_split(self, X, y, test_size=0.1):
        # X[0] = X[0][..., np.newaxis]
        # X[1] = X[1][..., np.newaxis]
        x0_train, x0_test, x1_train, x1_test, y_train, y_test = sk_ms.train_test_split(X[0], X[1], y, test_size=test_size, shuffle=True)
        return [x0_train, x1_train], [x0_test, x1_test], y_train, y_test

    def get_train_test_size(self, X_train, X_test):
        return X_train[0].shape[0], X_test[0].shape[0]

    def compile_model(self):
        conv_kernel_size_def = 5
        conv_kernel_size = min(conv_kernel_size_def, self.n_steps_in - 1)
        # First input - for the sales data
        input_conv = Input(shape=(self.n_steps_in, self.input_sales_length))
        # Define the convolutional part for sales data
        conv_model = Conv1D(filters=16, kernel_size=conv_kernel_size, activation='relu')(input_conv)
        conv_model = MaxPooling1D()(conv_model)
        # conv_model = Conv1D(filters=8, kernel_size=2, activation='relu')(conv_model)
        # conv_model = MaxPooling1D()(conv_model)
        conv_model = Flatten()(conv_model)

        # Second input - for the store data
        input_mlp = Input(shape=(self.input_store_length,))
        # Merge these two layers
        merge = concatenate([conv_model, input_mlp])
        # Define the MLP part for all of attributes
        mlp_model = Dense(units=32, activation='relu')(merge)
        mlp_model = Dense(units=8, activation='relu')(mlp_model)
        output = Dense(units=self._n_outputs, activation='relu')(mlp_model)
        # Define the actual model
        self._model = Model(inputs=[input_conv, input_mlp], outputs=output)
        self._model.compile(optimizer=Adadelta(), loss='mae')
        print(self._model.summary())
        print('Input (sales):', self.n_steps_in, 'steps x', self.input_sales_length, 'features')
        print('Input (store):', self.input_store_length, 'features')

    def get_model_name(self):
        return 'CNN-MLP Hybrid'


class TimeseriesLSTMModel(TimeseriesModel):

    model_name = 'LSTM'
    model_acronym = 'LSTM'

    def __init__(self):
        super().__init__()
        self._n_outputs = None
        self._n_steps_in = None
        self._n_features_in = None

    def split_sequence(self, sequence, n_steps_in, n_steps_out, pred_feature):
        self._n_steps_in = n_steps_in
        self._n_features_in = sequence.shape[1]
        self._n_outputs = n_steps_out
        X, y = list(), list()
        for i in range(0, len(sequence - n_steps_in - n_steps_out)):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            # Append rows from the sequence - without flattening
            X.append(sequence[i: idx_in_end])
            y.append(pred_feature[idx_in_end: idx_out_end])
        return array(X), array(y)

    def split_sequence_old(self, sales_data, store_data, n_steps_in, n_steps_out):
        self.n_steps_in = n_steps_in
        self._n_outputs = n_steps_out
        # Number of input features
        self._n_features_in = sales_data.shape[1] + store_data.shape[1] - 2
        X, y = list(), list()
        for i in range(0, len(sales_data) - n_steps_in - n_steps_out):
            # End index of the input series
            idx_in_end = i + n_steps_in
            # End index of the output series
            idx_out_end = idx_in_end + n_steps_out
            # If the first and the last element of the sample have different store ids
            if store_data[i, 0] != store_data[idx_out_end - 1, 0]:
                # Jump to the row with different (next) store id
                i = idx_out_end - 1
            else:
                # Stack sales and store data to one sequence (input sample)
                X.append(hstack((sales_data[i: idx_in_end, :-1], store_data[i: idx_in_end, 1:])))
                y.append(sales_data[idx_in_end: idx_out_end, -1])
        return array(X), array(y)

    def train_test_split(self, X, y, test_size=0.1):
        return sk_ms.train_test_split(X, y, test_size=test_size, shuffle=True)

    def compile_model(self):
        self._model = Sequential()
        self._model.add(LSTM(units=24, activation='relu', input_shape=(self._n_steps_in, self._n_features_in),
                             return_sequences=True))
        self._model.add(LSTM(units=24, activation='relu', return_sequences=True))
        self._model.add(LSTM(units=24, activation='relu'))
        # self._model.add(Dense(units=24, activation='relu'))
        self._model.add(Dense(units=self._n_outputs))
        self._model.compile(optimizer=Adam(lr=0.002), loss='mae')
        print(self._model.summary())
        print('Input shape:', self._n_steps_in, 'steps x', self._n_features_in, 'features')

    def get_model_name(self):
        return 'LSTM'


