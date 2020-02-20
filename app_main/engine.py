import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import app_main.models as models
import keras.backend as K
import pickle


np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=8)
np.set_printoptions(linewidth=100)
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 9)


def make_keras_picklable():
    import tempfile
    import keras.models

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(delete=True) as fd:
            fd.close()
            keras.models.save_model(self, fd.name)
            with open(fd.name, 'rb') as fd:
                model_str = fd.read()
                d = {'model_str': model_str}
                return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(delete=True) as fd:
            fd.close()
            with open(fd.name, 'wb') as fd:
                fd.write(state['model_str'])
                fd.flush()
                model = keras.models.load_model(fd.name)
                self.__dict__ = model.__dict__

                cls = keras.models.Model
                cls.__getstate__ = __getstate__
                cls.__setstate__ = __setstate__


class ForecasterEngine:

    def __init__(self):
        self.timeseries_model = None
        self.is_model_ready = False
        self.train_df = None
        self.test_df = None
        make_keras_picklable()


    ''' Generates a Deep Learning model for forecasting based on given data frame and options
    model_type - name of the model type returned by radio items from the layout
    nominal_features - list of names of features in the dataframe that are nominal
    predicted_feature - the name of the feature we want to predict
    datetime_feature - the name of the time-related feature
    extra_datetime_feature - list of names of extra date/time features to be extracted from the
                             time-related feature (year, month, hour,...)
    n_steps_in - number of input steps of the time series for the model
    n_steps_out - number of output steps of the time series to be predicted
    n_train_epochs - number of epochs of model training
    data_filename - name of the file that contained the trainig data'''
    def init_model(self, model_type, predicted_feature, nominal_features, datetime_feature, extra_datetime_features,
                   n_steps_in, n_steps_out, data_filename):

        if self.train_df is None:
            raise Exception('No data set found to generate model.')

        # Parse to sets
        nominal_features = {} if nominal_features is None else set(nominal_features)

        if extra_datetime_features is None:
            extra_datetime_features = []

        if predicted_feature in nominal_features:
            raise Exception('Predicted feature cannot be nominal. The model predicts only numerical features.')
        if predicted_feature == datetime_feature:
            raise Exception('The datetime feature cannot be predicted. Use one of the numerical features in the '
                            'data set for prediction.')

        # Clear tensorflow session
        K.clear_session()

        # Choose the model type
        if model_type == 'MLP':
            self.timeseries_model = models.TimeseriesMLPModel(predicted_feature, nominal_features, datetime_feature,
                                                              extra_datetime_features, n_steps_in, n_steps_out,
                                                              data_filename, scale_predicted_feature=True)
        elif model_type == 'CNN':
            self.timeseries_model = models.TimeseriesCNNModel(predicted_feature, nominal_features, datetime_feature,
                                                              extra_datetime_features, n_steps_in, n_steps_out,
                                                              data_filename, scale_predicted_feature=True)
        elif model_type == 'LSTM':
            self.timeseries_model = models.TimeseriesLSTMModel(predicted_feature, nominal_features, datetime_feature,
                                                               extra_datetime_features, n_steps_in, n_steps_out,
                                                               data_filename, scale_predicted_feature=True)
        else:
            raise Exception('Error initializing model. Model type either not specified or invalid. '
                            'Got model type: \'{}\''.format(model_type))

        # Prepare data for training
        X, y = self.timeseries_model.prepare_data(self.train_df, mode='training')
        print('Model inputs/outputs:')
        for i in range(10):
            print(i, ':', X[i], '<--', y[i])

        # Compile the model
        self.timeseries_model.build_network()

        # Return processed input/output sequences
        return X, y

    # Loads ready model
    def load_model(self, new_model):
        K.clear_session()
        self.timeseries_model = new_model

    # Splits the dataset into training and testing part
    def split_train_test(self, dataset):
        # Specify test size based on the size of full sequence
        # Minimum (30% of full sequence, 1000)
        test_size = 0.3 if len(dataset[0]) < 3333 else 1000
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=test_size, shuffle=False)
        # If the target feature was scaled for training, rescale it now
        if self.timeseries_model._scale_predicted_feature:
            y_test = self.timeseries_model.rescale_target_feature(y_test)
        # Return two tuples - training set and testing set
        return (X_train, y_train), (X_test, y_test)

    ''' Trains the timeseries model
    train_set - preprocessed training set
    epochs - number of epochs of training'''
    def train_model(self, train_set, epochs):
        self.timeseries_model.train_model(train_set[0], train_set[1], epochs)

    def evaluate_model(self, test_set):
        y_dash = self.timeseries_model.predict(test_set[0])
        evaluator = ModelEvaluator()
        evaluator.predict_evaluate(test_set[1], y_dash)
        return evaluator

    def make_prediction(self):
        if self.test_df is None:
            raise Exception('No test data set loaded for prediction!')
        if self.timeseries_model is None:
            raise Exception('No timeseries model in use to make a prediction!')
        X, _ = self.timeseries_model.prepare_data(self.test_df, mode='testing')
        return self.timeseries_model.predict(X)

    # Loads model from pickled string
    def load_model_unpickle(self, string):
        # Use global tensorflow graph when unpickling keras model
        with models.tf_graph.as_default():
            self.timeseries_model = pickle.loads(string)

    def get_model_info(self):
        return self.timeseries_model.get_model_info()


class ModelEvaluator:

    def __init__(self):
        self.mae = None
        self.rmse = None
        self.mape = None
        self.r2_score = None
        self.rmspe = None
        self.y_test = None
        self.y_dash = None

    ''' Makes prediction and evaluates trained timeseries model. Plots results of prediction'''

    def predict_evaluate(self, y_test, y_dash):
        self.y_test = copy.deepcopy(y_test)
        self.y_dash = copy.deepcopy(y_dash)
        print('\nPrediction / actual results:')
        for i in range(50):
            print(i + 1, '. ', y_dash[i], ' - ', y_test[i], sep='')
        print('...')
        print('\nEvaluation metrics:')
        # Mean Absolute Error
        self.mae = mean_absolute_error(y_test, y_dash)
        print('Mean absolute error: %.3f' % self.mae)
        # Root Mean Squared Error
        self.rmse = np.sqrt(mean_squared_error(y_test, y_dash))
        print('Root mean squared error: %.3f' % self.rmse)
        # R^2 Score
        self.r2_score = r2_score(y_test, y_dash)
        print('R2 Score: %.4f' % self.r2_score)
        # Mean Absolute Percentage Error
        self.mape = 100 * np.mean(np.abs(1 - y_dash / y_test))
        print('Mean absolute percentage error: %.3f' % self.mape, '%', sep='')
        print('y_test.shape:', y_test.shape)
        # If we predicted only one step
        if y_test.shape[-1] == 1:
            tests_scores = np.stack(
                (np.arange(len(y_test)), y_dash.flatten(), y_test.flatten(), np.abs(y_test - y_dash).flatten()), axis=-1)
            tests_scores = tests_scores[tests_scores[:, 3].argsort()]
            print('\nTop 5 predictions:')
            for ts in tests_scores[:5]:
                print(int(ts[0]), '. %.3f' % ts[1], ' - ', ts[2], ' (error=%.3f)' % ts[3], sep='')
