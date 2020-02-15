import time
from datetime import datetime
import os


class RunResult:

    # def __init__(self, mae, rmse, r2_score, mape):
    #
    #     self.mean_absolute_error = mae
    #     self.root_mean_squared_error = rmse
    #     self.r2_score = r2_score
    #     self.mean_absolute_percentage_error = mape
    #
    # def __init__(self, n_stores, n_steps_in, n_steps_out, ):
    #     pass

    def set_conditions(self, n_stores, n_steps_in, n_steps_out, n_train_samples, n_test_samples):
        self.n_stores = n_stores
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples

    def set_learning_info(self, timeseries_model, training_time):
        self.model_name = timeseries_model.model_name
        self.model_acronym = timeseries_model.model_acronym
        self.training_time = training_time
        string_list = []
        timeseries_model.model.summary(print_fn=lambda x: string_list.append(x))
        self.model_summary = '\n'.join(string_list)


    def set_prediction_scores(self, mae, rmse, r2_score, mape, plot_figure):
        self.mean_absolute_error = mae
        self.root_mean_squared_error = rmse
        self.r2_score = r2_score
        self.mean_absolute_percentage_error = mape
        self.plot_figure = plot_figure

    def summary(self):
        string = '----------------- Timeseries prediction summary -----------------\n\n'
        string += 'Number of stores included: %d\n' % self.n_stores
        string += 'Number of steps predicted: %d\n' % self.n_steps_out
        string += 'Number of training samples: %d\n' % self.n_train_samples
        string += 'Used NN model: ' + self.model_name + '\n'
        string += 'Model summary:\n'
        string += self.model_summary + '\n'
        string += 'Number of past steps used for prediction: %d\n' % self.n_steps_in
        string += 'Training time: %.3f s\n' % self.training_time
        string += 'Number of testing samples: %d\n' % self.n_test_samples
        string += 'Prediction metrics:\n'
        string += 'Mean Absolute Error (MAE): %.3f\n' % self.mean_absolute_error
        string += 'Root Mean Squaret Error (RMSE): %.3f\n' % self.root_mean_squared_error
        string += 'R2 Score: %.3f\n' % self.r2_score
        string += 'Mean Absolute Percentage Error (MAPE): %.3f\n' % self.mean_absolute_percentage_error
        return string

    def save_to_file(self, log_dir='results', plot_dir='plots'):
        # Get current datetime
        current_dt = datetime.now()
        # Eg. MLP_19-01-20_15_24
        filename = self.model_acronym + '_' + str(self.n_stores) + 'st_' + current_dt.strftime('%d-%m-%y_%H_%M_%S')
        # Create directories if necessary
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        log_path = log_dir + '\\' + filename + '.txt'
        f = open(log_path, 'w+')
        f.write(self.summary())
        f.close()
        plot_path = plot_dir + '\\' + filename + '.png'
        self.plot_figure.savefig(plot_path)
        print('Saved log to ' + log_path + '.')
        print('Saved plot to ' + plot_path + '.')

    def __str__(self):
        return self.summary()


class Timer:

    def start_measure_time(self):
        self.start_time = time.perf_counter()

    def stop_measure_time(self):
        self.stop_time = time.perf_counter()
        self.time_elapsed = self.stop_time - self.start_time
