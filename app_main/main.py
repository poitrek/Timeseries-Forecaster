import base64
import io
import datetime
import sys
import traceback
import time
import pickle
import urllib.request

import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
from flask_caching import Cache
import numpy as np
from app_main.engine import ForecasterEngine
from app_main.layout_learn import tab_learn_section
from app_main.layout_predict import tab_predict_section
from app_main.utils import Timer


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(suppress_callback_exceptions=True,
                include_assets_files=True)

# Global instance of the engine (cannot be done any other way)
engine = ForecasterEngine()
is_model_ready = False

cache = Cache(app=app.server, config={'CACHE_TYPE': 'filesystem',
                                      'CACHE_DIR': 'app-cache'})

tab_style = {
    'padding': '6px',
    'color': '#606060',
    'borderBottom': '1px solid #d6d6d6',
}

# App layout - consists of two tabs (sections) - learning and prediction

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Training',
                children=[
                    tab_learn_section
                ],
                style=tab_style, selected_style=tab_style),
        dcc.Tab(label='Prediction',
                children=[
                    tab_predict_section
                ],
                style=tab_style, selected_style=tab_style)
    ],
        style={'fontSize': 20,
               'height': '44px'})
])


# Define app callbacks

# Helper callback for the loading component
@app.callback(Output('model-gen-result-div', 'loading_state'),
              [Input('upload-data', 'children')])
def update_div_once(children):
    return dash.no_update


@app.callback([Output('model-gen-result-div', 'children'),
               Output('current-model-info', 'children')],
              [Input('gen-model-button', 'n_clicks')],
              [State('model-choice', 'value'),
               State('select-predicted-feature', 'value'),
               State('select-features', 'value'),
               State('select-time-feature', 'value'),
               State('extra-time-features', 'value'),
               State('input-n_steps_in', 'value'),
               State('input-n_steps_out', 'value'),
               State('input-epochs', 'value'),
               State('upload-data', 'filename')])
def generate_and_evaluate_model(n_clicks, model_type, predicted_feature, nominal_features, time_feature,
                                extra_time_features,
                                n_steps_in, n_steps_out, n_train_epochs, data_filename):
    print('Generate_and_evaluate_model()')
    global is_model_ready
    # Get data frame from the cache
    try:
        timer = Timer()
        timer.start_measure_time()
        current_df = cache.get('current_df')
        if current_df is None:
            raise Exception('No data set found to generate model.')
        current_df = pickle.loads(current_df)
        validate_input(predicted_feature, time_feature, n_steps_in, n_steps_out, n_train_epochs)
        dataset = engine.init_model(df=current_df, model_type=model_type, datetime_feature=time_feature,
                                    predicted_feature=predicted_feature,
                                    nominal_features=nominal_features, n_steps_in=n_steps_in, n_steps_out=n_steps_out,
                                    extra_datetime_features=extra_time_features, data_filename=data_filename)
        train_set, test_set = engine.split_train_test(dataset)
        engine.train_model(train_set, epochs=n_train_epochs)
        eval_score = engine.evaluate_model(test_set)
        timer.stop_measure_time()
    except Exception as e:
        # Show error message to proper div
        is_model_ready = False
        results = generate_error_message(e)
        model_info = html.H5('No currently generated model.')
    else:
        # model_cache_key = 'forecast_model_{}'.format(model_type)
        # cache.set(model_cache_key, pickle.dumps(model))
        is_model_ready = True
        results = generate_model_results(eval_score, engine.timeseries_model.model_name, predicted_feature,
                                         len(train_set[0]), timer.time_elapsed)
        model_info = generate_current_model_info(engine.timeseries_model.model_name, data_filename)
    finally:
        return results, model_info


# Checks if the input for model generation is correct. Raises exception if any of the arguments is missing
def validate_input(predicted_feature, time_feature, n_steps_in, n_steps_out, n_train_epochs):
    if predicted_feature is None:
        raise Exception('Please specify the predicted column.')
    if time_feature is None:
        raise Exception('Please specify the time-related column.')
    if n_steps_in is None:
        raise Exception('Please enter the number of input steps.')
    if n_steps_out is None:
        raise Exception('Please enter the number of output steps.')
    if n_train_epochs is None:
        raise Exception('Please enter the number of training epochs.')


def generate_error_message(exception):
    return html.Div([
        html.H5('Error while generating model: ' + str(exception)),
        html.Div(children=[
            html.P(line, style={'lineHeight': '90%'}) for line in traceback.format_exc().splitlines()],
            style={'fontSize': 16})],
        style={'color': 'red'})


def generate_model_results(eval_score, model_name, feature_name, train_set_size, generation_time):
    eval_score.y_dash = eval_score.y_dash.flatten()
    eval_score.y_test = eval_score.y_test.flatten()
    min_value = min(eval_score.y_test.min(), eval_score.y_dash.min())
    max_value = max(eval_score.y_test.max(), eval_score.y_dash.max())
    return html.Div([
        html.H3('Model generation results'),
        html.Div([
            dcc.Graph(figure={
                'data':
                    [go.Scatter(y=eval_score.y_test,
                                mode='markers',
                                name='Real'),
                     go.Scatter(y=eval_score.y_dash,
                                mode='markers',
                                name='Predicted')],
                'layout': go.Layout(title='{} - Real vs Predicted'.format(feature_name),
                                    xaxis={'title': 'Sample no.'},
                                    yaxis={'title': feature_name},
                                    autosize=False,
                                    width=700,
                                    height=500,
                                    paper_bgcolor="#eef",
                                    margin={'l': 60, 't': 60, 'b': 40, 'r': 25},
                                    legend={'x': 0.8, 'y': 1.2})
            },
                style={'margin': '0px'})
        ],
            style={'display': 'inline-block',
                   'float': 'left',
                   'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            dcc.Graph(figure={
                'data': [
                    go.Scatter(x=eval_score.y_test,
                               y=eval_score.y_dash,
                               mode='markers'),
                    go.Scatter(x=[min_value, max_value],
                               y=[min_value, max_value],
                               mode='lines',
                               name='y = x')
                ],
                'layout': go.Layout(title='Q-Q plot',
                                    xaxis={'title': 'Real'},
                                    yaxis={'title': 'Predicted'},
                                    autosize=False,
                                    width=520,
                                    height=500,
                                    paper_bgcolor="#eee",
                                    margin={'l': 50, 't': 60, 'b': 40, 'r': 25},
                                    legend={'x': 0.8, 'y': 1.2})
            },
                style={'margin': '0px'})
        ],
            style={'display': 'inline-block',
                   'margin': '0px',
                   'padding': '0px'}),
        html.Div([
            html.H6('Used forecasting model: {}'.format(model_name)),
            html.H6('Generation time: {:.2f} s'.format(generation_time)),
            html.H6('Training set size: {:d}'.format(train_set_size)),
            html.H6('Tested samples: {}'.format(len(eval_score.y_test))),
            html.H6('Mean absolute error: {:.3f}'.format(eval_score.mae)),
            html.H6('Root mean squared error: {:.3f}'.format(eval_score.rmse)),
            html.H6('R2 Score: {:.3f}'.format(eval_score.r2_score)),
            html.H6('Mean absolute percentage error: {:.3f} %'.format(eval_score.mape))
        ]),
    ])


def generate_current_model_info(model_type, data_filename):
    return html.H5('Current model: {} used for {}'.format(model_type, data_filename))


# Updates the data table Div on uploading a CSV file
@app.callback(Output('data-table', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('upload-sep', 'value')])
def upload_data_set(content, filename, filedate, file_separator):
    print('Upload_data_set()')
    if content is not None:
        children = [
            parse_file_contents(content, filename, filedate, file_separator)
        ]
        return children


# Parses contents from loaded file
def parse_file_contents(content, filename, date, separator):
    # Split contents into type and the string
    content_type, content_string = content.split(',')

    # Decode from 64-base string
    content_decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(content_decoded.decode('utf-8')), sep=separator,
                low_memory=False)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ], style={'fontSize': 16})
    # Cache data frame
    cache.set('current_df', pickle.dumps(df))

    rows_limit = 7
    return html.Div([
        html.H5('Loaded dataset: \'{}\''.format(filename)),
        # html.H5(datetime.datetime.fromtimestamp(date)),
        # Make a data table from the data frame
        dash_table.DataTable(
            data=df.head(rows_limit).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': '100%', 'overflow': 'scroll'}
        )
    ],
        style={'marginTop': '10px'})


# Enables/disables button whether data table is loaded or not
@app.callback([Output('gen-model-button', 'disabled'),
               Output('select-predicted-feature-div', 'children'),
               Output('select-features-div', 'children'),
               Output('select-time-feature-div', 'children'),
               Output('extract-time-features-div', 'hidden'),
               Output('model-parameters-div', 'hidden')],
              [Input('data-table', 'children')])
def update_model_gen_div(table):
    if table is not None:
        return False, generate_select_predicted_feature_div(), generate_select_features_div(), \
               generate_select_time_feature_div(), False, False
    else:
        return True, [], [], [], True, True


# Creates div (label+dropdown) for selecting nominal features
def generate_select_features_div():
    # Get current dataframe from cache
    current_df = pickle.loads(cache.get('current_df'))
    col_options = []
    for col in current_df.columns:
        col_options.append({'label': col, 'value': col})
    return [html.Label('Select nominal columns from the set:'),
            dcc.Dropdown(id='select-features', options=col_options,
                         multi=True)]


# Creates div (label+dropdown) for selecting time feature
def generate_select_time_feature_div():
    # Get current dataframe from cache
    current_df = pickle.loads(cache.get('current_df'))
    col_options = []
    for col in current_df.columns:
        col_options.append({'label': col, 'value': col})
    return [html.Label('Select column that is related in time:'),
            dcc.Dropdown(id='select-time-feature', options=col_options,
                         multi=False)]


def generate_select_predicted_feature_div():
    # Get current dataframe from cache
    current_df = pickle.loads(cache.get('current_df'))
    col_options = []
    for col in current_df.columns:
        col_options.append({'label': col, 'value': col})
    return [html.Label('Which attribute do you want to predict?'),
            dcc.Dropdown(id='select-predicted-feature', options=col_options,
                         multi=False)]


@app.callback([Output('export-model-div', 'hidden'),
               Output('export-model-button', 'href')],
              [Input('model-gen-result-div', 'children')])
def update_export_model_div(children):
    global is_model_ready
    if is_model_ready:
        return False, generate_export_href()
    else:
        return True, ''


def generate_export_href():
    model_serial = pickle.dumps(engine.timeseries_model)
    file_string = 'data:text/plain;charset=utf-8,' + urllib.request.quote(model_serial)
    return file_string


if __name__ == '__main__':
    print('Start.')
    cache.clear()
    app.run_server(debug=True)
