import base64
import io
import datetime
import time
import pickle

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(suppress_callback_exceptions=True,
                include_assets_files=True)

# Global instance of the engine (cannot be done any other way)
engine = ForecasterEngine()

cache = Cache(app=app.server, config={'CACHE_TYPE': 'filesystem',
                                      'CACHE_DIR': 'app-cache'})

cache.set('key-one', 'isaac_newton')

tab_style = {
    'padding': '6px',
    'color': '#606060',
    'borderBottom': '1px solid #d6d6d6',
}

input_number_style = {
    'width': '65px',
    'height': '32px',
    'marginLeft': '4px',
    'fontSize': 15,
    # 'float': 'left'
}

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Training', children=[html.Div([
            html.H3('Upload data set'),
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=[html.Div([
                        'Drag and Drop or ',
                        html.A('Select File')])],
                    multiple=False,
                    style={
                        'width': '30%',
                        'position': 'relative',
                        # 'height': '50px',
                        'lineHeight': '50px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '0px',
                        'float': 'left'
                    }),
                html.Label([
                    'Separator:',
                    dcc.Input(id='upload-sep',
                              value=';',
                              maxLength=1,
                              style={'width': '30px',
                                     'height': '33px',
                                     'fontWeight': 'bold',
                                     'fontSize': 16,
                                     'marginLeft': '5px'
                                     })],
                    style={'float': 'none', 'position': 'relative'}),
            ],style={'border': '1px solid black', 'overflow': 'auto'}),
            # html.P(),
            html.Div(id='data-table',
                     style={
                         'width': '90%',
                         'paddinLeft': '20px',
                         'paddingTop': '20px'
                     }),

            html.Div(id='model-gen-setup-div', children=
            [
                html.Div(children=['Model Generation'],
                         style={
                             'fontSize': 20,
                             'margin': 'auto',
                             'border': '1px solid #d0d0d0',
                             'width': 'auto',
                             'text-align': 'center'
                         }),
                html.Div(id='select-predicted-feature-div'),
                html.Label('Choose model type:'),
                dcc.RadioItems(id='model-choice',
                               options=[
                                   {'label': 'MLP', 'value': 'MLP'},
                                   {'label': 'CNN', 'value': 'CNN'},
                                   {'label': 'LSTM', 'value': 'LSTM'}],
                               value='MLP',
                               labelStyle={'display': 'inline-block'}),
                html.Div(id='select-features-div'),
                html.Div(id='select-time-feature-div'),
                html.Div(children=[
                    html.Div(id='extract-time-features-div',
                             children=[
                                 html.Label('Do you want to extract time attributes from the date/time column?'),
                                 dcc.Checklist(id='extra-time-features',
                                               options=[
                                                   {'label': 'Year', 'value': 'year'},
                                                   {'label': 'Quarter', 'value': 'quarter'},
                                                   {'label': 'Month', 'value': 'month'},
                                                   {'label': 'Day of month', 'value': 'dayOfMonth'},
                                                   {'label': 'Day of week', 'value': 'dayOfWeek'},
                                                   {'label': 'Hour', 'value': 'hour'},
                                                   {'label': 'Minute', 'value': 'minute'},
                                                   {'label': 'Second', 'value': 'second'}
                                               ], style={'fontSize': 16})
                             ],
                             hidden=True),
                    html.Div(id='model-parameters-div', children=[
                        html.H5('Forecasting model parameters'),
                        html.Label(['Number of input steps:',
                                    dcc.Input(id='input-n_steps_in',
                                              type='number',
                                              style=input_number_style)]),
                        html.P(),
                        html.Label(['Number of output steps:',
                                    dcc.Input(id='input-n_steps_out',
                                              type='number',
                                              style=input_number_style)]),
                        html.P(),
                        html.Label(
                            children=['Number of training epochs:',
                                      dcc.Input(id='input-epochs',
                                                type='number',
                                                style=input_number_style)]),
                        html.Label('Differentiate time series'),
                        html.Label(['Do you want to subtract previous element from every observation in the predicted'
                                    ' feature (may improve prediction)?',
                                    dcc.Checklist(id='chck-differentiate-series',
                                                  options=[{'label': '',
                                                            'value': 'check'}],
                                                  style={'width': '25px'})],
                                   style={'fontSize': 16})
                    ],
                             hidden=True)
                ], style={'border': '2px solid red'}),
                html.Button(id='gen-model-button',
                            children='Generate Model',
                            style={'fontSize': 18,
                                   # 'position': 'absolute',
                                   'float': 'right',
                                   'clear': 'both'
                                   # 'bottom': '50px'
                                   },
                            disabled=True),
                html.Div(id='message-log')
            ],
                     style={
                         'width': '90%',
                         'border': '1px solid grey',
                         'position': 'relative',
                         'marginTop': '10px',
                         'fontSize': 18,
                         'padding': '6px',
                         'overflow': 'auto'
                     }),
            html.Div([
                html.H3('Model results'),
                html.Div(id='model-gen-result-div',
                         children=['Model generation progress'],
                         hidden=False)
            ])
        ],
            style={'fontSize': 16})
        ], style=tab_style, selected_style=tab_style),
        dcc.Tab(label='Prediction', children=[

        ], style=tab_style, selected_style=tab_style)
    ], style={'fontSize': 20,
              'height': '42px'})
])


@app.callback(Output('message-log', 'children'),
              [Input('gen-model-button', 'n_clicks')],
              [State('model-choice', 'value'),
               State('select-predicted-feature', 'value'),
               State('select-features', 'value'),
               State('select-time-feature', 'value'),
               State('extra-time-features', 'value'),
               State('input-n_steps_in', 'value'),
               State('input-n_steps_out', 'value'),
               State('input-epochs', 'value'),
               State('chck-differentiate-series', 'value')])
def generate_model(n_clicks, model_type, predicted_feature, nominal_features, time_features, extra_time_features,
                   n_steps_in, n_steps_out, n_train_epochs, differentiate_series):
    # Get data frame from the cache
    current_df = pickle.loads(cache.get('current_df'))
    if current_df is None:
        return 'No data set loaded for a model!'
    else:
        engine.generate_model(df=current_df, model_type=model_type, datetime_feature=time_features,
                              predicted_feature=predicted_feature,
                              nominal_features=nominal_features, n_steps_in=n_steps_in, n_steps_out=n_steps_out,
                              extra_datetime_features=extra_time_features, n_train_epochs=n_train_epochs,
                              differentiate_series=(differentiate_series != []))
        # model_cache_key = 'forecast_model_{}'.format(model_type)
        # cache.set(model_cache_key, pickle.dumps(model))
        return 'Model generated.'


#
# @cache.memoize(timeout=5000)
# def get_tangent_plot_data():
#     x = np.linspace(-5.0, 5.0, 150)
#     time.sleep(5)
#     tan_x = np.tan(x)
#     scatter = go.Scatter(x=x,
#                          y=tan_x,
#                          mode='lines+markers')
#     serial_scatter = pickle.dumps(scatter)
#     # print('Pickled data (before caching):')
#     # print(serial_scatter)
#     return serial_scatter

#
# @app.callback(Output('plot-1', 'figure'),
#               [Input('generate-button', 'n_clicks')])
# def update_plot(n_clicks):
#     data = pickle.loads(get_tangent_plot_data())
#     # print('Unpickled data:')
#     # print(data)
#     return {'data': [data],
#             'layout': go.Layout(title='Tangent function plot',
#                                 xaxis={'title': 'x'},
#                                 yaxis={'title': 'y'})}


# Updates the data table Div on uploading a CSV file
@app.callback(Output('data-table', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('upload-sep', 'value')])
def upload_data_set(content, filename, filedate, file_separator):
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


if __name__ == '__main__':
    print('Start.')
    cache.clear()
    app.run_server(debug=True)
