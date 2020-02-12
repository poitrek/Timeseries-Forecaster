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

app = dash.Dash(external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

# Global instance of the engine (cannot be done any other way)
engine = ForecasterEngine()

cache = Cache(app=app.server, config={'CACHE_TYPE': 'filesystem',
                                      'CACHE_DIR': 'app-cache'})

cache.set('key-one', 'isaac_newton')

app.layout = html.Div([
    html.H3('Upload data set'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '50px',
            'lineHeight': '50px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='data-table',
             style={
                 'width': '90%',
                 'paddinLeft': '20px',
                 'paddingTop': '20px'
             }),

    html.Div(id='model-generation-div', children=
    [
        html.Div(children=['Model Generation'],
                 style={
                     'fontSize': 20,
                     'margin': 'auto',
                     'border': '1px solid #d0d0d0',
                     'width': 'auto',
                     'text-align': 'center'
                 }),
        html.Label('Choose model type:'),
        dcc.RadioItems(id='model-choice',
                       options=[
                           {'label': 'MLP', 'value': 'MLP'},
                           {'label': 'CNN', 'value': 'CNN'},
                           {'label': 'LSTM', 'value': 'LSTM'}],
                       value='MLP'),
        html.Div(id='select-features-div'),
        html.Div(id='select-time-feature-div'),
        html.Div(id='extract-time-features-div',
                 children=[
                     html.Label('Do you want to extract time attributes from the date/time column?'),
                     dcc.Checklist(id='extra-time-features',
                                   options=[
                                       {'label': 'Year', 'value': 'year'},
                                       {'label': 'Year season', 'value': 'season'},
                                       {'label': 'Month', 'value': 'month'},
                                       {'label': 'Week', 'value': 'week'},
                                       {'label': 'Day', 'value': 'day'}
                                   ], style={'fontSize': 16})
                 ],
                 hidden=True),
        html.Button(id='gen-model-button',
                    children='Generate Model',
                    style={'fontSize': 18,
                           'position': 'absolute',
                           'right': '30px',
                           # 'bottom': '50px'
                           },
                    disabled=True),
        html.Div(id='message-log')
    ],
             style={
                 'width': '66%',
                 'border': '2px solid grey',
                 'position': 'relative',
                 'marginTop': '10px',
                 'fontSize': 18,
                 'padding': '6px'
             })

])


@app.callback(Output('message-log', 'children'),
              [Input('gen-model-button', 'n_clicks')],
              [State('model-choice', 'value'),
               State('select-features', 'value'),
               State('select-time-feature', 'value'),
               State('extra-time-features', 'value')])
def generate_model(n_clicks, model_type, nominal_features, time_features, extra_time_features):
    # Get data frame from the cache
    current_df = pickle.loads(cache.get('current_df'))
    if current_df is None:
        return 'No data set loaded for a model!'
    else:
        engine.generate_model(df=current_df, model_type=model_type, datetime_features=set(time_features),
                              nominal_features=set(nominal_features), n_steps_in=6, n_steps_out=1)
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


# Parses contents from loaded file
def parse_contents(content, filename, date):
    # Split contents into type and the string
    content_type, content_string = content.split(',')

    # Decode from 64-base string
    content_decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(content_decoded.decode('utf-8')), sep=';', parse_dates=['Date'],
                low_memory=False)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    # Cache data frame
    cache.set('current_df', pickle.dumps(df))

    rows_limit = 7
    return html.Div([
        html.H4('Loaded dataset: \'{}\''.format(filename)),
        # html.H5(datetime.datetime.fromtimestamp(date)),
        # Make a data table from the data frame
        dash_table.DataTable(
            data=df.head(rows_limit).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        )
    ])


# Updates the data table Div on uploading a CSV or XLS file
@app.callback(Output('data-table', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_data_table(content, filename, filedate):
    if content is not None:
        children = [
            parse_contents(content, filename, filedate)
        ]
        return children


# Enables/disables button whether data table is loaded or not
@app.callback([Output('gen-model-button', 'disabled'),
               Output('select-features-div', 'children'),
               Output('select-time-feature-div', 'children'),
               Output('extract-time-features-div', 'hidden')],
              [Input('data-table', 'children')])
def update_model_gen_div(table):
    if table is not None:
        return False, generate_select_features_div(), generate_select_time_feature_div(), False
    else:
        return True, None, None, True


def generate_select_features_div():
    # Get current dataframe from cache
    current_df = pickle.loads(cache.get('current_df'))
    col_options = []
    for col in current_df.columns:
        col_options.append({'label': col, 'value': col})
    return [html.Label('Select nominal columns from the set:'),
            dcc.Dropdown(id='select-features', options=col_options,
                         multi=True)]


def generate_select_time_feature_div():
    # Get current dataframe from cache
    current_df = pickle.loads(cache.get('current_df'))
    col_options = []
    for col in current_df.columns:
        col_options.append({'label': col, 'value': col})
    return [html.Label('Select column that is related in time:'),
            dcc.Dropdown(id='select-time-feature', options=col_options,
                         multi=True)]


if __name__ == '__main__':
    print('Start.')
    cache.clear()
    app.run_server(debug=True)
