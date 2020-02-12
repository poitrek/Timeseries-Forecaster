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
import version_first.forecaster as core


app = dash.Dash()

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
    html.Div(id='data-table'),
    html.Button(id='gen-model-button',
                children='Generate Model',
                style={'fontSize': 22},
                disabled=True),
    dcc.RadioItems(id='model-choice',
                   options=[
                       {'label': 'MLP', 'value': 'MLP'},
                       {'label': 'CNN', 'value': 'CNN'},
                       {'label': 'LSTM', 'value': 'LSTM'}],
                   value='MLP'),
    html.Div(id='message-log'),
])


@app.callback(Output('message-log', 'children'),
              [Input('gen-model-button', 'n_clicks')],
              [State('model-choice', 'value')])
def generate_model(n_clicks, model_type):
    # Get data frame from the cache
    current_df = pickle.loads(cache.get('current_df'))
    if current_df is None:
        return 'No data set loaded for a model!'
    else:
        model = core.generate_model(current_df, model_type, n_steps_in=6, n_steps_out=1)
        model_cache_key = 'forecast_model_{}'.format(model_type)
        cache.set(model_cache_key, pickle.dumps(model))
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
        html.H4(filename),
        html.H5(datetime.datetime.fromtimestamp(date)),
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
@app.callback(Output('gen-model-button', 'disabled'),
              [Input('data-table', 'children')])
def update_gen_model_button(children):
    print(cache.get('key-one'))
    return children is None


if __name__ == '__main__':
    print('Start.')
    cache.clear()
    app.run_server(debug=True)
