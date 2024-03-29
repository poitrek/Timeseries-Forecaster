import base64
import io
import traceback
import pickle
import urllib.request

import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
import numpy as np
from app_main.engine import ForecasterEngine
from app_main.layout_train import tab_train_section
from app_main.layout_predict import tab_predict_section
from app_main.utils import Timer


app = dash.Dash(suppress_callback_exceptions=True,
                include_assets_files=True)

# Global instance of the engine (cannot be done any other way)
engine = ForecasterEngine()

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
                    tab_train_section
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

# Performs the generation and evaluation of forecasting model
@app.callback(Output('model-gen-result-div', 'children'),
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
    try:
        timer = Timer()
        timer.start_measure_time()
        validate_input(predicted_feature, time_feature, n_steps_in, n_steps_out, n_train_epochs)
        dataset = engine.init_model(model_type=model_type, datetime_feature=time_feature,
                                    predicted_feature=predicted_feature,
                                    nominal_features=nominal_features, n_steps_in=n_steps_in, n_steps_out=n_steps_out,
                                    extra_datetime_features=extra_time_features, data_filename=data_filename)
        train_set, test_set = engine.split_train_test(dataset)
        engine.train_model(train_set, epochs=n_train_epochs)
        eval_score = engine.evaluate_model(test_set)
        timer.stop_measure_time()
    except Exception as e:
        # Show error message to proper div
        engine.is_model_ready = False
        results = generate_error_message(e, 'Error while generating model: ')
        # model_info = html.H5('No model currently in use.')
    else:
        engine.is_model_ready = True
        results = generate_model_results(eval_score, engine.timeseries_model.model_name, predicted_feature,
                                         len(train_set[0]), timer.time_elapsed)
        # model_info = engine.get_model_info()
    finally:
        return results


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


def generate_error_message(exception, title):
    return html.Div([
        html.H5(title + str(exception)),
        html.Div(children=[
            html.P(line, style={'lineHeight': '90%'}) for line in traceback.format_exc().splitlines()],
            style={'fontSize': 16})],
        style={'color': 'red'})


def generate_model_results(eval_score, model_name, feature_name, train_set_size, generation_time):
    y_dash = eval_score.y_dash
    y_test = eval_score.y_test
    test_shape = eval_score.y_test.shape
    min_value = min(eval_score.y_test.min(), eval_score.y_dash.min())
    max_value = max(eval_score.y_test.max(), eval_score.y_dash.max())
    # Number of test samples to show on the first plot
    samples_to_show = min(300, len(eval_score.y_test))
    plot_data = [go.Scatter(
        x=np.arange(samples_to_show),
        y=y_test[:samples_to_show, 0],
        mode='lines+markers',
        name='Real')]
    for i in range(y_dash.shape[1]):
        plot_data.append(go.Scatter(
            x=np.arange(samples_to_show) + i,
            y=y_dash[:samples_to_show, i],
            mode='lines+markers',
            opacity=0.8,
            name='Pred. step [t+{}]'.format(i),
            visible=True if i == 0 else 'legendonly'
        ))

    return html.Div([
        html.H3('Model generation results'),
        html.Div([
            dcc.Graph(figure={
                'data': plot_data,
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
                    go.Scatter(x=y_test.flatten(),
                               y=y_dash.flatten(),
                               mode='markers',
                               name='',
                               showlegend=False),
                    go.Scatter(x=[min_value, max_value],
                               y=[min_value, max_value],
                               mode='lines',
                               name='y = x',
                               showlegend=True)
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
            html.H6('Tested samples: {:d} x {:d} step{}'.format(test_shape[0], test_shape[1],
                                                                's' if test_shape[1] > 1 else '')),
            html.H6('Mean absolute error: {:.3f}'.format(eval_score.mae)),
            html.H6('Root mean squared error: {:.3f}'.format(eval_score.rmse)),
            html.H6('R2 Score: {:.3f}'.format(eval_score.r2_score)),
            html.H6('Mean absolute percentage error: {:.3f} %'.format(eval_score.mape))
        ]),
    ])


# Updates the data table Div on uploading a CSV file
@app.callback(Output('data-table', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-sep', 'value')])
def upload_data_set(content, filename, file_separator):
    print('Upload_data_set()')
    if content is not None:
        data_table_div = generate_data_table(content, filename, file_separator)
        return data_table_div


# Parses contents from loaded file
def generate_data_table(content, filename, separator):
    # Split contents into type and the string
    content_type, content_string = content.split(',')

    # Decode from 64-base string
    content_decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(content_decoded.decode('utf-8')), sep=separator,
                low_memory=False)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ], style={'fontSize': 16})
    # Save df to engine
    engine.train_df = df
    rows_limit = 7
    data_table_div = html.Div([
        html.H5('Loaded dataset: \'{}\''.format(filename),
                style={'float': 'left'}),
        html.H6('{} rows x {} columns'.format(len(df), len(df.columns)),
                style={'float': 'right',
                       'marginRight': '10px'}),
        # Make a data table from the data frame
        dash_table.DataTable(
            data=df.head(rows_limit).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': '100%', 'overflow': 'scroll'}
        ),
    ],
        style={'marginTop': '10px'})
    return data_table_div


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
    current_df = engine.train_df
    col_options = []
    for col in current_df.columns:
        col_options.append({'label': col, 'value': col})
    return [html.Label('Select nominal columns from the set:'),
            dcc.Dropdown(id='select-features', options=col_options,
                         multi=True)]


# Creates div (label+dropdown) for selecting time feature
def generate_select_time_feature_div():
    current_df = engine.train_df
    col_options = []
    for col in current_df.columns:
        col_options.append({'label': col, 'value': col})
    return [html.Label('Select column that is related in time:'),
            dcc.Dropdown(id='select-time-feature', options=col_options,
                         multi=False)]


def generate_select_predicted_feature_div():
    current_df = engine.train_df
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
    if engine.is_model_ready:
        return False, generate_export_href()
    else:
        return True, ''


def generate_export_href():
    model_serial = pickle.dumps(engine.timeseries_model)
    file_string = 'data:text/plain;charset=utf-8,' + urllib.request.quote(model_serial)
    return file_string


# ======================= Callbacks for 'Prediction' module =======================


@app.callback(Output('current-model-info', 'children'),
              [Input('model-gen-result-div', 'children'),
               Input('upload-model', 'contents')])
def update_current_model_info(gen_res_div, upload_content):
    triggered = dash.callback_context.triggered[0]
    print('triggered.prop_id:', triggered['prop_id'])
    if triggered['prop_id'] == 'model-gen-result-div.children':
        if engine.is_model_ready:
            model = engine.timeseries_model
            return html.H4('(Internal) {}'.format(engine.get_model_info()))
        else:
            return dash.no_update
    elif triggered['prop_id'] == 'upload-model.contents':
        return parse_uploaded_model(upload_content)
    else:
        return dash.no_update


def parse_uploaded_model(content):
    try:
        # Split contents into type and the string
        content_type, content_string = content.split(',')
        # Decode from 64-base string
        content_decoded = base64.b64decode(content_string)
        engine.load_model_unpickle(content_decoded)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing the file: {}'.format(e)
        ], style={'fontSize': 16})
    else:
        return html.H4('(Uploaded) {}'.format(engine.get_model_info()))


# Updates the test data table Div on uploading a CSV file
@app.callback(Output('data-table-test', 'children'),
              [Input('upload-data-test', 'contents')],
              [State('upload-data-test', 'filename'),
               State('upload-test-sep', 'value')])
def upload_test_data_set(content, filename, file_separator):
    print('Upload_data_set()')
    if content is not None:
        data_table_div, df = generate_test_data_table(content, filename, file_separator)
        engine.test_df = df
        return data_table_div


# Parses contents from loaded file
def generate_test_data_table(content, filename, separator):
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

    rows_limit = 5
    data_table_div = html.Div([
        html.H5('Testing data set: \'{}\''.format(filename),
                style={'float': 'left'}),
        html.H6('{} rows x {} columns'.format(len(df), len(df.columns)),
                style={'float': 'right',
                       'marginRight': '10px'}),
        # Make a data table from the data frame
        dash_table.DataTable(
            data=df.head(rows_limit).to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': '100%', 'overflow': 'scroll'}
        ),
    ],
        style={'marginTop': '10px'})
    return data_table_div, df


@app.callback([Output('prediction-result-div', 'children'),
               Output('download-results', 'hidden'),
               Output('download-results', 'href')],
              [Input('make-prediction', 'n_clicks')])
def make_prediction(n_clicks):
    if n_clicks == 0:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        y_predicted = engine.make_prediction()

    except Exception as e:
        return generate_error_message(e, 'Error while making prediction: '), True, ''
    else:
        return generate_prediction_result(y_predicted, engine.timeseries_model), False,\
                    generate_prediction_csv_href(y_predicted)


def generate_prediction_result(y_dash, model):
    time_attribute = model.get_datetime_array()
    # Limit samples to show
    samples_limit = 600
    n_samples = y_dash.shape[0]
    if n_samples > samples_limit:
        y_dash = y_dash[:samples_limit, :]
        time_attribute = time_attribute[:samples_limit]
    n_steps_out = y_dash.shape[1]
    # Extend time attribute by several new timestamps
    time_delta = time_attribute[1] - time_attribute[0]
    extra_time = [time_attribute.iloc[-1] + time_delta]
    for i in range(n_steps_out - 1):
        extra_time.append(extra_time[-1] + time_delta)
    time_attribute = time_attribute.append(pd.Series(extra_time), ignore_index=True)
    print('time_attribute:')
    print(time_attribute)
    return html.Div([
        html.H3('Prediction results'),
        html.H6('Used model: {}'.format(model.model_name)),
        html.H6('Predicted column name: {}'.format(model.predicted_feature)),
        html.H6('Number of samples: {}'.format(n_samples)),
        html.H6('Number of steps predicting: {}'.format(n_steps_out)),
        dcc.Graph(figure={
            'data':
                # Define trace for every output step
                [go.Scatter(
                    x=time_attribute[i: i-n_steps_out],
                    y=y_dash[:, i],
                    mode='lines+markers',
                    name='step {}'.format(i),
                    visible=True if i==0 else 'legendonly',
                    showlegend=True)
                    for i in range(n_steps_out)],
            'layout': go.Layout(title='{} - Prediction'.format(model.predicted_feature),
                                xaxis={'title': 'Date/time'},
                                yaxis={'title': model.predicted_feature},
                                autosize=True,
                                paper_bgcolor='#fee')
        })
    ])


# Generates the content of downloading csv link
def generate_prediction_csv_href(y_predicted):
    predicted_feature_name = engine.timeseries_model.predicted_feature
    # Number of steps predicted
    n_steps_out = y_predicted.shape[1]
    # List of columns names
    cols = ['{}_step_{}'.format(predicted_feature_name, i+1) for i in range(n_steps_out)]
    # Make a df
    prediction_df = pd.DataFrame(y_predicted, columns=cols)
    print('Model prediction df')
    print(prediction_df)
    content_bytes = bytearray(prediction_df.to_csv(), 'utf-8')
    return 'data:text/csv;charset=utf-8,' + urllib.request.quote(content_bytes)


if __name__ == '__main__':
    print('Start.')
    app.run_server(debug=True)
