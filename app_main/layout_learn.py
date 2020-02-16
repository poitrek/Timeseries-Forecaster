import dash
import dash_core_components as dcc
import dash_html_components as html

input_number_style = {
    'width': '65px',
    'height': '32px',
    'marginLeft': '4px',
    'fontSize': 15,
    # 'float': 'left'
}

tab_learn_section = html.Div([
    html.H3('Upload data set'),
    # html.Button(id='update-loading-button',
    #             children='Stop updating'),
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
                      value=',',
                      maxLength=1,
                      style={'width': '30px',
                             'height': '33px',
                             'fontWeight': 'bold',
                             'fontSize': 16,
                             'marginLeft': '5px'
                             })],
            style={'float': 'none',
                   'position': 'relative',
                   'display': 'inline',
                   'top': '15px',
                   'left': '10px'})],
        # Probably this doesn't matter at all
        # style={'overflow': 'visible'},
    ),
    # html.P(),
    html.Div(id='data-table',
             style={
                 'width': '94%',
                 'paddinLeft': '20px',
                 'paddingTop': '20px',
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
                                      # Temporary
                                      value=6,
                                      min=1,
                                      style=input_number_style)]),
                html.P(),
                html.Label(['Number of output steps:',
                            dcc.Input(id='input-n_steps_out',
                                      type='number',
                                      # Temporary
                                      value=1,
                                      min=1,
                                      style=input_number_style)]),
                html.P(),
                html.Label(
                    children=['Number of training epochs:',
                              dcc.Input(id='input-epochs',
                                        type='number',
                                        min=1,
                                        value=40,
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
        ]),
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
    html.Div(id='bottom-div', children=[
        dcc.Loading(id='loading-1', children=[
            html.Div(id='model-gen-result-div',
                     hidden=False,
                     loading_state={'is_loading': True}),
        ],
            type='circle'),
        html.Div(children='Generating model...',
                 id='loading-msg',
                 hidden=True,
                 style={'align': 'center',
                        'fontSize': 20})
    ]),
    html.Div(id='export-model-div',
             children=[
                 html.A(id='export-model-button',
                        children=html.Button('Export model to file',
                                             style={'fontSize': 16,
                                                    'color': '#1966b3',
                                                    'borderColor': '#1966b3'}
                                             ),
                        download='forecast-model.txt',
                        href='',
                        target='_blank',
                        )
             ],
             hidden=True)
],
    style={'fontSize': 16})
