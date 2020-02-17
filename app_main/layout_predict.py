import dash
import dash_core_components as dcc
import dash_html_components as html

# from app_main.main import app
# from app_main.main import engine


tab_predict_section = html.Div([
    html.H3('Upload test data set'),
    html.Div([
        dcc.Upload(
            id='upload-data-test',
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
            dcc.Input(id='upload-test-sep',
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
        style={'overflow': 'auto'}
    ),
    html.Div(id='data-table-test',
             style={
                 'width': '84%',
                 'paddinLeft': '20px',
                 'paddingTop': '20px',
             }),
    html.Br(),
    html.Div([
        dcc.Upload(
            id='upload-model',
            children=[
                html.Button('Upload forecasting model',
                            style={'fontSize': 18,
                                   'color': '#107010',
                                   'borderColor': '#107010'}
                            )
            ]
        ),
        html.H5('...or use lastly generated model.'),
        html.H4('Current model:'),
        html.Div(id='current-model-info',
                 children=html.H5('No model currently in use.')),
        html.Hr(),
        html.Button(id='make-prediction',
                    children='Make prediction',
                    style={'fontSize': 18},
                    disabled=False)
    ])
])
