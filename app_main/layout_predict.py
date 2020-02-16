import dash
import dash_core_components as dcc
import dash_html_components as html


tab_predict_section = html.Div([
    html.H3('Prediction section'),
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
        )
    ])
])

