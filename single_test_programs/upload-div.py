import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div([
    html.Div(children=[
        dcc.Upload(id='upload', children=['Upload file here'],
                   style={
                       'width': '50%',
                       # 'height': '50px',
                       'lineHeight': '50px',
                       'borderWidth': '1px',
                       'borderStyle': 'dashed',
                       'borderRadius': '5px',
                       'textAlign': 'center',
                       'margin': '0px',
                       'float': 'left'
                   }),
        html.Label('Seperator=...',
                   style={'float': 'none'}),
    ], style={'border': '1px solid black', 'overflow': 'auto'}),
    html.H3('Another label',
            style={'fontWeight': 'bold', 'marginTop': '0px'})
])

if __name__ == '__main__':
    app.run_server()
