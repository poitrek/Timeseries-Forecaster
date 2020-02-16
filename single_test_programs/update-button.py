import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash()

app.layout = html.Div([
    html.Button(id='but1',
                children='Click here'),
    html.Div(id='out1'),
    html.Button(id='but2',
                children='This will update'),
    html.Div(id='out2')
])


@app.callback(Output('out2', 'children'),
              [Input('but2', 'n_clicks')])
def show_but2_update(n_clicks):
    return 'n_click value of but2: {}'.format(n_clicks)


@app.callback([Output('but2', 'n_clicks'),
               Output('out1', 'children')],
              [Input('but1', 'n_clicks')],
              [State('but2', 'n_clicks')])
def update_but2(n_clicks_b1, n_clicks_b2):
    print('but1 clicks:', n_clicks_b1)
    print('but2 clicks:', n_clicks_b2)
    return n_clicks_b2 + 1, 'Number of clicks of but1: {}'.format(n_clicks_b1)


if __name__ == '__main__':
    app.run_server()
