import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash()

app.layout = html.Div([
    html.Button(id='but1',
                children='Click here'),
    html.Button(id='but2',
                children='Click here too'),
    html.Div(id='out2')
])


@app.callback(Output('out2', 'children'),
              [Input('but1', 'n_clicks'),
               Input('but2', 'n_clicks')])
def update_but2(n_clicks_b1, n_clicks_b2):
    print('Callback context:')
    ctx = dash.callback_context
    print('inputs:', ctx.inputs)
    print('states:', ctx.states)
    print('triggered:', ctx.triggered)
    print('response:', ctx.response)
    print('but1 clicks:', n_clicks_b1)
    print('but2 clicks:', n_clicks_b2)
    return 'buttons clicked {} times'.format(n_clicks_b1 + n_clicks_b2)


if __name__ == '__main__':
    app.run_server()
