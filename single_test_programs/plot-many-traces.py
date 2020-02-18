import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go


y_dash = np.arange(700).reshape((100, 7))
y_test = np.arange(97)

steps = 7

data = [go.Scatter(
    y=y_dash[:, i],
    name='step [t+{}]'.format(i),
    mode='lines+markers',
    visible='legendonly'
) for i in range(7)]

data.append(go.Scatter(
    y=y_test,
    name='real',
    mode='lines+markers'
))

app = dash.Dash()

app.layout = html.Div([
    dcc.Graph(
        figure={
            'data': data,
            'layout': go.Layout()
        }
    )
])

if __name__ == '__main__':
    app.run_server()
