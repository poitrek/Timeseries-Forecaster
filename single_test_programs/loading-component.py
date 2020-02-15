# -*- coding: utf-8 -*-
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import time

from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

app.scripts.config.serve_locally = True

app.layout = html.Div(
    children=[
        dcc.Loading(id="loading-1", children=[html.Div(id="output-1")], type="default"),
        html.Button(id="input-1", children='Trigger'),
        html.Div(
            [
                dcc.Loading(
                    id="loading-2",
                    children=[html.Div([html.Div(id="output-2")])],
                    type="circle",
                ),
                html.Button(id="input-2", children='Trigger'),
            ]
        ),
    ],
)

@app.callback(Output("output-1", "children"), [Input("input-1", "n_clicks")])
def input_triggers_spinner(value):
    time.sleep(2)
    return value


@app.callback(Output("output-2", "children"), [Input("input-2", "n_clicks")])
def input_triggers_nested(value):
    time.sleep(2)
    return value


if __name__ == "__main__":
    app.run_server(debug=False)