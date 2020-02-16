import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import urllib.request

app = dash.Dash()

# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(id='div1', children=[
    html.H3('Hello'),
    html.Div(id='download-button',
             children=[
                 html.A(children=[html.Button('Download data')],
                        id='download-link',
                        download='some_data.txt',
                        href='',
                        target='_blank', )
             ],
             )
])


@app.callback(Output('download-link', 'href'),
              [Input('div1', 'children')])
def update_download_link(children):
    string = 'this is whats written in the file'
    file = 'data:text/txt;charset=utf-8,' + urllib.request.quote(string)
    return file


if __name__ == '__main__':
    app.run_server()
