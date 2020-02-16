import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import flask
import StringIO

app = dash.Dash()

app.layout = html.Div([

])


@app.callback(Output('my-link', 'href'), [Input('my-dropdown', 'value')])
def update_link(value):
    return '/dash/urlToDownload?value={}'.format(value)

@app.server.route('/dash/urlToDownload')
def download_csv():
    value = flask.request.args.get('value')
    # create a dynamic csv or file here using `StringIO`
    # (instead of writing to the file system)
    strIO = StringIO.StringIO()
    strIO.write('You have selected {}'.format(value))
    strIO.seek(0)
    return send_file(strIO,
                     mimetype='text/csv',
                     attachment_filename='downloadFile.csv',
                     as_attachment=True)


if __name__ == '__main__':
    app.run_server()
