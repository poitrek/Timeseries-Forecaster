import dash
import dash_html_components as html
import flask

app = dash.Dash(__name__)

app.layout = html.Div([
    html.A("Download", href="/downloads/data.csv",),
])

@app.server.route("/downloads/<path:path_to_file>")
def serve_file(path_to_file):

    return flask.send_file(
        "data.csv",
        as_attachment=True,
    )

if __name__ == '__main__':
    app.run_server(debug=True)