import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd


class Program:

    app = dash.Dash(suppress_callback_exceptions=True)

    def __init__(self):

        self.df = pd.read_csv('https://raw.githubusercontent.com/Pierian-Data/Plotly-Dashboards-with-Dash/master/Data/gapminderDataFiveYear.csv')
        print(self.df)

        # List of year options for the dropdown
        year_options = []
        for year in self.df['year'].unique():
            year_options.append({'label': str(year), 'value': year})

        Program.app.layout = html.Div([
            dcc.Graph(id='graph1'),
            dcc.Dropdown(id='year-picker', options=year_options,
                         value=self.df['year'].min())
        ])

    # @staticmethod
    def run(self):
        Program.app.run_server()

    # Updates the graph on picking a year
    @app.callback(Output(component_id='graph1', component_property='figure'),
                  [Input(component_id='year-picker', component_property='value')])
    def update_figure(self, selected_year):
        # Data only for selected year from the dropdown
        df_of_year = self.df[self.df['year'] == selected_year]

        traces = []
        for continent_name in df_of_year['continent'].unique():
            df_by_continent = df_of_year[df_of_year['continent'] == continent_name]
            traces.append(go.Scatter(
                x=df_by_continent['gdpPercap'],
                y=df_by_continent['lifeExp'],
                text=df_by_continent['country'],
                mode='markers',
                opacity=0.7,
                marker={'size': 15},
                name=continent_name
            ))
        return {'data': traces,
                'layout': go.Layout(title='My Plot',
                                    xaxis={'title': 'GDP per capita', 'type': 'log'},
                                    yaxis={'title': 'Life expectancy'},
                                    hovermode='closest')}


if __name__ == '__main__':

    program = Program()
    program.run()
