import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from pymongo import MongoClient
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# The limit of reviews that will be used in the views
LIMIT = 500000

# Style from external source
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialise the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
# Get the data from the database
db = client.reviews
collection = db["reviews"]
reviews = collection.find(limit=LIMIT)
data = pd.DataFrame(list(reviews))

reviews = data

# Styling for HTML elements
pageStyle = {
    'margin-left': '15px',
    'margin-right': '15px',
    'margin-top': '20px'
}
pageStyleMap = {
    'margin-left': '15px',
    'margin-right': '15px',
    'margin-top': '20px',
    'height': '600px',
}

nationality_count = data.Reviewer_Nationality.value_counts().to_frame()[:20]
nationality_negative_count = data.where(data['positive'] == False).Reviewer_Nationality.value_counts().to_frame()[:20]
nationality_positive_count = data.where(data['positive']).Reviewer_Nationality.value_counts().to_frame()[:20]

count_bar = px.bar(data, x=nationality_count.index, y=nationality_count.Reviewer_Nationality, barmode="group")
negative_count_bar = px.bar(data, x=nationality_negative_count.index, y=nationality_negative_count.Reviewer_Nationality, barmode="group")
positive_count_bar = px.bar(data, x=nationality_positive_count.index, y=nationality_positive_count.Reviewer_Nationality, barmode="group")

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout for page 1
barView = html.Div(
    children=[
        html.H3(
            'All reviews nationality count',
            style=pageStyle
        ),
        dcc.Graph(
            id='bar',
            figure=count_bar,
            style=pageStyle,
        ),
        html.H3('Postive reviews natinalilty count', style=pageStyle),
        dcc.Graph(
            id='Postive Count-nationality',
            figure=positive_count_bar,
            style=pageStyle
        ),
        html.H3('Negative reviews natinalilty count', style=pageStyle),
        dcc.Graph(
            id='Negative Count-nationality',
            figure=negative_count_bar,
            style=pageStyle
        ),
    ]
)

# Get top 10 countries by nationality count
countries = list(data.Reviewer_Nationality.value_counts()[:10].index)
counts = list(data.Reviewer_Nationality.value_counts()[:10].values)
CountsPieChart = px.pie(values=counts, names=countries, color_discrete_sequence=px.colors.sequential.Agsunset)

negativeCounts = list(data.where(data['positive'] == False).Reviewer_Nationality.value_counts()[:10].values)
NegativeCountsPieChart = px.pie(values=negativeCounts, names=countries,
                                color_discrete_sequence=px.colors.sequential.Agsunset)

positiveCounts = list(data.where(data['positive']).Reviewer_Nationality.value_counts()[:10].values)
positiveCountsPieChart = px.pie(values=positiveCounts, names=countries,
                                color_discrete_sequence=px.colors.sequential.Agsunset)


pieView = html.Div([
    html.H3('All reviews nationality count', style=pageStyle),
    dcc.Graph(
        id='Count-nationality',
        figure=CountsPieChart,
        style=pageStyle
    ),
    html.H3('Postive reviews natinalilty count', style=pageStyle),
    dcc.Graph(
        id='Postive Count-nationality',
        figure=positiveCountsPieChart,
        style=pageStyle
    ),
    html.H3('Negative reviews natinalilty count', style=pageStyle),
    dcc.Graph(
        id='Negative Count-nationality',
        figure=NegativeCountsPieChart,
        style=pageStyle
    ),
])

# Filter and prepare data for the map view
mapdata = data.drop_duplicates('Hotel_Name').set_index("_id")
mapdata['amount'] = data['Hotel_Name'].value_counts()[mapdata.Hotel_Name]
mapdata['count'] = data['Hotel_Name'].value_counts()[mapdata.Hotel_Name]
mapdata = mapdata[(mapdata.lat != 'NA') & (mapdata.lng != 'NA') & (mapdata.lat != 'NULL') & (mapdata.lng != 'NULL')]

# Convert latitude and longitude to float
mapdata['lat'] = mapdata['lat'].astype(float)
mapdata['lng'] = mapdata['lng'].astype(float)

# Set Mapbox access token
px.set_mapbox_access_token('pk.eyJ1IjoiaG91dGtqIiwiYSI6ImNsaHRpYmJxMTBpMGIzcXA1bjYwMWtyd3kifQ.YVKkiRGw707TiM4kFYX3Cw')

# Create scatter mapbox figure
mapFig = px.scatter_mapbox(
    mapdata,
    lat='lat',
    lon='lng',
    hover_name='Hotel_Name',
    color='Average_Score',
    color_continuous_scale=[(0, "red"), (0.5, "yellow"), (1, "green")],
    center=dict(lat=52.36, lon=4.89),
    zoom=5
)

mapFig.update_traces(marker=dict(size=10))

mapView = html.Div([
    html.H3('Location of hotels with color rating', style=pageStyle),
    html.Div(id='map-content'),
    dcc.Graph(
        id='Map',
        figure=mapFig,
        style=pageStyleMap
    )
])


# Define the callback to handle page navigation
@app.callback(Output("content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return barView
    elif pathname == "/bar-view":
        return barView
    elif pathname == "/pie-view":
        return pieView
    elif pathname == "/map-view":
        return mapView
    else:
        return html.Div("404 - Page not found")


nav = html.Div(
    [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Bar chart", href="/bar-view")),
                dbc.NavItem(dbc.NavLink("Pie chart", href="/pie-view")),
                dbc.NavItem(dbc.NavLink("Map", href="/map-view")),
            ],
            brand="Kevin van Hout - BD assignment 2",
            color="primary",
            dark=True,
        ),
    ]
)

# Define the overall layout of the dashboard
app.layout = html.Div(
    children=[
        nav,
        dcc.Location(id="url", refresh=False),
        html.Div(id="content")
    ]
)

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=False)
