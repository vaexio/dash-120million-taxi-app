import json
import logging
import os

import dash
from dash.dependencies import Input, Output, State

import dash_core_components as dcc

import dash_html_components as html

import dash_table

from flask_caching import Cache

import numpy as np

import plotly.express as px
import plotly.graph_objs as go

import vaex


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('vaex-dash')

external_stylesheets = []
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # used by gunicorn in production mode
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
# set negative to disable (useful for testing/benchmarking)
CACHE_TIMEOUT = int(os.environ.get('DASH_CACHE_TIMEOUT', '60'))

# Get auxiliary information about zones, boroughs, and their relations
geo_filename = 'aux_data/taxi_zones-tiny.json'
with open(geo_filename) as f:
    geo_json = json.load(f)
zone_filename = 'aux_data/zone.json'
with open(zone_filename) as f:
    zmapper = json.load(f)
with open('aux_data/borough.json', 'rb') as f:
    bmapper = json.load(f)
with open('aux_data/zone_to_borough.json', 'rb') as f:
    zbmapper = json.load(f)

zone_index_to_name = {int(index): name for index, name in zmapper.items()}
zone_name_to_index = {name: int(index) for index, name in zmapper.items()}
borough_index_to_name = {int(index): name for index, name in bmapper.items()}
borough_name_to_index = {name: int(index) for index, name in bmapper.items()}
zone_index_to_borough_index = {int(index): borough_name_to_index[zbmapper[name]] for index, name in zmapper.items()}


# Open the main data
taxi_path = 's3://vaex/taxi/yellow_taxi_2012_zones.hdf5?anon=true'
# override the path, e.g. $ export TAXI_PATH=/data/taxi/yellow_taxi_2012_zones.hdf5
taxi_path = os.environ.get('TAXI_PATH', taxi_path)
df_original = vaex.open(taxi_path)

# Make sure the data is cached locally
used_columns = ['pickup_longitude',
                'pickup_latitude',
                'dropoff_longitude',
                'dropoff_latitude',
                'total_amount',
                'trip_duration_min',
                'trip_speed_mph',
                'pickup_hour',
                'pickup_day',
                'dropoff_borough',
                'dropoff_zone',
                'pickup_borough',
                'pickup_zone']
for col in used_columns:
    print(f'Making sure column "{col}" is cached...')
    df_original.nop(col, progress=True)

# Treat these columns as categorical - improves groupby performance.
df_original.categorize(df_original.pickup_day, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], inplace=True)
df_original.categorize(df_original.pickup_zone, inplace=True)
df_original.categorize(df_original.dropoff_zone, inplace=True)

# The starting point
heatmap_limits_initial = [[-74.03647198, -73.77135504], [40.6371054, 40.80003402]]
limits_amount = [0, 50]
limits_duration = [0, 50]
bins = 25
n_largest = 5
resolution_initial = 75

zone_initial = 89  # JFK
trip_start_initial = -73.79413852703125, 40.65619859765626  # JFK
trip_end_initial = -73.99194061898439, 40.75039170609375  # Manhatten


# This has to do with layout/styling
fig_layout_defaults = dict(
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
)

# Markdown / descriptions (should be moved where it fits)
about_md = '''
### Dash and Vaex: Big data exposed

An example of an interactive dashboard created with [Vaex](https://github.com/vaexio/vaex) and
[Dash](https://plotly.com/dash/). Vaex is a high performance DataFrame library enabling efficient, out-of-core computing
for large datasets comprising millions or billions of samples. Thie example uses Vaex as an engine for computing statistics
and aggregations which are passed to Plotly to create beautiful diagrams. The dataset shown comprises nearly 120
million trips conducted by the Yellow Taxies throughout New York City in 2012, and is available via the [Taxi &
Limousine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

Read [this article](link_placeholder) to learn how to create such dashboards with Vaex and Dash.
'''

overview_md = f'''
### Dataset size {len(df_original):,} rows
The following filters are applied to all of the data
'''


data_summary_filtered_md_template = 'Selected {:,} trips'
data_summary_filtered_md = data_summary_filtered_md_template.format(len(df_original))


def create_figure_empty():
    layout = go.Layout(plot_bgcolor='white', width=10, height=10,
                       xaxis=go.layout.XAxis(visible=False),
                       yaxis=go.layout.YAxis(visible=False))
    return go.Figure(layout=layout)


# Taken from https://dash.plotly.com/datatable/conditional-formatting
def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #96dbfa 0%,
                    #96dbfa {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles

#######################################
# Figure/plotly function
#######################################


def create_figure_histogram(x, counts, title=None, xlabel=None, ylabel=None):

    # settings
    color = 'royalblue'

    # list of traces
    traces = []

    # Create the figure
    line = go.scatter.Line(color=color, width=2)
    hist = go.Scatter(x=x, y=counts, mode='lines', line_shape='hv', line=line, name=title, fill='tozerox')
    traces.append(hist)

    # Layout
    title = go.layout.Title(text=title, x=0.5, y=1, font={'color': 'black'})
    margin = go.layout.Margin(l=0, r=0, b=0, t=30)
    legend = go.layout.Legend(orientation='h',
                              bgcolor='rgba(0,0,0,0)',
                              x=0.5,
                              y=1,
                              itemclick=False,
                              itemdoubleclick=False)
    layout = go.Layout(height=230,
                       margin=margin,
                       legend=legend,
                       title=title,
                       xaxis=go.layout.XAxis(title=xlabel),
                       yaxis=go.layout.YAxis(title=ylabel),
                       **fig_layout_defaults)

    # Now calculate the most likely value (peak of the histogram)
    peak = np.round(x[np.argmax(counts)], 2)

    return go.Figure(data=traces, layout=layout), peak


def create_figure_heatmap(data_array, heatmap_limits, trip_start, trip_end):
    logger.info("Figure: update heatmap heatmap_limits=%r", heatmap_limits)

    # Set up the layout of the figure
    legend = go.layout.Legend(orientation='h',
                              x=0.0,
                              y=-0.05,
                              font={'color': 'azure'},
                              bgcolor='royalblue',
                              itemclick=False,
                              itemdoubleclick=False)
    margin = go.layout.Margin(l=0, r=0, b=0, t=30)
    # if we don't explicitly set the width, we get a lot of autoresize events
    layout = go.Layout(height=600,
                       title=None,
                       margin=margin,
                       legend=legend,
                       xaxis=go.layout.XAxis(title='Longitude', range=heatmap_limits[0]),
                       yaxis=go.layout.YAxis(title='Latitude', range=heatmap_limits[1]),
                       **fig_layout_defaults)

    fig = go.Figure(layout=layout)

    # add the heatmap
    # Use plotly express in combination with xarray - easy plotting!
    fig = px.imshow(np.log1p(data_array.T), origin='lower')
    fig.layout = layout

    counts = data_array.data
    (xmin, xmax), (ymin, ymax) = heatmap_limits
    dx = (xmax - xmin) / counts.shape[0]
    dy = (ymax - ymin) / counts.shape[1]

    fig.add_trace(go.Heatmap(z=np.log10(counts.T+1),
                             colorscale='plasma',
                             zmin=None, zmax=None,
                             x0=xmin, dx=(dx),
                             y0=ymin, dy=(dy),
                             showscale=False,
                             hoverinfo=['x', 'y', 'z']))

    # add markers for the points clicked
    def add_point(x, y, **kwargs):
        fig.add_trace(go.Scatter(x=[x], y=[y], marker_color='azure', marker_size=8, mode='markers', showlegend=True, **kwargs))

    if trip_start:
        add_point(trip_start[0], trip_start[1], name='Trip start', marker_symbol='circle')

    if trip_end:
        add_point(trip_end[0], trip_end[1], name='Trip end', marker_symbol='x')

    return fig


def create_figure_geomap(pickup_counts, zone, zoom=10, center={"lat": 40.7, "lon": -73.99}):
    geomap_data = {
        'count': pickup_counts,
        'log_count': np.log10(pickup_counts),
        'zone_name': list(zmapper.values())
    }

    fig = px.choropleth_mapbox(geomap_data,
                               geojson=geo_json,
                               color="log_count",
                               locations="zone_name",
                               featureidkey="properties.zone",
                               mapbox_style="carto-positron",
                               hover_data=['count'],
                               zoom=zoom,
                               center=center,
                               opacity=0.5,
                               )
    # Custom tool-tip
    hovertemplate = '<br>Zone: %{location}' \
                    '<br>Number of trips: %{customdata:.3s}'
    fig.data[0]['hovertemplate'] = hovertemplate

    # draw the selected zone
    geo_json_selected = geo_json.copy()
    geo_json_selected['features'] = [
        feature for feature in geo_json_selected['features'] if feature['properties']['zone'] == zone_index_to_name[zone]
    ]

    geomap_data_selected = {
        'zone_name': [
            geo_json_selected['features'][0]['properties']['zone'],
        ],
        'default_value': ['start'],
        'log_count': [geomap_data['log_count'][zone]],
        'count': [geomap_data['count'][zone]],
    }

    fig_temp = px.choropleth_mapbox(geomap_data_selected,
                                    geojson=geo_json_selected,
                                    color='default_value',
                                    locations="zone_name",
                                    featureidkey="properties.zone",
                                    mapbox_style="carto-positron",
                                    hover_data=['count'],
                                    zoom=9,
                                    center={"lat": 40.7, "lon": -73.99},
                                    opacity=1.,
                                    )
    fig.add_trace(fig_temp.data[0])
    # Custom tool-tip
    hovertemplate = '<br>Zone: %{location}' \
                    '<br>Number of trips: %{customdata:.3s}' \
                    '<extra></extra>'
    fig.data[1]['hovertemplate'] = hovertemplate

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, coloraxis_showscale=False, showlegend=False)
    return fig


def create_figure_sankey(df_outflow_top, df_outflow_rest, df_outflow_borough, pickup_zone):
    pickup_borough = zone_index_to_borough_index[pickup_zone]

    df_outflow_top['dropoff_borough'] = df_outflow_top.dropoff_zone.map(zone_index_to_borough_index)

    label_offset_zones = len(list(bmapper.values()))
    labels = list(bmapper.values()) + list(zmapper.values()) + list([f'Other: {k}' for k in bmapper.values()])

    # overwrite the borough label with one that includes the zone name
    start_name = f'{borough_index_to_name[pickup_borough]} - {zone_index_to_name[pickup_zone]}'
    labels[pickup_borough] = start_name

    source = df_outflow_borough.pickup_borough.astype('int').tolist() + df_outflow_top.dropoff_borough.astype('int').tolist() + df_outflow_rest.dropoff_borough.astype('int').tolist()

    zone_indices = (df_outflow_top.dropoff_zone.astype('int') + label_offset_zones).tolist()
    zone_indices_other = (df_outflow_rest.dropoff_zone.astype('int') + label_offset_zones).tolist()
    target = df_outflow_borough.dropoff_borough.astype('int').tolist() + zone_indices + zone_indices_other
    link_labels = [labels[t] for t in target]

    counts = np.array(df_outflow_borough['count_trips'].tolist() +
                      df_outflow_top['count_trips'].tolist() +
                      df_outflow_rest['count_trips'].tolist())

    line_sankey = go.sankey.node.Line(color='black', width=0.5)
    node_sankey = go.sankey.Node(pad=15, thickness=20, line=line_sankey, label=labels, color='blue')
    link_hovertemplate = 'Origin: %{source.label}<br>Destination %{target.label}'
    link_sankey = go.sankey.Link(source=source, target=target, value=counts, label=link_labels, hovertemplate=link_hovertemplate)
    fig_sankey = go.Figure(data=[go.Sankey(node=node_sankey, link=link_sankey)])

    title_text = f"Outflow of taxis from {zone_index_to_name[pickup_zone]} to other Boroughs, and top {n_largest} zones"
    fig_sankey.update_layout(title_text=title_text, font_size=10, **fig_layout_defaults)
    return fig_sankey


def create_figure_sunburst(df_outflow_top, df_outflow_rest, df_outflow_borough, pickup_zone):
    for df in [df_outflow_borough, df_outflow_top]:
        df['dropoff_borough_name'] = df.dropoff_borough.astype('int').map(borough_index_to_name)
    for df in [df_outflow_top]:
        df['dropoff_zone_name'] = df.dropoff_zone.astype('int').map(zone_index_to_name)
        df['dropoff_borough_name'] = df.dropoff_borough.astype('int').map(borough_index_to_name)

    start_name = zone_index_to_name[pickup_zone]
    labels = [start_name] + df_outflow_borough.dropoff_borough_name.tolist() + df_outflow_top.dropoff_zone_name.tolist()
    parents = [""] + [start_name] * len(df_outflow_borough) + df_outflow_top.dropoff_borough_name.tolist()
    counts = np.array([len(df)] + df_outflow_borough.count_trips.tolist() + df_outflow_top.count_trips.tolist())

    fig_sunburst = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=np.log10(counts+10),
        customdata=np.stack([counts], axis=-1),
        hovertemplate='Numer of trips to <b>%{label}</b>:<br> %{customdata[0]:,}<extra></extra>',
        # branchvalues="total"
        ))
    fig_sunburst.layout = go.Layout(**fig_layout_defaults)
    return fig_sunburst


def create_table_data(df_outflow_top):
    df = df_outflow_top.copy()
    last_borough = None
    df['zone'] = df.dropoff_zone.astype('int').map(zone_index_to_name)
    df['borough'] = df.dropoff_zone.map(zone_index_to_borough_index).map(borough_index_to_name)
    records = []
    for i, record in df[['borough', 'zone', 'count_trips']].iterrows():
        if record['borough'] != last_borough:
            last_borough = record['borough']
        else:
            record['borough'] = ''  # don't repeat borough
        records.append(record)

    return records, data_bars(df, 'count_trips')


# ######################################
# Compute/dataframe functions
# ######################################

def create_selection(days, hours):
    df = df_original.copy()
    selection = None
    if hours:
        hour_min, hour_max = hours
        if hour_min > 0:
            df.select((hour_min <= df.pickup_hour), mode='and')
            selection = True
        if hour_max < 23:
            df.select((df.pickup_hour <= hour_max), mode='and')
            selection = True
    if (len(days) > 0) & (len(days) < 7):
        df.select(df.pickup_day.isin(days), mode='and')
        selection = True
    return df, selection


@cache.memoize(timeout=CACHE_TIMEOUT)
def compute_heatmap_data(days, hours, heatmap_limits):
    logger.info("Compute: heatmap data: days=%r hours=%r limits=%r", days, hours, heatmap_limits)
    df, selection = create_selection(days, hours)
    heatmap_data_array = df.count(binby=[df.pickup_longitude, df.pickup_latitude],
                                  selection=selection,
                                  limits=heatmap_limits,
                                  shape=256,
                                  array_type="xarray")
    return heatmap_data_array


@cache.memoize(timeout=CACHE_TIMEOUT)
def compute_geomap_data(days, hours):
    logger.info("Compute: geomap data: days=%r hours=%r", days, hours)
    df, selection = create_selection(days, hours)
    return df.count(binby=df.pickup_zone, selection=selection)


@cache.memoize(timeout=CACHE_TIMEOUT)
def compute_trip_details(days, hours, trip_start, trip_end):
    logger.info("Compute: trip details: days=%r hours=%r trip_start=%r trip_end=%r", days, hours, trip_start, trip_end)
    df, selection = create_selection(days, hours)
    # Filter the dataframe
    r = 0.0145 / 20 * 3  # One mile is ~0.0145 deg and 20 blocks per mile.
    pickup_long, pickup_lat = trip_start
    dropoff_long, dropoff_lat = trip_end

    selection_pickup = (df.pickup_longitude - pickup_long)**2 + (df.pickup_latitude - pickup_lat)**2 <= r**2
    selection_dropoff = (df.dropoff_longitude - dropoff_long)**2 + (df.dropoff_latitude - dropoff_lat)**2 <= r**2
    df.select(selection_pickup & selection_dropoff, mode='and')
    selection = True  # after this the selection is always True

    return {
        'counts': df.count(selection=selection),
        'counts_total': df.count(binby=[df.total_amount], limits=[limits_amount], shape=bins, selection=selection),
        'counts_duration': df.count(binby=[df.trip_duration_min], limits=[limits_duration], shape=bins, selection=selection),
    }


@cache.memoize(timeout=CACHE_TIMEOUT)
def compute_flow_data(days, hours, zone):
    logger.info("Compute: flow data: days=%r hours=%r zone=%r", days, hours, zone)
    df, selection = create_selection(days, hours)
    df.select(df.pickup_zone == zone, mode='and')
    selection = True
    df_flow_zone = df.groupby([df.pickup_zone, df.dropoff_zone], agg={'count_trips': vaex.agg.count(selection=selection)})
    # sort descending so we can take the top N
    df_flow_zone = df_flow_zone.sort('count_trips', ascending=False)

    df_flow_zone['pickup_borough'] = df_flow_zone.pickup_zone.map(zone_index_to_borough_index)
    df_flow_zone['dropoff_borough'] = df_flow_zone.dropoff_zone.map(zone_index_to_borough_index)

    pickup_zone = zone
    pickup_borough = zone_index_to_borough_index[pickup_zone]

    # Now to include the total count of all trips for zones that are not the top N
    # only trips leaving from this zone and to a different borough
    df_outflow_zone = df_flow_zone[(df_flow_zone.pickup_zone == pickup_zone)]
    df_outflow_zone = df_outflow_zone[df_outflow_zone.dropoff_borough != pickup_borough]

    df_outflows_top = []
    df_outflows_rest = []

    for dropoff_borough in range(6):
        if dropoff_borough == pickup_borough:
            continue
        # outflow from this zone, to a particular borough
        df_outflow_zone_borough = df_outflow_zone[df_outflow_zone.dropoff_borough == dropoff_borough]
        if len(df_outflow_zone_borough):
            n_max = min(len(df_outflow_zone_borough), n_largest)
            # top N zones of outflow from this zone, to a particular borough
            df_outflows_top.append(df_outflow_zone_borough[:n_max])

            if len(df_outflow_zone_borough) > n_largest:
                count_other = df_outflow_zone_borough[n_largest:]['count_trips'].sum()

                # rest of the outflow from this zone, to a particular borough
                df_outflows_rest.append(vaex.from_scalars(pickup_borough=pickup_borough,
                                        dropoff_borough=dropoff_borough,
                                        dropoff_zone=len(zone_index_to_name) + dropoff_borough,
                                        count_trips=count_other))

    df_outflow_top = vaex.concat(df_outflows_top)
    df_outflow_borough = df_outflow_zone.groupby(['pickup_borough', 'dropoff_borough'],
                                                 agg={'count_trips': vaex.agg.sum('count_trips')}
                                                 )
    if df_outflows_rest:
        df_outflow_rest = vaex.concat(df_outflows_rest)
    else:
        # create an empy dataframe with the same schema to make the rest of the code simpler
        df_outflow_rest = vaex.from_scalars(pickup_borough=-1,
                                            dropoff_borough=-1,
                                            dropoff_zone=-1,
                                            count_trips=-1)[:0]

    # return as dict and lists so it can be serialized by the memoize decorator
    flow_data = dict(
        outflow_top=df_outflow_top.to_dict(array_type='list'),
        outflow_rest=df_outflow_rest.to_dict(array_type='list'),
        outflow_borough=df_outflow_borough.to_dict(array_type='list')
    )
    return flow_data


# ######################################
# Dash specific part
# ######################################

heatmap_data_initial = compute_heatmap_data([], [0, 23], heatmap_limits_initial)
geomap_data_initial = compute_geomap_data([], [0, 23])
trip_detail_data_initial = compute_trip_details([], [0, 23], trip_start_initial, trip_end_initial)
flow_data_initial = compute_flow_data([], [0, 23], zone_initial)

df_outflow_top_initial = vaex.from_dict(flow_data_initial['outflow_top'])
df_outflow_rest_initial = vaex.from_dict(flow_data_initial['outflow_rest'])
df_outflow_borough_initial = vaex.from_dict(flow_data_initial['outflow_borough'])

figure_sankey_initial = create_figure_sankey(df_outflow_top_initial, df_outflow_rest_initial, df_outflow_borough_initial, zone_initial)
figure_sunburst_initial = create_figure_sunburst(df_outflow_top_initial, df_outflow_rest_initial, df_outflow_borough_initial, zone_initial)

table_records_intitial, table_style_initial = create_table_data(df_outflow_top_initial)

zone_summary_template_md = '''
**{}**: **{:,}** taxi trips leaving this zone.

_Click on the map to change the zone._
'''

zone_pickup_count = geomap_data_initial[zone_initial]
zone_summary_md = zone_summary_template_md.format(zone_index_to_name[zone_initial], zone_pickup_count, len(df_original))

# The app layout
app.layout = html.Div(className='app-body', children=[
    # Stores
    dcc.Store(id='map_clicks', data=0),
    dcc.Store(id='zone', data=zone_initial),
    dcc.Store(id='trip_start', data=trip_start_initial),
    dcc.Store(id='trip_end', data=trip_end_initial),
    dcc.Store(id='heatmap_limits', data=heatmap_limits_initial),
    # About the app + logos
    html.Div(className="row", children=[
        html.Div(className='twelve columns', children=[
            html.Div(style={'float': 'left'}, children=[
                    html.H1('Dash and Vaex: Big data exposed'),
                    html.H4(f'Exploring {len(df_original)//1_000_000:} Million Taxi trips in Real Time')
                ]
            ),
            html.Div(style={'float': 'right'}, children=[
                html.A(
                    html.Img(
                        src=app.get_asset_url("vaex-logo.png"),
                        style={'float': 'right', 'height': '35px', 'margin-top': '20px'}
                    ),
                    href="https://vaex.io/"),
                html.A(
                    html.Img(
                        src=app.get_asset_url("dash-logo.png"),
                        style={'float': 'right', 'height': '75px'}
                    ),
                    href="https://dash.plot.ly/")
            ]),
        ]),
    ]),
    # Control panel
    html.Div(className="row", id='control-panel', children=[
        html.Div(className="four columns pretty_container", children=[
            dcc.Loading(
                className="loader",
                id="loading",
                type="default",
                children=[
                    html.Div(id='loader-trigger-1', style={"display": "none"}),
                    html.Div(id='loader-trigger-2', style={"display": "none"}),
                    html.Div(id='loader-trigger-3', style={"display": "none"}),
                    html.Div(id='loader-trigger-4', style={"display": "none"}),
                    dcc.Markdown(id='data_summary_filtered', children=data_summary_filtered_md),
                    html.Progress(id="selected_progress", max=f"{len(df_original)}", value=f"{len(df_original)}"),
                ]),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select pick-up hours'),
            dcc.RangeSlider(id='hours',
                            value=[0, 23],
                            min=0, max=23,
                            marks={i: str(i) for i in range(0, 24, 3)}),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select pick-up days'),
            dcc.Dropdown(id='days',
                         placeholder='Select a day of week',
                         options=[{'label': 'Monday', 'value': 0},
                                  {'label': 'Tuesday', 'value': 1},
                                  {'label': 'Wednesday', 'value': 2},
                                  {'label': 'Thursday', 'value': 3},
                                  {'label': 'Friday', 'value': 4},
                                  {'label': 'Saturday', 'value': 5},
                                  {'label': 'Sunday', 'value': 6}],
                         value=[],
                         multi=True),
        ]),
    ]),

    # The Visuals
    dcc.Tabs(id='tab', children=[
        dcc.Tab(label='Popular destinations in New York', children=[
            html.Div(className="row", children=[
                html.Div(className="eight columns pretty_container", children=[
                    dcc.Markdown(id='zone_summary', children=zone_summary_md),
                    dcc.Graph(id='geomap_figure',
                              figure=create_figure_geomap(geomap_data_initial, zone_initial),
                              config={"modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                ]),
                html.Div(className="four columns pretty_container", children=[
                    dcc.Graph(id='flow_sunburst_figure',
                              figure=figure_sunburst_initial),
                ])
            ]),
            html.Div(className="row", children=[
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='flow_sankey_figure',
                              figure=figure_sankey_initial,
                              config={"modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                ]),
                html.Div(className="fix columns pretty_container", children=[
                    dash_table.DataTable(id='table', columns=[
                            {'name': 'Destination Borough', 'id': 'borough'},
                            {'name': 'Destination zone', 'id': 'zone'},
                            {'name': 'Number of trips', 'id': 'count_trips'},
                        ],
                        data=table_records_intitial,
                        style_data_conditional=table_style_initial,
                        style_as_list_view=True,
                    )
                ]),
            ]),
        ]),
        dcc.Tab(label='Trip planner', children=[
            html.Div(className="row", children=[
                html.Div(className="seven columns pretty_container", children=[
                    dcc.Markdown(children='_Click on the map to select trip start and destination._'),
                    dcc.Graph(id='heatmap_figure',
                              figure=create_figure_heatmap(heatmap_data_initial,
                                                           heatmap_limits_initial,
                                                           trip_start_initial,
                                                           trip_end_initial),
                              config={"modeBarButtonsToRemove": ['lasso2d', 'select2d', 'hoverCompareCartesian']})
                ]),
                html.Div(className="five columns pretty_container", children=[
                            dcc.Graph(id='trip_summary_amount_figure'),
                            dcc.Graph(id='trip_summary_duration_figure'),
                            dcc.Markdown(id='trip_summary_md'),
                ])
            ]),
        ]),
    ]),
    html.Hr(),
    dcc.Markdown(children=about_md),

])


# Heatmap related computation callbacks

# Map figure
@app.callback([Output('heatmap_figure', 'figure'),
               Output('loader-trigger-4', 'children')],
              [Input('days', 'value'),
               Input('hours', 'value'),
               Input('heatmap_limits', 'data'),
               Input('trip_start', 'data'),
               Input('trip_end', 'data')],
              prevent_initial_call=True)
def update_heatmap_figure(days, hours, heatmap_limits, trip_start, trip_end):
    data_array = compute_heatmap_data(days, hours, heatmap_limits)
    return create_figure_heatmap(data_array, heatmap_limits, trip_start, trip_end), "trigger loader"


# HeatMap navigation (zoom/pan)
@app.callback(
    Output('heatmap_limits', 'data'),
    [Input('heatmap_figure', 'relayoutData')],
    [State('heatmap_limits', 'data')],
    prevent_initial_call=True)
def update_limits(relayoutData, heatmap_limits):
    logger.info('Interaction: map/zoom on heatmap detected: relayoutData=%r heatmap_limits=%r', relayoutData, heatmap_limits)
    if relayoutData is None:
        raise dash.exceptions.PreventUpdate
    elif relayoutData is not None and 'xaxis.range[0]' in relayoutData:
        d = relayoutData
        heatmap_limits = [[d['xaxis.range[0]'], d['xaxis.range[1]']], [d['yaxis.range[0]'], d['yaxis.range[1]']]]
    else:
        raise dash.exceptions.PreventUpdate
        if heatmap_limits is None:
            heatmap_limits = heatmap_limits_initial
    return heatmap_limits


# Heatmap interaction (clicking)

@app.callback([Output('map_clicks', 'data'),
               Output('trip_start', 'data'),
               Output('trip_end', 'data')],
              [Input('heatmap_figure', 'clickData')],
              [State('map_clicks', 'data'),
               State('trip_start', 'data'),
               State('trip_end', 'data')],
              prevent_initial_call=True)
def click_heatmap_action(click_data_heatmap, map_clicks, trip_start, trip_end):
    logger.info('Interaction: click on heatmap detected: %r', click_data_heatmap)
    if click_data_heatmap is not None:
        point = click_data_heatmap['points'][0]['x'], click_data_heatmap['points'][0]['y']
        new_location = point[0], point[1]
        # the 1st and 3rd and 5th click change the start point
        if map_clicks % 2 == 0:
            trip_start = new_location
            trip_end = None  # and reset the end point
        else:
            # the 2nd, 4th etc set the end point
            trip_end = new_location
        map_clicks += 1
    logger.info('Interaction: trip_start=%r trip_end=%r', trip_start, trip_end)
    return map_clicks, trip_start, trip_end


# Geographical map

# Geographical map click on geojson area

@app.callback(Output('zone', 'data'),
              [Input('geomap_figure', 'clickData'),
               Input('flow_sunburst_figure', 'clickData'),
               Input('flow_sankey_figure', 'clickData'),
               Input('table', 'active_cell')],
              [State('zone', 'data'),
               State('table', 'data')],
              prevent_initial_call=True)
def click_action(click_data_geomap, click_data_sunburst, click_data_sunkey, click_data_table, zone, table_data):

    # What triggered the callback
    trg = dash.callback_context.triggered
    logger.info('Interaction: click on popular destinations tab detected: %r', trg)

    if trg is not None:
        component = trg[0]['prop_id'].split('.')[0]

        if component == 'geomap_figure':
            zone_name = trg[0]['value']['points'][0]['location']
            zone = zone_name_to_index[zone_name]

        if (component == 'flow_sunburst_figure') | (component == 'flow_sankey_figure'):
            zone_name = trg[0]['value']['points'][0]['label']
            if zone_name in zone_name_to_index.keys():
                zone = zone_name_to_index[zone_name]
            else:
                raise dash.exceptions.PreventUpdate()

        if component == 'table':
            if trg[0]['value']['column_id'] == 'zone':
                table_row = trg[0]['value']['row']
                zone_name = table_data[table_row]['zone']
                zone = zone_name_to_index[zone_name]
            else:
                raise dash.exceptions.PreventUpdate()

    # if click_data_geomap is not None:
    #     zone_name = click_data_geomap['points'][0]['location']
    #     zone = zone_name_to_index[zone_name]
    return zone


# Geographical map data
@app.callback([Output('geomap_figure', 'figure'),
               Output('data_summary_filtered', 'children'),
               Output('zone_summary', 'children'),
               Output('selected_progress', 'value'),
               Output('loader-trigger-3', 'children')],
              [Input('days', 'value'),
               Input('hours', 'value'),
               Input('zone', 'data')],
              [State('geomap_figure', 'figure')],
              prevent_initial_call=True)
def update_geomap_figure(days, hours, zone, current_figure):
    logger.info('Figure: updating geo map for: days=%r hours=%r zone=%r', days, hours, zone)

    zoom = current_figure['layout']['mapbox']['zoom']
    center = current_figure['layout']['mapbox']['center']

    pickup_counts = compute_geomap_data(days, hours)
    fig = create_figure_geomap(pickup_counts, zone, zoom=zoom, center=center)

    # we piggy back on the calculated pickup_counts to calculate what are the # filtered rows
    # instead of doing another calculation / pass over the data
    count = pickup_counts.sum()
    markdown_text = data_summary_filtered_md_template.format(count)

    zone_pickup_count = pickup_counts[zone]
    zone_summary_md = zone_summary_template_md.format(zone_index_to_name[zone], zone_pickup_count, len(df_original))

    return fig, markdown_text, zone_summary_md, str(count), "trigger loader"


# Flow section
@app.callback(
    [Output('flow_sankey_figure', 'figure'),
     Output('flow_sunburst_figure', 'figure'),
     Output('table', 'data'),
     Output('table', 'style_data_conditional'),
     Output('loader-trigger-1', 'children')
     ],
    [Input('days', 'value'),
     Input('hours', 'value'),
     Input('zone', 'data'),
     ], prevent_initial_call=True
)
def update_flow_figures(days, hours, zone):
    logger.info('Figure: update sankey and sunburst for days=%r hours=%r zone=%r', days, hours, zone)
    flow_data = compute_flow_data(days, hours, zone)
    df_outflow_top = vaex.from_dict(flow_data['outflow_top'])
    df_outflow_rest = vaex.from_dict(flow_data['outflow_rest'])
    df_outflow_borough = vaex.from_dict(flow_data['outflow_borough'])

    pickup_zone = zone
    fig_sankey = create_figure_sankey(df_outflow_top, df_outflow_rest, df_outflow_borough, pickup_zone)
    fig_sunburst = create_figure_sunburst(df_outflow_top, df_outflow_rest, df_outflow_borough, pickup_zone)
    table_records, table_style = create_table_data(df_outflow_top)

    return fig_sankey, fig_sunburst, table_records, table_style, 'trigger loader'


# Trip plotting

@app.callback([Output('trip_summary_amount_figure', 'figure'),
               Output('trip_summary_duration_figure', 'figure'),
               Output('trip_summary_md', 'children'),
               Output('loader-trigger-2', 'children')],
              [Input('days', 'value'),
               Input('hours', 'value'),
               Input('trip_start', 'data'),
               Input('trip_end', 'data')]
              )
def trip_details_summary(days, hours, trip_start, trip_end):
    if trip_start is None or trip_end is None:
        fig_empty = create_figure_empty()
        if trip_start is None:
            text = '''Please select a start location on the map.'''
        else:
            text = '''Please select a destination location on the map.'''
        return fig_empty, fig_empty, text, "trigger loader"

    trip_detail_data = compute_trip_details(days, hours, trip_start, trip_end)
    logger.info('Figure: trip details summary for %r to %r', trip_start, trip_end)

    counts = trip_detail_data['counts']
    counts_total = np.array(trip_detail_data['counts_total'])
    counts_duration = np.array(trip_detail_data['counts_duration'])
    fig_amount, peak_amount = create_figure_histogram(df_original.bin_edges(df_original.total_amount, limits_amount, shape=bins),
                                                      counts_total,
                                                      title=None,
                                                      xlabel='Total amount [$]',
                                                      ylabel='Numbe or rides')
    # The trip duration
    fig_duration, peak_duration = create_figure_histogram(df_original.bin_edges(df_original.trip_duration_min, limits_amount, shape=bins),
                                                          counts_duration,
                                                          title=None,
                                                          xlabel='Trip duration [min]',
                                                          ylabel='Numbe or rides')

    trip_stats = f'''
                    **Trip statistics:**
                    - Number of rides: {counts}
                    - Most likely trip total cost: ${peak_amount}
                    - Most likely trip duration: {peak_duration} minutes
                    '''

    return fig_amount, fig_duration, trip_stats, "trigger loader"


if __name__ == '__main__':
    app.run_server(debug=True)
