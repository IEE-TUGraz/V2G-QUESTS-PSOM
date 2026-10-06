import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import pickle
from dash import Dash, html, Input, Output
import dash_leaflet as dl
import json


def import_tartu_data(data_folder: str, file: str) -> gpd.GeoDataFrame:
    """
    Import Tartu data from the given folder, specified by the file name.

    :param data_folder: The folder where the data is stored.
    :param file: The relative path to the file in the folder structure.

    :return: The imported data.
    """
    
    filepath = os.path.join(data_folder, file)
    # check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    
    # excel file
    if file.endswith('.xlsx'):
        data = pd.read_excel(filepath)
    # csv file
    elif file.endswith('.csv'):
        data = pd.read_csv(filepath, sep=',', decimal='.', encoding='latin1')
    else:
        raise ValueError(f"File format not supported: {file}")
    
    points_A = [shapely.geometry.Point(xy) for xy in zip(data['y1'], data['x1'])] # needs to be switched for later use in epsg:4326
    points_B = [shapely.geometry.Point(xy) for xy in zip(data['y2'], data['x2'])]
    class_data = data['Class']
    data = gpd.GeoDataFrame(data, geometry=[shapely.geometry.LineString([A, B]) for A, B in zip(points_A, points_B)])
    data = data[['geometry']]
    data['class'] = class_data
    

    data = data.set_crs(epsg=3301)
    data = data.to_crs(epsg=4326)

    return data


def import_tartu_trafo(data_folder: str, file: str) -> gpd.GeoDataFrame:
    """
    Import Tartu data from the given folder, specified by the file name.

    :param data_folder: The folder where the data is stored.
    :param file: The relative path to the file in the folder structure.

    :return: The imported data.
    """
    
    filepath = os.path.join(data_folder, file)
    # check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    
    data = pd.read_csv(filepath, sep=',', decimal='.', encoding='latin1')

    data_gpd = gpd.GeoDataFrame(data, geometry=[shapely.geometry.Point(xy) for xy in zip(data['y'], data['x'])])

    if file == 'MVNetwork\DistrTrafo.csv':
        rated_voltage = data['Rated primary voltage (V)']

    data = data_gpd[['geometry']]

    if file == 'MVNetwork\DistrTrafo.csv':
        data['rated_voltage'] = rated_voltage


    data = data.set_crs(epsg=3301)
    data = data.to_crs(epsg=4326)

    return data


def get_node_connections(lines: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, tol: float = 5e-5) -> gpd.GeoDataFrame:
    """
    Characterize the line objects based on the nodes that they are connected to.
    :param lines: The GeoDataFrame that contains the line data.
    :param nodes: The GeoDataFrame that contains the node data.

    :return: A nested dictionary that contains the line objects index as keys and a dictionary with 
             information about the type of connection and the connected nodes as value.
    """

    line_endpoints = [(line.geometry.coords[0], line.geometry.coords[-1]) for line in lines.itertuples()]
    start_points = [line[0] for line in line_endpoints]
    end_points = [line[1] for line in line_endpoints]

    line_indices = [line.Index for line in lines.itertuples()]


    # create empty geodataframe with the line indices as index

    line_objects = gpd.GeoDataFrame(index=line_indices,
                                    columns=['node1', 'node1_type', 'node1_point',
                                             'node2', 'node2_type', 'node2_point',
                                             'linetype', 'is_slack'])
    line_objects['linetype'] = 'MV'
    extra_node = 0

    for start, end, index in zip(start_points, end_points, line_indices):
        connected_nodes = []
        connected_nodes_types = []
        start_dummy = False
        end_dummy = False
        for node, node_id, node_type in zip(nodes.geometry, nodes.index, nodes['type']):

            if shapely.dwithin(node, shapely.geometry.Point(*start), tol):
                connected_nodes.append(node_id)
                connected_nodes_types.append(node_type)
                start_dummy = True
            elif shapely.dwithin(node, shapely.geometry.Point(*end), tol):
                connected_nodes.append(node_id)
                end_dummy = True
                connected_nodes_types.append(node_type)

     
        if len(connected_nodes) == 1:
            node_type = connected_nodes_types[0]
            line_objects.at[index, 'node1'] = connected_nodes[0]
            line_objects.at[index, 'node1_type'] = node_type
            line_objects.at[index, 'node1_point'] = nodes.at[connected_nodes[0], 'geometry']
            line_objects.at[index, 'is_slack'] = 0
            if start_dummy:

                line_objects.at[index, 'node2'] = str(extra_node)
                line_objects.at[index, 'node2_type'] = 'outside'
                line_objects.at[index, 'node2_point'] = shapely.geometry.Point(*end)
                line_objects.at[index, 'is_slack'] = 0
                extra_node += 1
            elif end_dummy:
                line_objects.at[index, 'node2'] = str(extra_node)
                line_objects.at[index, 'node2_type'] = 'outside'
                line_objects.at[index, 'node2_point'] = shapely.geometry.Point(*start)
                line_objects.at[index, 'is_slack'] = 0
                extra_node += 1

        elif len(connected_nodes) == 2:
            node1_type = connected_nodes_types[0]
            node2_type = connected_nodes_types[1]
            line_objects.at[index, 'node1'] = connected_nodes[0]
            line_objects.at[index, 'node1_type'] = node1_type
            line_objects.at[index, 'node1_point'] = nodes.at[connected_nodes[0], 'geometry']
            line_objects.at[index, 'node2'] = connected_nodes[1]
            line_objects.at[index, 'node2_type'] = node2_type
            line_objects.at[index, 'node2_point'] = nodes.at[connected_nodes[1], 'geometry']
            if node1_type == 'slack' or node2_type == 'slack':
                line_objects.at[index, 'is_slack'] = 1
            else:
                line_objects.at[index, 'is_slack'] = 0

        elif len(connected_nodes) > 2:
            print('Something is wrong')

    line_objects = line_objects.dropna()
    line_objects = line_objects.reset_index(drop=True)

    return line_objects


def connect_lines_and_substations(lines: gpd.GeoDataFrame, data_substation: gpd.GeoDataFrame, tol: float=1e-4) -> gpd.GeoDataFrame:

    line_endpoints = [(line.geometry.coords[0], line.geometry.coords[-1]) for line in lines.itertuples()]
    start_points = [line[0] for line in line_endpoints]
    end_points = [line[1] for line in line_endpoints]

    # # iterate though all substations and change all start and endpoints withina certain radius to the substation
    # for substation_coords in data_substation['geometry']:
    #     for start in start_points:
    #         if shapely.dwithin(substation_coords, shapely.geometry.Point(start), tol):
    #             start_points[start_points.index(start)] = substation_coords
    #     for end in end_points:
    #         if shapely.dwithin(substation_coords, shapely.geometry.Point(end), tol):
    #             end_points[end_points.index(end)] = substation_coords

    # iterate through all start and endpoints, when they are within a certain radius of a substation, change them to the NEAREST substation
    for start in start_points:
        distances = [shapely.geometry.Point(start).distance(substation) for substation in data_substation['geometry']]
        if min(distances) < tol:
            start_points[start_points.index(start)] = data_substation['geometry'][distances.index(min(distances))]
    for end in end_points:
        distances = [shapely.geometry.Point(end).distance(substation) for substation in data_substation['geometry']]
        if min(distances) < tol:
            end_points[end_points.index(end)] = data_substation['geometry'][distances.index(min(distances))]
            

    # return the new line objects
    lines['geometry'] = [shapely.geometry.LineString([start, end]) for start, end in zip(start_points, end_points)]

    return lines


def identify_nodes(lines: gpd.GeoDataFrame, data_substation: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    line_endpoints = [(line.geometry.coords[0], line.geometry.coords[-1]) for line in lines.itertuples()]
    start_points = [line[0] for line in line_endpoints]
    end_points = [line[1] for line in line_endpoints]
    all_points = start_points + end_points
    all_points = [shapely.geometry.Point(point) for point in all_points if shapely.geometry.Point(point) not in data_substation['geometry'].values]
    node_points = [point for point in all_points if all_points.count(point) > 1]
    node_type_node = ['node' for _ in node_points]
    node_type_substation = ['substation' for _ in data_substation['geometry']]
    nodes_and_substations = data_substation['geometry'].values.tolist() + node_points
    nodes = gpd.GeoDataFrame(geometry=nodes_and_substations)
    nodes['type'] = node_type_substation + node_type_node
    nodes = nodes.reset_index(drop=True)

    return nodes


def connect_data(data_lines: gpd.GeoDataFrame, linetype: str) -> gpd.GeoDataFrame:
    """
    Connect lines that are connected to each other. This function will return a new 
    GeoDataFrame with the connected lines.
    :param data_lines: The GeoDataFrame that contains the line data.
    :param linetype: The linetype of the line data.
    
    :return: The connected GeoDataFrame.
    """

    line_segments = []
    line_classes = []
    for line_segment, line_class in zip(data_lines.geometry, data_lines['class']):
        if isinstance(line_segment, LineString):
            line_segments.append(line_segment)
            line_classes.append(line_class)
        else:
            print('Warning: Line is not of type "LineString".')
            print(type(line_segment))
            # append the individual line segments to the list
            line_segments.extend([line for line in line_segment.geoms])
            line_class.extend([line_class for line in line_segment.geoms])

    print('Number of line segments before merging: ', len(line_segments))
    multi_line = MultiLineString(line_segments)
    merged_line = linemerge(multi_line)
    ml_list = list(merged_line.geoms) if merged_line.geom_type == 'MultiLineString' else [merged_line]

    # get all geometries with class 240 part
    ls_240 = [line_segment for line_segment, line_class in zip(line_segments, line_classes) if '240' in line_class]
    # get all geometries with class 185 part
    ls_185 = [line_segment for line_segment, line_class in zip(line_segments, line_classes) if '185' in line_class]

    # check if ml linestring contains 240 part
    ml_240 = [line for line in ml_list if any([line.contains(ls) for ls in ls_240])]
    # check if ml linestring contains 185 part
    ml_185 = [line for line in ml_list if any([line.contains(ls) for ls in ls_185]) and line not in ml_240]
    # put the rest in 'rest' list
    rest = [line for line in ml_list if line not in ml_240 and line not in ml_185]
    ml_gdf = gpd.GeoDataFrame(columns=['geometry','linetype'])

    all_gemoetries = ml_240 + ml_185 + rest
    all_linetypes = ['240' for _ in ml_240] + ['185' for _ in ml_185] + ['rest' for _ in rest]

    ml_gdf['geometry'] = all_gemoetries
    ml_gdf['linetype'] = all_linetypes

    print('Number of line segments after merging: ', len(ml_gdf))

    return ml_gdf


def crop_data(data: gpd.GeoDataFrame, segment: shapely.Polygon) -> gpd.GeoDataFrame:
    """
    Crop the data to the specified boundary.
    Any object that is outside the specified boundary will be removed as a whole.
    The function will return a new GeoDataFrame with the cropped data.
    :param data: The GeoDataFrame that needs to be cropped.

    :return: The cropped GeoDataFrame.
    """

    indices = []
    idx = 0
    data = data.reset_index(drop=True) # make sure the index starts at 0
    for obj in data.geometry:
        bool = True
        if not segment.contains(obj) and not segment.crosses(obj):
                bool = 0
        if bool:
            indices.append(idx)    
        idx +=1
    data_cropped = data.iloc[indices]
    return data_cropped


def combine_nodes(nodes: gpd.GeoDataFrame, tol: float=1e-3) -> gpd.GeoDataFrame:
    """
    Combine nodes that are very close to each other.
    :param nodes: The GeoDataFrame containing the nodes.
    :param tol: The tolerance within which nodes are considered to be the same.
    :return: The GeoDataFrame with combined nodes.
    """
    # iterate through all substations and set substations within tol to the same coordinates
    for i, node in nodes.iterrows():
        for j, node2 in nodes.iterrows():
            if i != j:
                if node['geometry'].distance(node2['geometry']) < tol:
                    # set the coordinates of node2 to node
                    nodes.at[j, 'geometry'] = node['geometry']

    # finally remove all duplicates
    nodes = nodes.drop_duplicates(subset='geometry')
    nodes = nodes.reset_index(drop=True)
    return nodes


def add_slack_node(nodes: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add a slack node to the network.
    The slack node is the main station to the east of the network.
    :param nodes: The GeoDataFrame containing the nodes.
    :param main_station: The GeoDataFrame containing the main stations.
    :return: The GeoDataFrame with the slack node added.
    """

    start_points = []
    line_endpoints = [(line.geometry.coords[0], line.geometry.coords[-1]) for line in lines.itertuples()]
    start_points = [line[0] for line in line_endpoints]
    end_points = [line[1] for line in line_endpoints]

    # get the eastern most point
    x_start = [point[0] for point in start_points]
    x_end = [point[0] for point in end_points]
    y_start = [point[1] for point in start_points]
    y_end = [point[1] for point in end_points]
    x = x_start + x_end
    y = y_start + y_end
    slack_x = max(x)
    slack_y = y[x.index(slack_x)]
    # append slack node to the nodes
    idx_max = nodes.index.max()
    slack_point = shapely.geometry.Point(slack_x, slack_y)
    nodes.loc[idx_max+1] = [slack_point, 'slack']
    return nodes


def plot_network():

    # load the data from the pickle files
    line_objects = pd.read_pickle((os.path.join('pickle_files', 'line_objects.pkl')))
    node_objects = pd.read_pickle((os.path.join('pickle_files', 'node_objects.pkl')))

    # select relevant columns
    df_bus = node_objects[['geometry', 'node_id']].copy()
    df_bus['lat'] = df_bus['geometry'].apply(lambda x: x.y)
    df_bus['long'] = df_bus['geometry'].apply(lambda x: x.x)
    df_bus.drop(columns=['geometry'], inplace=True)
    df_branch = line_objects[['node1', 'node2']].copy()
    # rename columns
    df_branch = df_branch.rename(columns={'node1': 'fbus', 'node2': 'tbus'})


    ids = list(df_bus.node_id.values)
    bus_to_index = {bus: idx for idx, bus in enumerate(df_bus.node_id.values)}
    lat_values = df_bus.lat.values
    lon_values = df_bus.long.values


    nodes_radii = [10 for _ in range(len(df_bus))]
    nodes = [{'id': ids[i], 'lat': lat_values[i], 'lon': lon_values[i], 'radius': nodes_radii[i]} for i in range(len(df_bus))]


    connections = []
    colors = []
    click_nodes = []

    for idx, line in enumerate(df_branch.iterrows()):
        fbus = bus_to_index[line[1]['fbus']]
        tbus = bus_to_index[line[1]['tbus']]
        
        connections.append([(lat_values[fbus], lon_values[fbus]), (lat_values[tbus], lon_values[tbus])])
        colors.append("#08AEEF")

    bounds = [[min(lat_values), min(lon_values)], [max(lat_values), max(lon_values)]]

    # Dash app layout
    app = Dash()
    app.layout = html.Div([
        dl.Map(
            id='map',
            style={'width': '100%', 'height': '50vh'},
            center=[(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2],
            zoom=12,    
            children=[   
        
        # dl.TileLayer(url="https://tile.jawg.io/jawg-light/{z}/{x}/{y}{r}.png?access-token=K29acH6XcPPrHIAdDENjCaooa47B7CDZieD4CPBI2EiVrg8iKRjqiS80D8ACx9Np"),
        dl.TileLayer(),
            # Network connections
        *[
            dl.Polyline(positions=connection,
                        color=colors[idx],
                        weight=4,
                        ) for idx, connection in enumerate(connections)
        ],
            # Network nodes as circles
        *[
            dl.CircleMarker(
                center=(node['lat'], node['lon']),
                radius=node['radius'],
                fillColor="#08AEEF",
                color="black",
                fill=True,
                fillOpacity=1,
                children=dl.Popup(node['id'])
            ) for node in nodes
        ]
        ,
    ],

    ),
    html.Div(id='out'),
    ], style={'height': '100vh'})

    @app.callback(
        Output('out', 'children'),
        Output('map', 'children'),
        Input('map', 'clickData'),
        prevent_initial_call=True
    )
    def do(click_data):
        # extract coordinates form click_data
        coordinates = click_data['latlng']
        lat, lon = coordinates.values()
        click_nodes.append([lat, lon])

        return json.dumps(click_nodes), [dl.TileLayer(),
                                        *[dl.Marker(position=[lat, lon]) for lat, lon in click_nodes],
                *[dl.Polyline(positions=connection,
                            color=colors[idx],
                            weight=4,
                            ) for idx, connection in enumerate(connections)
            ],
            # Network nodes as circles
        *[
            dl.CircleMarker(
                center=(node['lat'], node['lon']),
                radius=node['radius'],
                fillColor="#08AEEF",
                color="black",
                fill=True,
                fillOpacity=1,
                children=dl.Popup(node['id'])
            ) for node in nodes
        ]]
    # Run the app
    app.run(debug=False)


