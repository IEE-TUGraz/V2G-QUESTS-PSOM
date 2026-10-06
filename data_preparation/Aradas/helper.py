import os
import pickle
import time

import dash_leaflet as dl
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dash import Dash, html, Input, Output, State, dcc, ctx
from scipy.spatial import cKDTree


def get_substations():
    # load boundary data using pickle
    with open('pickle_files/boundary.pkl', 'rb') as f:
        boundary = pickle.load(f)
    segment = boundary.geometry[0]

    # load secondary substation data from csv
    data = gpd.read_file('raw_data/secondary_substations.csv')

    # transform "Geograpic Coordinates" column to shapely points
    data['geometry'] = data['Geographic Coordinates'].apply(
        lambda x: gpd.points_from_xy([float(x.split(',')[1])], [float(x.split(',')[0])])[0])
    data.drop(columns=['Usage Level [%]', 'Installed Power [kVA]', 'Geographic Coordinates'], inplace=True)

    # check if points are within boundary
    data = data[data['geometry'].apply(lambda x: x.within(segment))]
    data.reset_index(drop=True, inplace=True)
    # save nodes to pickle
    data.to_pickle('pickle_files/node_objects.pkl')
    return data


def plot_network(mode='new'):
    node_filepath = os.path.join('pickle_files', 'node_objects.pkl')
    clicked_data_filepath = os.path.join('intermediate_data', 'clicked_data.csv')

    node_objects = pd.read_pickle(node_filepath)

    df_bus = node_objects.copy()
    df_bus['lat'] = df_bus['geometry'].apply(lambda x: x.y)
    df_bus['lon'] = df_bus['geometry'].apply(lambda x: x.x)
    df_bus['bus'] = df_bus.index

    nodes = [
        {
            'id': row['bus'],
            'lat': row['lat'],
            'lon': row['lon'],
            'radius': 10
        }
        for _, row in df_bus.iterrows()
    ]

    bounds = [[df_bus['lat'].min(), df_bus['lon'].min()], [df_bus['lat'].max(), df_bus['lon'].max()]]
    app = Dash()

    circle_markers = [
        dl.CircleMarker(
            center=(node['lat'], node['lon']),
            radius=node['radius'],
            fillColor="#08AEEF",
            color="black",
            fill=True,
            fillOpacity=1,
            children=dl.Popup(str(node['id']))
        ) for node in nodes
    ]

    if mode == 'existing' and os.path.exists(clicked_data_filepath):
        df = pd.read_csv(clicked_data_filepath)
        coords = df[['lat', 'lon']].values.tolist()
        line_segments = [coords[i:i + 2] for i in range(0, len(coords) - 1, 2)]

        polylines = [
            dl.Polyline(
                positions=segment,
                id=f'line-{i}',
                color='red',
                weight=4
            ) for i, segment in enumerate(line_segments)
        ]

        app.layout = html.Div([
            dl.Map(
                id='map',
                center=[sum(x) / 2 for x in zip(*bounds)],
                zoom=12,
                style={'height': '90%', 'width': '100%'},
                children=[dl.TileLayer()] + circle_markers + polylines
            ),
            html.Div([
                dcc.Store(id='segments-store', data=line_segments),
                html.Button("Save Changes", id='save-changes'),
                html.Div(id='out')
            ], style={'padding': '10px'})
        ], style={'height': '100vh', 'width': '100vw'})

        # Create a single callback to detect which static line was clicked
        @app.callback(
            Output('segments-store', 'data'),
            Output('map', 'children'),
            [Input(f'line-{i}', 'n_clicks') for i in range(len(line_segments))],
            State('segments-store', 'data'),
            prevent_initial_call=True
        )
        def delete_static_line(*args):
            segments = args[-1]
            triggered = ctx.triggered_id
            if triggered and triggered.startswith('line-'):
                index = int(triggered.split('-')[1])
                if 0 <= index < len(segments):
                    segments.pop(index)

            updated_polylines = [
                dl.Polyline(
                    positions=seg,
                    id=f'line-{i}',
                    color='red',
                    weight=4
                ) for i, seg in enumerate(segments)
            ]
            map_children = [dl.TileLayer()] + circle_markers + updated_polylines
            return segments, map_children

        # Save final data
        @app.callback(
            Output('out', 'children'),
            Input('save-changes', 'n_clicks'),
            State('segments-store', 'data'),
            prevent_initial_call=True
        )
        def save_segments(n_clicks, segments):
            flat = [pt for seg in segments for pt in seg]
            df = pd.DataFrame(flat, columns=['lat', 'lon'])
            df.to_csv(clicked_data_filepath, index=False)
            return f"Changes saved to '{clicked_data_filepath}'."

    else:
        # New mode
        existing_segments = []
        if os.path.exists(clicked_data_filepath):
            df = pd.read_csv(clicked_data_filepath)
            coords = df[['lat', 'lon']].values.tolist()
            existing_segments = [coords[i:i + 2] for i in range(0, len(coords) - 1, 2)]

        existing_polylines = [
            dl.Polyline(positions=seg, color='red', weight=4, opacity=0.5)
            for seg in existing_segments
        ]

        app.layout = html.Div([
            dl.Map(
                id='map',
                center=[sum(x) / 2 for x in zip(*bounds)],
                zoom=12,
                style={'height': '90%', 'width': '100%'},
                children=[dl.TileLayer()] + circle_markers + existing_polylines
            ),
            html.Div([
                dcc.Store(id='click-store', data=[]),
                dcc.Store(id='existing-segments', data=existing_segments),
                html.Button("Save & Finish", id='save-button'),
                html.Div(id='out')
            ], style={'padding': '10px'})
        ], style={'height': '100vh', 'width': '100vw'})

        @app.callback(
            Output('click-store', 'data'),
            Output('map', 'children'),
            Input('map', 'clickData'),
            State('click-store', 'data'),
            State('existing-segments', 'data'),
            prevent_initial_call=True
        )
        def on_click(click_data, stored_clicks, existing_segs):
            if click_data is None:
                return stored_clicks, []

            lat, lon = click_data['latlng'].values()
            stored_clicks.append([lat, lon])

            new_segments = [stored_clicks[i:i + 2] for i in range(0, len(stored_clicks) - 1, 2)]
            new_lines = [dl.Polyline(positions=seg, color='blue', weight=4) for seg in new_segments]
            markers = [dl.Marker(position=pt) for pt in stored_clicks]
            existing_lines = [dl.Polyline(positions=seg, color='red', weight=4, opacity=0.5) for seg in existing_segs]

            children = [dl.TileLayer()] + circle_markers + existing_lines + new_lines + markers
            return stored_clicks, children

        @app.callback(
            Output('out', 'children'),
            Input('save-button', 'n_clicks'),
            State('click-store', 'data'),
            State('existing-segments', 'data'),
            prevent_initial_call=True
        )
        def save_all(n_clicks, new_points, existing_segs):
            new_segments = [new_points[i:i + 2] for i in range(0, len(new_points) - 1, 2)]
            all_segments = existing_segs + new_segments
            flat = [pt for seg in all_segments for pt in seg]
            df = pd.DataFrame(flat, columns=['lat', 'lon'])
            df.to_csv(clicked_data_filepath, index=False)
            return f"Changes saved to '{clicked_data_filepath}'."
    app.run(debug=True)


def prepare_line_and_node_data(
        input_filepath_lines=os.path.join('intermediate_data, clicked_data.csv'),
        input_filepath_nodes=os.path.join('pickle_files', 'node_objects.pkl'),
        output_filepath_lines=os.path.join('final_data', 'line_data.csv'),
        output_filepath_nodes=os.path.join('final_data', 'node_data.csv')
):
    # --- Load CSV Line Endpoints ---
    df = pd.read_csv(input_filepath_lines)
    raw_points = df[['lat', 'lon']].values

    # Ensure even number of points
    if len(raw_points) % 2 != 0:
        raise ValueError("CSV has an odd number of points; lines must have 2 endpoints each.")

    # --- Load Node Coordinates ---
    node_gdf = pd.read_pickle(input_filepath_nodes)
    node_coords = np.array([(geom.y, geom.x) for geom in node_gdf['geometry']])  # (lat, lon)

    # --- KDTree for Nearest-Node Search ---
    tree = cKDTree(node_coords)

    # --- Snap Each Endpoint ---
    _, indices = tree.query(raw_points, k=1)
    snapped_points = node_coords[indices]  # Replace with node coordinates

    # --- Save Updated CSV ---
    updated_df = pd.DataFrame(snapped_points, columns=['lat', 'lon'])
    updated_df["i"] = indices  # Add nearest node index

    # only take the columns "i"
    lines_data = updated_df["i"].values
    # change from int to Node_{i} format
    lines_data = [f"Node_{i}" for i in lines_data]
    # transform the lines data to a list of tuples
    lines_data = [(lines_data[i], lines_data[i + 1]) for i in range(0, len(lines_data) - 1, 2)]
    line_objects = pd.DataFrame(lines_data, columns=["i", "j"])

    print(line_objects.head())
    # save the line data to a csv file
    line_objects.to_csv(output_filepath_lines, index=False)

    # save the coordinates and indices to a csv file
    df['lat'] = snapped_points[:, 0]
    df['lon'] = snapped_points[:, 1]
    df["i"] = indices  # Add nearest node index
    # remove duplicates in the dataframe
    df.drop_duplicates(subset=["i"], inplace=True)
    # sort according to indices in ascending order
    df.sort_values(by="i", inplace=True)
    # change i to Node_{i} format
    df["i"] = df["i"].apply(lambda x: f"Node_{x}")
    df.to_csv(output_filepath_nodes, index=False)
    return updated_df


def prepare_demand_data():
    # load the demand data
    print("Loading demand data...")
    start = time.time()
    filename_demand = os.path.abspath(
        "C:\GIT\V2G-QUESTS\data_preparation\Aradas\intermediate_data\consumption_data_new_CP7.csv")
    demand_data = pd.read_csv(
        filename_demand,
        sep=',',
    )

    # set time to datetime
    demand_data['Time'] = pd.to_datetime(demand_data['Time'], format="%Y-%m-%d %H:%M:%S%z")
    # instead of columns Time and all nodes, we want to have a column with the time and a column with the values for the nodes
    # melt the dataframe to have a column with the time and a column with the values for the nodes
    demand_data = pd.melt(demand_data, id_vars=["Time"], var_name="CP7", value_name="Active energy (kWh)")
    # replace nan values with 0
    demand_data.fillna(0, inplace=True)
    print("Demand data loaded in", time.time() - start, "seconds")
    print(demand_data.head())

    # loadzip code data from txt file
    filename_zipcode = os.path.abspath("C:\GIT\V2G-QUESTS\data\Aradas\raw_data\CP7_Portugal_nov2022.txt")
    # Load the file into a DataFrame
    print("Loading zipcode data...")
    start = time.time()
    zipcode_data = pd.read_csv(
        filename_zipcode,
        sep='\t',
        quotechar='"',
        na_values='NA'
    )
    # only take columns "CP7", "Long, 'Lat"
    zipcode_data = zipcode_data[["CP7", "Long", "Lat"]]
    print("Zipcode data loaded in", time.time() - start, "seconds")

    # only take the rows with "CP7" in the zipcode data that are also in the demand data
    zipcode_data = zipcode_data[zipcode_data["CP7"].isin(demand_data["CP7"])]
    # remove duplicates in the zipcode data
    zipcode_data.drop_duplicates(subset=["CP7"], inplace=True)

    # find closest node to ech zip code
    # # load node data from csv file
    node_objects = pd.read_csv('final_data/node_data.csv')
    node_coords = np.array([[row['lat'], row['lon']] for _, row in node_objects.iterrows()])
    # create a KDTree for fast nearest neighbor search
    tree = cKDTree(node_coords)
    # find the nearest node for each zip code
    _, indices = tree.query(zipcode_data[["Lat", "Long"]].values, k=1)
    # add nearest node to the zipcode data
    zipcode_data["Nearest Node"] = [f"Node_{i}" for i in indices]

    # add lat long to the demand data
    demand_data = demand_data.merge(zipcode_data, on="CP7", how="left")
    # rename Datetime to time, Long to lon, lat to lan, Nearest Node to i
    demand_data.rename(columns={"Time": "time", "Long": "lon", "Lat": "lat", "Nearest Node": "i"}, inplace=True)
    # drop lat and lon columns from demand data
    demand_data.drop(columns=["lat", "lon"], inplace=True)
    print(demand_data.head())
    # print all rows where i = "Node_58"
    abc_demand_data = demand_data[demand_data["i"] == "Node_58"]
    # get unique CP7 values
    unique_cp7 = abc_demand_data["CP7"].unique()
    print("Unique CP7 values:", unique_cp7)
    # print coordinates of the unique CP7 values
    for cp7 in unique_cp7:
        coords = zipcode_data[zipcode_data["CP7"] == cp7][["Lat", "Long"]].values[0]
        print(f"Coordinates of {cp7}: {coords}")

    # Assuming your current dataframe is called df
    demand_data['time'] = pd.to_datetime(demand_data['time'])  # Ensure 'time' is in datetime format

    demand_data_pivot = demand_data.pivot_table(
        index='time',
        columns='i',
        values='Active energy (kWh)',
        aggfunc='sum'  # Or 'mean', 'max', etc., depending on what makes sense
    )

    print(demand_data_pivot.head())
    # save the demand data to a csv file
    demand_data_pivot.to_csv("final_data//demand_data_all.csv", index=False)

    # make a histogram of the sum of each column
    demand_data_pivot.sum().plot(kind='hist', bins=20)
    plt.show()
    # get the column name with the highest sum
    highest_sum_col = demand_data_pivot.sum().idxmax()
    print("Column with highest sum:", highest_sum_col)
    # get the sum of the column with the highest sum
    highest_sum = demand_data_pivot[highest_sum_col].sum()
    print("Sum of column with highest sum:", highest_sum)

    center = (zipcode_data["Lat"].mean(), zipcode_data["Long"].mean())
    app = Dash()

    app.layout = html.Div([
        dl.Map(
            center=center,
            zoom=7,
            style={'height': '90vh', 'width': '100%'},
            children=[
                dl.TileLayer(),
                *[
                    dl.CircleMarker(
                        center=(row["Lat"], row["Long"]),
                        radius=5,
                        color='blue',
                        fill=True,
                        fillColor='blue' if row["CP7"] not in unique_cp7 else 'red',
                        fillOpacity=0.6,
                        id=str(row["CP7"]),
                        children=[dl.Popup(str(row["CP7"]))]
                    )
                    for _, row in zipcode_data.iterrows()
                ],
                *[
                    dl.Marker(
                        position=(x, y),
                        id=str(idx),
                        children=[dl.Popup(f"Node {idx}")],
                    )
                    for idx, (x, y) in enumerate(node_coords)
                ],
            ]
        )
    ])

    app.run(debug=True, use_reloader=False)


prepare_demand_data()

# slack_node = (40.617881171715005,-8.61707018812261)
# # load node data and add slack node
# node_objects = pd.read_pickle('pickle_files/node_objects.pkl')
# node_objects = pd.concat([node_objects, gpd.GeoDataFrame([{'geometry': gpd.points_from_xy([slack_node[1]], [slack_node[0]])[0]}])], ignore_index=True)
# node_objects.reset_index(drop=True, inplace=True)
# node_objects.to_pickle('pickle_files/node_objects.pkl')
