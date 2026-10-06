import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

import helper as h

"""
Network extraction and preprocessing script for the Annelinn MV network.

This module loads raw MV network data (lines and substations), crops it to the
Annelinn boundary, identifies network nodes and line connections, applies a
handful of manual corrections, and writes cleaned node/line tables suitable
for downstream use.

The main processing is performed by the :func:`main` function.

Files read (relative to project root)
-------------------------------------
- raw_data/MVNetwork/MVPart.xlsx
- raw_data/MVNetwork/DistrSubstation.csv
- raw_data/boundary.csv

Files written
-------------
- final_data/node_data.csv   (columns: i, lat, lon)
- final_data/line_data.csv   (columns: i, j)
- pickle_files/line_objects.pkl
- pickle_files/node_objects.pkl

Dependencies
------------
- pandas as pd
- geopandas as gpd
- shapely.geometry.Polygon
- your helper module `h` (functions: import_tartu_data, import_tartu_trafo,
  connect_data, combine_nodes, connect_lines_and_substations, identify_nodes,
  crop_data, add_slack_node, get_node_connections)
- os

Notes
-----
- The script expects `boundary.csv` to contain point coordinates in columns named
  `p1` (latitude) and `p2` (longitude) — adjust if your file uses different names.
- Current code sets CRS with the deprecated form ``{'init': 'epsg:4326'}``;
  prefer `annelinn.set_crs("EPSG:4326", inplace=True)` to avoid warnings.
- A small set of manual corrections (lines removed and nodes dropped) are
  applied; these are hard-coded in `lines_to_remove` and the node drop list.
"""
def main():
    """
        Build cleaned node and line tables from raw MV network data.

        Steps performed
        ---------------
        1. Load line element and substation files from `raw_data/`.
        2. Load Annelinn boundary points, convert to a GeoDataFrame and compose a
           polygon used for spatial cropping.
        3. Connect line geometry, combine nearby substations, and link lines to
           substations with a small tolerance.
        4. Identify network nodes (intersections / branch endpoints).
        5. Crop both lines and nodes to the Annelinn polygon.
        6. Add a slack node and build `line_objects` describing node-to-node links.
        7. Remove lines whose endpoints are tagged as "outside".
        8. Build a GeoDataFrame of unique node geometries and mark the eastern-most
           node as the slack node.
        9. Apply manual corrections (explicitly remove a small list of edges and
           drop two nodes).
        10. Persist intermediate objects to pickle and export final CSVs:
            - `final_data/node_data.csv` contains columns ["i", "lat", "lon"]
              where "i" is `Node_{index}`.
            - `final_data/line_data.csv` contains columns ["i", "j"] with node ids.

        Returns
        -------
        None
            The function writes files to disk and prints a preview of nodes to stdout.

        Side effects
        ------------
        - Writes CSV and pickle files as listed in the module docstring.
        - Prints `node_objects.head()` to stdout for quick inspection.

        Tips & gotchas
        --------------
        - If your boundary CSV uses different column names than `p1`/`p2`, update the
          call to `gpd.points_from_xy(...)`.
        - Replace the deprecated CRS assignment:
            annelinn.set_crs("EPSG:4326", inplace=True)
          instead of `annelinn.crs = {'init': 'epsg:4326'}`.
        - Manual corrections (lines_to_remove, node drops) are intentional and
          dataset-specific; review them before re-running on other datasets.

        Example
        -------
        Run the script from the repository root:

        >>> if __name__ == "__main__":
        >>>     main()

        """
    data_folder = 'raw_data'
    file_line_elements = 'MVNetwork\MVPart.xlsx'
    file_substation = 'MVNetwork\DistrSubstation.csv'
    file_boundary = 'boundary.csv'

    # load data
    data_elements = h.import_tartu_data(data_folder, file_line_elements)
    data_substation = h.import_tartu_trafo(data_folder, file_substation)

    # import Annelinn boundaries and split data at ','
    filename = os.path.join(data_folder, file_boundary)
    annelinn = pd.read_csv(filename, delimiter=',')

    # make a polygon from the boundary using eopands
    annelinn = gpd.GeoDataFrame(annelinn, geometry=gpd.points_from_xy(annelinn.p2, annelinn.p1))

    # set the coordinate sysstem to lat lon
    annelinn.crs = {'init': 'epsg:4326'}
    polygon_geom = Polygon(annelinn.geometry)

    lines_connected = h.connect_data(data_elements, 'MVLine')
    data_substation = h.combine_nodes(data_substation, tol=1e-4)
    lines_connected = h.connect_lines_and_substations(lines_connected, data_substation, tol=1e-4)
    nodes = h.identify_nodes(lines_connected, data_substation)

    connected_lines_cropped = h.crop_data(lines_connected, polygon_geom)
    connected_nodes_cropped = h.crop_data(nodes, polygon_geom)
    # reset index
    connected_nodes_cropped = connected_nodes_cropped.reset_index(drop=True)

    connected_nodes_cropped = h.add_slack_node(connected_nodes_cropped, connected_lines_cropped)

    # get line_objects
    line_objects = h.get_node_connections(connected_lines_cropped, connected_nodes_cropped, tol=2e-4)
    # remove all lines where a node is of type 'outside'
    line_objects = line_objects[~line_objects['node1_type'].str.contains('outside')]
    line_objects = line_objects[~line_objects['node2_type'].str.contains('outside')]

    # get node objects
    node1 = []
    node2 = []
    for line in line_objects.iterrows():
        line = line[1]
        node1.append(line['node1'])
        node2.append(line['node2'])
    # get unique nodes
    nodes = list(set(node1 + node2))
    nodes_geom = []
    for node in nodes:
        nodes_geom.append(connected_nodes_cropped.iloc[node]['geometry'])

    node_objects = gpd.GeoDataFrame(index=nodes)
    node_objects['node_type'] = 'node'
    node_objects['geometry'] = nodes_geom
    # define eastern_most node as slack node
    eastern_most = node_objects[node_objects['geometry'].x == max(node_objects['geometry'].x)].index[0]
    node_objects.loc[eastern_most, 'node_type'] = 'slack'
    node_objects['node_id'] = node_objects.index

    ### MANUAL CORRECTIONS ###

    lines_to_remove = [(34, 35), (24, 27), (19, 13), (19, 14), (27, 14), (7, 8)]

    for node1, node2 in lines_to_remove:
        line_objects = line_objects[
            ~(
                    ((line_objects['node1'] == node1) & (line_objects['node2'] == node2)) |
                    ((line_objects['node1'] == node2) & (line_objects['node2'] == node1))
            )
        ]

    # remove nodes 34 and 35
    node_objects = node_objects.drop([34, 35])
    # reset index
    node_objects = node_objects.reset_index(drop=True)

    print(node_objects.head(50))
    # print(line_objects.head(50))
    # save to pickle
    line_objects.to_pickle(os.path.join('pickle_files', 'line_objects.pkl'))
    node_objects.to_pickle(os.path.join('pickle_files', 'node_objects.pkl'))

    node_df_final = pd.DataFrame(columns=["i", "lat", "lon"])
    node_df_final["i"] = [f"Node_{i}" for i in node_objects.node_id]
    node_df_final["lat"] = node_objects.geometry.y
    node_df_final["lon"] = node_objects.geometry.x

    line_df_final = pd.DataFrame(columns=["i", "j"])
    line_df_final["i"] = [f"Node_{i}" for i in line_objects.node1]
    line_df_final["j"] = [f"Node_{i}" for i in line_objects.node2]

    demand_df = pd.DataFrame(columns=["i", "consumption"])
    demand_df["i"] = node_df_final["i"].values

    # save to csv
    node_df_final.to_csv(os.path.join('final_data', 'node_data.csv'), index=False)
    line_df_final.to_csv(os.path.join('final_data', 'line_data.csv'), index=False)


if __name__ == "__main__":
    main()
    # h.plot_network()
