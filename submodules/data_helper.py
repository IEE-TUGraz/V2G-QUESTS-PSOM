"""
This module contains helper functions to read, process, and manipulate data for various test cases and simulations. 
It includes utilities for graph analysis, behavior profile generation, data extraction, demand time series creation, 
and visualization. The functions are designed to work with power grid data, electric vehicle behavior profiles, 
and Pyomo models, among other use cases.

Key Features
------------
- Graph analysis for detecting branching levels in network structures.
- SQLite database integration for saving and extracting Pyomo model data.
- Demand time series generation for specific case studies.
- Visualization of power grid networks using Plotly.
- Loading and processing of bus, branch, and demand data for simulations.
Usage:
This module is designed to be imported and used as a utility toolkit in the V2G-QUESTS project.
"""

import math
import os
import sqlite3
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyomo.core.base.set
import pyomo.environ as pyo
from geopy import distance

from submodules.helper_functions import load_config

pd.set_option('display.max_columns', None)


def detect_branching_levels(Branches, root=0):
    """
    Analyze a graph structure to determine node levels, branching points,
    branch counts before each node, and counts of further nodes in each subtree.

    Parameters
    ----------
    Branches : list of dict
        Each dictionary represents an edge of the graph and should contain:
        - 'fbus': From bus ID
        - 'tbus': To bus ID
    root : int, optional
        The root node of the graph. Defaults to 0.

    Returns
    -------
    tuple
        - levels (dict): Maps each node to its level (distance from the root)
        - branch_points (list): Nodes that are branching points (more than one child)
        - branch_count_before (dict): Count of branch points before each node
        - further_nodes_count (dict): Count of nodes in each node's subtree (excluding itself)
    """

    # Build adjacency list for the graph
    edges = [(branch['fbus'], branch['tbus']) for branch in Branches]
    # Build adjacency list for the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # BFS to determine levels and branch points
    levels = {root: 0}
    branch_count_before = {root: 0}
    visited = set([root])
    queue = deque([root])
    branch_points = []

    while queue:
        node = queue.popleft()
        children = 0

        # Count children and track neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                levels[neighbor] = levels[node] + 1
                children += 1

        # Record branching points
        if children > 1:
            branch_points.append(node)

        # Update branch counts for each child
        for neighbor in graph[node]:
            if neighbor not in branch_count_before:
                branch_count_before[neighbor] = branch_count_before[node] + (1 if children > 1 else 0)

        further_nodes_count = {}

        def dfs(node, parent):
            subtree_size = 1  # Include the node itself
            for neighbor in graph[node]:
                if neighbor != parent:
                    subtree_size += dfs(neighbor, node)
            further_nodes_count[node] = subtree_size - 1  # Exclude the node itself
            return subtree_size

        dfs(root, None)
    return levels, branch_points, branch_count_before, further_nodes_count


def model_to_sqlite(model: pyo.base.Model, filename):
    """
    Save components of a Pyomo model to an SQLite database.

    Parameters
    ----------
    model : pyo.base.Model
        The Pyomo model containing components to save.
    filename : str
        Path to the SQLite database file. Creates the directory if it does not exist.

    Behavior
    --------
    - OrderedScalarSet: Saves set data as a DataFrame.
    - IndexedVar, IndexedParam, ScalarParam: Extracts values and saves with indices.
    - ScalarObjective: Saves the objective value as a single-row DataFrame.
    - ConstraintList: Skipped.
    - Unsupported component types: Logs a message and skips.

    Notes
    -----
    - Uses component names as table names in SQLite.
    - Replaces existing tables with the same name.

    Raises
    ------
    Exception
        Propagates any exceptions related to SQLite or Pyomo processing.

    Example
    -------
    ```python
    import pyomo.environ as pyo
    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2, 3], initialize=0)
    model_to_sqlite(model, "output/model_data.sqlite")
    ```
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    cnx = sqlite3.connect(filename)
    for o in model.component_objects():
        match type(o):
            case pyomo.core.base.set.OrderedScalarSet:
                df = pd.DataFrame(o.data())
            case pyomo.core.base.var.IndexedVar | pyomo.core.base.param.IndexedParam | pyomo.core.base.param.ScalarParam:
                indices = [str(i) for i in o.index_set().subsets()]
                df = pd.DataFrame(pd.Series(o.extract_values()), columns=['values'])
                if len(indices) == len(df.index.names):
                    if len(indices) > 1:
                        df = df.reset_index().rename(columns={f"level_{i}": b for i, b in enumerate(indices)})
                    else:
                        df = df.reset_index().rename(columns={"index": indices[0]})
                    df = df.set_index(indices)
                    print(f"Pyomo-Type {type(o)} with indices {indices} saved to SQLite")

            case pyomo.core.base.objective.ScalarObjective:
                df = pd.DataFrame([pyo.value(o)], columns=['values'])
                print(f"Pyomo-Type {type(o)} saved to SQLite")
            case pyomo.core.base.constraint.ConstraintList:  # Those will not be saved by decision
                continue
            case _:
                print(f"Pyomo-Type {type(o)} not implemented, {o.name} will not be saved to SQLite")
                continue
        df.to_sql(o.name, cnx, if_exists='replace')
        cnx.commit()
    cnx.close()
    pass


def extract_table_to_dataframe(con, table_name):
    """
    Extract the contents of a database table into a pandas DataFrame.

    Parameters
    ----------
    con : sqlite3.Connection
        Database connection object.
    table_name : str
        Name of the table to extract.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with the table's data if successful; None if an error occurs.

    Raises
    ------
    Exception
        Prints an error message if table extraction fails.
    """

    try:
        # Query the table
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, con)

        return df

    except Exception as e:
        print(f"Error extracting table {table_name}: {e}")
        return None


def safe_slice(data, start_t, num_t):
    if start_t + num_t > len(data):
        raise ValueError(f"Requested slice ({start_t}:{start_t + num_t}) exceeds data length {len(data)}")

    # For pandas DataFrame
    if hasattr(data, 'iloc'):
        return data.iloc[start_t:start_t + num_t, :].reset_index(drop=True)

    # For NumPy arrays or Series
    return data[start_t:start_t + num_t]


def build_demand_timeseries(case_study: str, num_t: int, start_t: int):
    """
    Construct node-level demand time series for a given case study.

    Supports three case studies: 'Kanaleneiland', 'Aradas', 'Annelinn'.

    Parameters
    ----------
    case_study : str
        Case study name:
        - 'Kanaleneiland': Uses node-specific yearly totals and normalized templates.
        - 'Aradas': Interpolates monthly profiles with typical daily shapes.
        - 'Annelinn': Uses a single base profile with random noise.
    num_t : int
        Number of time steps (hours) to generate.
    start_t : int
        Starting time step (hour offset).

    Returns
    -------
    pd.DataFrame
        Shape (num_t, num_nodes), each column is a node's hourly demand.

    Raises
    ------
    ValueError
        If an unsupported case study is provided.

    Notes
    -----
    - 'Kanaleneiland': Each node scales a normalized template by annual demand.
    - 'Aradas': Monthly-resolution demand scaled to match annual totals.
    - 'Annelinn': Base profile replicated across nodes with Gaussian noise.
    """

    if case_study == 'Kanaleneiland':
        path_to_load_profiles = os.path.join('data_preparation', 'demand_profiles')
        path_to_demand_data = os.path.join('data_preparation', 'Kanaleneiland', 'final_data')
        filename_demand = 'consumption_data_year.csv'
        demand_year_df = pd.read_csv(os.path.join(path_to_demand_data, filename_demand), sep=',')
        demand_hourly_df = pd.DataFrame(columns=demand_year_df["i"].values)

        for row in demand_year_df.itertuples(index=True):
            idx = row.Index
            node = row
            # load profile from pickle file
            # Each node gets a corresponding load profile with the same index. eg.: Node_1 gets demand_profile_1.pkl
            load_profile = pd.read_pickle(os.path.join(path_to_load_profiles, f"demand_profile_{idx}.pkl"))
            consumption_hourly = (load_profile["electricity"].values / np.sum(
                load_profile["electricity"].values) * node.consumption)/1000  # kWh to MWh
            demand_hourly_df[node.i] = safe_slice(consumption_hourly, start_t, num_t)

        # save demand data to csv file
        demand_hourly_df.to_csv(os.path.join(path_to_demand_data, 'demand_data.csv'), index=False)

    elif case_study == 'Aradas':

        path_to_demand_data = os.path.join('data_preparation', 'Aradas', 'final_data')
        filename_demand = 'demand_data_all.csv'
        raw_df = pd.read_csv(os.path.join(path_to_demand_data, filename_demand), sep=',')
        raw_df = raw_df.fillna(0)
        num_hours_year = 8760
        nodes = raw_df.columns
        interpolated_df = pd.DataFrame(index=range(num_hours_year), columns=nodes)

        # Normalize monthly data and interpolate to create hourly profiles for a whole year using standard load profiles
        for idx, node in enumerate(nodes):
            monthly_data = raw_df[node].values
            monthly_data_norm = monthly_data / np.mean(monthly_data)
            template_year = np.tile(monthly_data_norm, int(np.ceil(num_hours_year / len(monthly_data))))
            template_year = template_year[:num_hours_year]
            path_to_load_profiles = os.path.join('data_preparation', 'demand_profiles')
            profile_path = os.path.join(path_to_load_profiles, f"demand_profile_{idx}.pkl")
            yearly_profile = pd.read_pickle(profile_path)["electricity"].values
            yearly_profile_norm = yearly_profile / np.mean(yearly_profile)
            interpolated_profile = template_year * yearly_profile_norm
            target_sum = np.sum(monthly_data) * (8760 / len(monthly_data))
            interpolated_profile *= target_sum / np.sum(interpolated_profile)
            interpolated_df[node] = interpolated_profile

        # Scale to the expected annual demand
        total_annual_Mwh = 19447.200
        actual_total = interpolated_df.to_numpy().sum()
        scaling_factor = total_annual_Mwh / actual_total
        interpolated_df *= scaling_factor

        # Apply time slicing
        demand_hourly_df = safe_slice(interpolated_df, start_t, num_t)
        # Save output
        demand_hourly_df.to_csv(os.path.join(path_to_demand_data, 'demand_data.csv'), index=False)

    elif case_study == 'Annelinn':
        path_to_demand_data = os.path.join('data_preparation', 'Annelinn', 'raw_data')
        path_to_node_data = os.path.join('data_preparation', 'Annelinn', 'final_data')

        filename_demand = 'Moisavahe43_electricity_aggregated.xlsx'
        filename_nodes = 'node_data.csv'

        nodes_df = pd.read_csv(os.path.join(path_to_node_data, filename_nodes), sep=',')
        demand_profile_df = pd.read_excel(os.path.join(path_to_demand_data, filename_demand), header=0, nrows=8760)
        base_profile = demand_profile_df["Sum"].fillna(0)
        demand_hourly_df = pd.DataFrame(index=range(8760), columns=nodes_df["i"].values)

        for node in nodes_df["i"]:
            noise = np.random.normal(0, 0.05, size=8760)
            demand_hourly_df[node] = (base_profile.to_numpy() * (1 + noise)) /len(nodes_df["i"])

        # Apply time slicing
        demand_hourly_df = safe_slice(demand_hourly_df, start_t, num_t)

        # Save to CSV
        demand_hourly_df.to_csv(os.path.join(path_to_node_data, 'demand_data.csv'), index=False)


    else:
        raise ValueError(
            f"Unknown case study: {case_study}. Please choose from 'Kanaleneiland', 'Aradas', or 'Annelinn'.")
    return demand_hourly_df


def plot_input_only(buses_df, branches_df):
    """
    Plot a geographical visualization of buses and branches on a map.

    Parameters
    ----------
    buses_df : pd.DataFrame
        Bus information with columns:
        - 'id': Bus ID
        - 'lat': Latitude
        - 'lon': Longitude
    branches_df : pd.DataFrame
        Branch information with columns:
        - 'fbus': From bus ID
        - 'tbus': To bus ID

    Returns
    -------
    None

    Notes
    -----
    - Uses Plotly to display a Mercator-projected map scoped to Europe.
    - Dynamically adjusts geographical range based on bus lat/lon.
    """

    fig = go.Figure()

    # Add branches as lines
    for _, row in branches_df.iterrows():
        f_node = buses_df[buses_df['id'] == row['fbus']]
        t_node = buses_df[buses_df['id'] == row['tbus']]
        if not f_node.empty and not t_node.empty:
            fig.add_trace(go.Scattergeo(
                lon=[f_node.iloc[0]['lon'], t_node.iloc[0]['lon']],
                lat=[f_node.iloc[0]['lat'], t_node.iloc[0]['lat']],
                mode='lines',
                line=dict(width=4, color='black'),
                showlegend=False
            ))

    # Add nodes with text
    fig.add_trace(go.Scattergeo(
        lon=buses_df['lon'],
        lat=buses_df['lat'],
        mode='markers+text',
        text=buses_df['id'],  # show bus IDs
        textposition='top center',  # position above the markers
        marker=dict(size=10, color='#8C564B'),
        name='Buses'
    ))

    fig.update_layout(
        geo=dict(
            scope='europe',  # or 'usa' or your preferred zoom
            projection_type='mercator',
            showcountries=False,
            showland=True,
            landcolor='rgb(243, 243, 243)',
            lataxis=dict(range=[buses_df['lat'].min() - 0.01, buses_df['lat'].max() + 0.01]),
            lonaxis=dict(range=[buses_df['lon'].min() - 0.01, buses_df['lon'].max() + 0.01])
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.show()


def load_capacity_data(case_study: str, num_t: int, start_t: int, enable_pv: bool, enable_wind: bool,
                       enable_hydro: bool):
    """
    Load PV, wind, and hydro capacity data for a given case study.

    Parameters
    ----------
    case_study : str
        Case study name ('Kanaleneiland', 'Aradas', or 'Annelinn').
    num_t : int
        Number of time steps to load.
    start_t : int
        Starting time index.
    enable_pv : bool, optional
        Enable PV generation. Defaults to True.
    enable_wind : bool, optional
        Enable wind generation. Defaults to True.
    enable_hydro : bool, optional
        Enable hydro generation. Defaults to True.

    Returns
    -------
    tuple of pandas.DataFrame
        (pv_df, wind_df, hydro_df) with capacity time series in column 'electricity'.

    Raises
    ------
    ValueError
        If the case study is not recognized.

    Notes
    -----
    - Data is loaded from `data_preparation/<case_study>/final_data/`.
    - PV and wind files skip 3 header rows; hydro does not.
    - Only the 'electricity' column is used.
    """

    # Load configuration file
    capacity_data_pv_df = pd.DataFrame()
    capacity_data_wind_df = pd.DataFrame()
    capacity_data_hydro_df = pd.DataFrame()

    # Define the filepath based for the case study
    base_path = os.path.join('data_preparation', case_study, 'final_data')
    if case_study in ['Kanaleneiland', 'Aradas', 'Annelinn']:
        capacity_pv_filepath = os.path.join(base_path, 'capacity_data_pv.csv')
        capacity_wind_filepath = os.path.join(base_path, 'capacity_data_wind.csv')
        capacity_hydro_filepath = os.path.join(base_path, 'capacity_data_hydro.csv')
    else:
        raise ValueError(
            f"Unknown case study: {case_study}. Please choose from 'Kanaleneiland', 'Aradas', or 'Annelinn'.")

    # Load the data
    capacity_data_pv_df = pd.read_csv(capacity_pv_filepath, sep=',', skiprows=3)
    capacity_data_pv_df = safe_slice(capacity_data_pv_df, start_t, num_t)[["electricity"]]
    if not enable_pv:
        capacity_data_pv_df["electricity"] = 0

    # Wind
    capacity_wind_data_df = pd.read_csv(capacity_wind_filepath, sep=',', skiprows=3)
    capacity_data_wind_df = safe_slice(capacity_wind_data_df, start_t, num_t)[["electricity"]]
    if not enable_wind:
        capacity_data_wind_df["electricity"] = 0

    # Hydro
    capacity_hydro_data_df = pd.read_csv(capacity_hydro_filepath, sep=',')
    capacity_data_hydro_df = safe_slice(capacity_hydro_data_df, start_t, num_t)[["electricity"]]
    if not enable_hydro:
        capacity_data_hydro_df["electricity"] = 0

    return capacity_data_pv_df, capacity_data_wind_df, capacity_data_hydro_df


def load_price_data(case_study: str, num_t: int, start_t: int, year: int):
    """
    Load electricity price time series for a given case study and year.

    Parameters
    ----------
    case_study : str
        'Kanaleneiland', 'Aradas', or 'Annelinn'.
    num_t : int
        Number of time steps to load.
    start_t : int
        Starting time index.
    year : int
        Year of the price data (e.g., 2035, 2040, 2050).

    Returns
    -------
    pandas.DataFrame
        Price time series for the selected case study and year.

    Raises
    ------
    ValueError
        If the case study or year is not supported.
    """
    if case_study == 'Kanaleneiland':
        if year in [2035, 2040, 2050]:
            price_filepath = os.path.join('data_preparation', 'Kanaleneiland', 'final_data', 'price_time_series.xlsx')
        else:
            raise ValueError(f"Unsupported year: {year}. Please choose from 2035, 2040, or 2050.")
    elif case_study == 'Aradas':
        if year in [2035, 2040, 2050]:
            price_filepath = os.path.join('data_preparation', 'Aradas', 'final_data', 'price_time_series.xlsx')
        else:
            raise ValueError(f"Unsupported year: {year}. Please choose from 2035, 2040, or 2050.")
    elif case_study == 'Annelinn':
        if year in [2035, 2040, 2050]:
            price_filepath = os.path.join('data_preparation', 'Annelinn', 'final_data', 'price_time_series.xlsx')
        else:
            raise ValueError(f"Unsupported year: {year}. Please choose from 2035, 2040, or 2050.")
    else:
        raise ValueError(
            f"Unknown case study: {case_study}. Please choose from 'Kanaleneiland', 'Aradas', or 'Annelinn'.")

    price_data_df = pd.read_excel(price_filepath, sheet_name=str(year))

    price_data_df = safe_slice(price_data_df, start_t, num_t)
    return price_data_df


def create_availability_matrix(evs: pd.DataFrame, Buses: pd.DataFrame, num_t: int):
    """
    Create a 3D availability matrix of EVs across nodes and time.

    Parameters
    ----------
    evs : pandas.DataFrame
        EV data with columns ['vehicle_name', 'node', 'arrival_time [h]', 'departure_time [h]'].
    Buses : pandas.DataFrame
        Bus data with column 'id' representing node identifiers.
    num_t : int
        Number of time steps in the simulation horizon.

    Returns
    -------
    tuple
        - vehicles : list of EV names
        - availability_matrix : numpy array of shape (n_vehicles, n_nodes, num_t),
          1 indicates EV presence at a node, 0 otherwise.

    Notes
    -----
    - If the time horizon exceeds the available EV data, the pattern may be repeated.
    """
    # Unique vehicles and nodes
    vehicles = sorted(evs['vehicle_name'].unique())
    nodes = sorted(Buses['id'].unique())

    # Mapping dicts
    vehicle_to_idx = {v: i for i, v in enumerate(vehicles)}
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Find maximum departure time in data
    max_time = int(evs['departure_time [h]'].max())

    # If horizon is longer than dataset
    repeat = False
    if num_t > max_time:
        answer = input(
            f"⚠️ EV behavior data is only available for {max_time} hours but the chosen time interval covers {num_t} hours.\n"
            "Do you want to repeat the available EV data to fill the rest? (y/n): "
        )
        if answer.strip().lower() in ("y", "yes"):
            repeat = True

    # Initialize matrix
    availability_matrix = np.zeros((len(vehicles), len(nodes), num_t), dtype=int)

    # Fill availability
    for _, row in evs.iterrows():
        v = vehicle_to_idx[row['vehicle_name']]
        n = node_to_idx[row['node']]
        start = int(row['arrival_time [h]'])
        end = int(row['departure_time [h]'])
        duration = end - start

        if repeat:
            # Repeat pattern across horizon
            for offset in range(0, num_t, max_time):
                s = (start + offset) % num_t
                e = (s + duration) % num_t
                if s < e:
                    availability_matrix[v, n, s:e] = 1
                else:  # wrap-around
                    availability_matrix[v, n, s:] = 1
                    availability_matrix[v, n, :e] = 1
        else:
            # Just fill once, no repetition
            if start < num_t:  # avoid going beyond horizon
                availability_matrix[v, n, start:min(end, num_t)] = 1

    def availability_to_dataframe(availability_matrix, vehicle_to_idx, node_to_idx, num_t):
        idx_to_vehicle = {i: v for v, i in vehicle_to_idx.items()}
        idx_to_node = {i: n for n, i in node_to_idx.items()}

        records = []
        for v in range(len(vehicle_to_idx)):
            for n in range(len(node_to_idx)):
                for t in range(num_t):
                    if availability_matrix[v, n, t] == 1:
                        records.append({
                            "vehicle": idx_to_vehicle[v],
                            "node": idx_to_node[n],
                            "time": t,
                            "available": 1
                        })
        return pd.DataFrame(records)

    availability_df = availability_to_dataframe(availability_matrix, vehicle_to_idx, node_to_idx, num_t)

    return availability_matrix, availability_df


def load_ev_data(case_study: str):
    """
    Load aggregated electric vehicle (EV) data for a given case study.

    Parameters
    ----------
    case_study : str
        Name of the case study ('Kanaleneiland', 'Aradas', 'Annelinn').

    Returns
    -------
    pandas.DataFrame
        DataFrame containing EV trip and availability data from the
        'adjusted_intervals_hrs' sheet.

    Raises
    ------
    ValueError
        If the case study name is not recognized.
    """
    if case_study == 'Kanaleneiland':
        ev_filepath = os.path.join('data_preparation', 'Kanaleneiland', 'final_data',
                                   'Kanaleneiland_vehicles_aggregated.xlsx')
    elif case_study == 'Aradas':
        ev_filepath = os.path.join('data_preparation', 'Aradas', 'final_data', 'Aradas_vehicles_aggregated.xlsx')
    elif case_study == 'Annelinn':
        ev_filepath = os.path.join('data_preparation', 'Annelinn', 'final_data', 'Annelinn_vehicles_aggregated.xlsx')
    else:
        raise ValueError(
            f"Unknown case study: {case_study}. Please choose from 'Kanaleneiland', 'Aradas', or 'Annelinn'.")
    ev_data_df = pd.read_excel(ev_filepath, sheet_name='adjusted_intervals_hrs', header=0)

    return ev_data_df


def load_ev_battery_data():
    """
   Load electric vehicle (EV) battery parameters for each archetype.

   Returns
   -------
   pandas.DataFrame
       DataFrame containing EV battery parameters, including:
       - Charge efficiency
       - Discharge efficiency
       - Initial state of charge (SOC)
       - Battery capacity
       - Maximum charging power
       - Maximum discharging power
   """
    ev_battery_filepath = os.path.join('data_preparation', 'standard_ev_battery_data.xlsx')
    ev_battery_data_df = pd.read_excel(ev_battery_filepath, header=0)
    return ev_battery_data_df


def load_generator_data(case_study: str, year: int):
    """
    Load generator data for a given case study and scenario year.

    Parameters
    ----------
    case_study : str
        Name of the case study ('Kanaleneiland', 'Aradas', or 'Annelinn').
    year : int
        Scenario year (e.g., 2035, 2040, 2050).

    Returns
    -------
    pandas.DataFrame
        Cleaned generator dataset containing capacity and operational parameters.

    Raises
    ------
    ValueError
        If the case study or year is not supported.
    """
    if case_study == 'Kanaleneiland':
        if year in [2035, 2040, 2050]:
            generator_pv_filepath = os.path.join('data_preparation', 'Kanaleneiland', 'final_data',
                                                 f'generators_{year}.csv')
        else:
            raise ValueError(f"Unsupported year: {year}. Please choose from 2035, 2040, or 2050.")
    elif case_study == 'Aradas':
        if year in [2035, 2040, 2050]:
            generator_pv_filepath = os.path.join('data_preparation', 'Aradas', 'final_data', f'generators_{year}.csv')
        else:
            raise ValueError(f"Unsupported year: {year}. Please choose from 2035, 2040, or 2050.")
    elif case_study == 'Annelinn':
        if year in [2035, 2040, 2050]:
            generator_pv_filepath = os.path.join('data_preparation', 'Annelinn', 'final_data', f'generators_{year}.csv')
        else:
            raise ValueError(f"Unsupported year: {year}. Please choose from 2035, 2040, or 2050.")
    else:
        raise ValueError(
            f"Unknown case study: {case_study}. Please choose from 'Kanaleneiland', 'Aradas', or 'Annelinn'.")

    generator_data_df = pd.read_csv(generator_pv_filepath, sep=';')
    generator_data_df = generator_data_df.dropna()
    return generator_data_df


def load_bus_branch_demand_data(case_study: str, num_t: int, start_t: int, slack_bus: int):
    """
    Load, preprocess, and combine network topology and demand data.

    This function loads node, line, and demand data for the specified case study,
    assigns electrical parameters to buses, computes line impedances, admittances,
    and limits, and identifies branches connected to the slack bus.

    Parameters
    ----------
    case_study : str
        Name of the case study ('Kanaleneiland', 'Aradas', or 'Annelinn').
    num_t : int
        Number of time steps to extract from the demand time series.
    start_t : int
        Starting index for the time series slice.
    slack_bus : str
        ID or name of the slack/reference bus.

    Returns
    -------
    tuple of pandas.DataFrame
        (
            Buses_df,      # Bus-level electrical and geographic data
            Branches_df,   # Line parameters (admittance, limits, angle bounds)
            demand_p_df,   # Active power demand time series
            demand_q_df,   # Reactive power demand time series
            node_data      # Raw node metadata (coordinates, IDs)
        )

    Processing Steps
    ----------------
    1. Load standard line parameters and case-specific node/line data.
    2. Assign geographic coordinates (lat/lon) to each node.
    3. Build `Buses_df` with voltage and angle limits (radians).
    4. Construct time-dependent demand data; missing nodes receive zero demand.
    5. Identify branches connected to the slack bus.
    6. Compute line impedance, admittance, and capacity values.
    7. Return fully processed DataFrames for model integration.

    Raises
    ------
    ValueError
        If the case study is not recognized or required columns are missing.
    """
    line_parameters_filepath = r"data_preparation\standard_line_parameters.xlsx"
    parameter_filepath = os.path.join('case_studies', 'standard_parameter.yml')
    parameter = load_config(parameter_filepath)
    standard_line_parameters_df = pd.read_excel(line_parameters_filepath, sheet_name='line_parameters')
    standard_line_parameters_df.columns = standard_line_parameters_df.columns.str.strip()

    if case_study == 'Kanaleneiland':
        line_data_filepath = os.path.join('data_preparation', 'Kanaleneiland', 'final_data', 'line_data.csv')
        node_data_filepath = os.path.join('data_preparation', 'Kanaleneiland', 'final_data', 'node_data.csv')
        slack_node = slack_bus
    elif case_study == 'Aradas':
        line_data_filepath = os.path.join('data_preparation', 'Aradas', 'final_data', 'line_data.csv')
        node_data_filepath = os.path.join('data_preparation', 'Aradas', 'final_data', 'node_data.csv')
        slack_node = slack_bus
    elif case_study == 'Annelinn':
        line_data_filepath = os.path.join('data_preparation', 'Annelinn', 'final_data', 'line_data.csv')
        node_data_filepath = os.path.join('data_preparation', 'Annelinn', 'final_data', 'node_data.csv')
        slack_node = slack_bus
    else:
        raise ValueError(
            f"Unknown case study: {case_study}. Please choose from 'Kanaleneiland', 'Aradas', or 'Annelinn'.")

    line_data = pd.read_csv(line_data_filepath, sep=',')
    node_data = pd.read_csv(node_data_filepath, sep=',')
    demand_data_df = build_demand_timeseries(case_study, num_t, start_t)
    if not {'i', 'j'}.issubset(line_data.columns):
        raise ValueError("line_data must contain 'i' and 'j' columns representing line endpoints")

    # Rename for clarity
    node_data = node_data.rename(columns={'i': 'node'})
    # Prepare node lookup
    coord_lookup = node_data.set_index('node')[['lat', 'lon']].to_dict('index')
    # Node parameters
    Gs = parameter["Gs"]
    Bs = parameter["Bs"]
    Vmax = parameter["Vmax"]
    Vmin = parameter["Vmin"]
    pMin_Ang = parameter["Min_Ang"]
    pMax_Ang = parameter["Max_Ang"]
    pSBase = parameter["SBase"]

    # Collect bus data as a list of dicts
    bus_rows = []

    for node in node_data['node'].values:
        lat = coord_lookup[node]['lat']
        lon = coord_lookup[node]['lon']
        bus_rows.append({
            'id': node,
            'Gs': Gs,
            'Bs': Bs,
            'Vmax': Vmax,
            'Vmin': Vmin,
            'lat': lat,
            'lon': lon,
            'pMin_Ang': math.radians(pMin_Ang),
            'pMax_Ang': math.radians(pMax_Ang)
        })

    # Convert to DataFrame
    Buses_df = pd.DataFrame(bus_rows, columns=['id', 'Gs', 'Bs', 'Vmax', 'Vmin', 'lat', 'lon', 'pMin_Ang', 'pMax_Ang'])

    # Demand Data
    # iterate through all nodes, check if a column exists in demand_data_df, if not, set values to 0
    for node in node_data['node'].values:
        if node not in demand_data_df.columns:
            demand_data_df[node] = 0
            print(f"Node {node} not in demand data, setting values to 0")
    demand_p_df = demand_data_df.copy()
    demand_q_df = demand_data_df.copy()
    # Line Data
    # Merge coordinates for node i
    line_data = line_data.merge(node_data, left_on='i', right_on='node').rename(
        columns={'lat': 'lat_i', 'lon': 'lon_i'}).drop(columns='node')

    # Merge coordinates for node j
    line_data = line_data.merge(node_data, left_on='j', right_on='node').rename(
        columns={'lat': 'lat_j', 'lon': 'lon_j'}).drop(columns='node')

    # Function to compute distance using geopy
    def compute_distance(row):
        coord_i = coord_lookup.get(row['i'])
        coord_j = coord_lookup.get(row['j'])
        if coord_i and coord_j:
            dist = distance.distance((coord_i['lat'], coord_i['lon']), (coord_j['lat'], coord_j['lon'])).kilometers
            return max(dist, 0.05)  # Ensure minimum distance of 0.05 km
        else:
            return 0.05

    standard_line_data = line_data.copy()
    standard_line_data['connected_to_slack'] = (standard_line_data['i'] == slack_node) | (
            standard_line_data['j'] == slack_node)
    fixed_line_types = {}

    if case_study == 'Kanaleneiland':
        # Set slack node parameters
        slack_node_params = standard_line_parameters_df[standard_line_parameters_df['Type'] == 535].iloc[0]
        # Specify certain line and their corresponding types
        # fixed_line_types = {frozenset(['Node_15', 'Node_30']): 185,
        #                     frozenset(['Node_15', 'Node_11']): 185,}
        # Define types for the remaining lines (excluding slack node and fixed line) From the standard parameters excluding! the given types. Those get randomly assigned
        other_types = standard_line_parameters_df[~standard_line_parameters_df['Type'].isin([535, 300, 240, 185, 150, 120, 35, 25])]
    elif case_study == 'Aradas':
        slack_node_params = standard_line_parameters_df[standard_line_parameters_df['Type'] == 240].iloc[0]
        other_types = standard_line_parameters_df[~standard_line_parameters_df['Type'].isin([535, 300, 240, 185])]
    elif case_study == 'Annelinn':
        slack_node_params = standard_line_parameters_df[standard_line_parameters_df['Type'] == 240].iloc[0]
        other_types = standard_line_parameters_df[~standard_line_parameters_df['Type'].isin([535, 300, 240, 185])]

    standard_line_data['fixed_type'] = standard_line_data.apply(
        lambda r: fixed_line_types.get(frozenset([r['i'], r['j']])),
        axis=1)

    params_by_type = standard_line_parameters_df.set_index('Type')

    def assign_params(row):
        # 1) Slack-connected lines
        if row['connected_to_slack']:
            return pd.Series(slack_node_params)
        # 2) Explicitly specified lines
        if pd.notna(row['fixed_type']):
            return pd.Series(params_by_type.loc[row['fixed_type']])
        # 3) All remaining lines → random
        return other_types.sample(1).iloc[0]

    assigned_params = standard_line_data.apply(assign_params, axis=1)
    final_data = pd.concat([standard_line_data.drop(columns=['connected_to_slack']), assigned_params], axis=1)

    line_data['length_km'] = line_data.apply(compute_distance, axis=1)
    # Electrical parameters
    line_data['pRLine'] = (final_data['r'] / pSBase) * line_data['length_km']
    line_data['pXLine'] = (final_data['x'] / pSBase) * line_data['length_km']
    line_data['pSmax_line'] = final_data['Smax']
    line_data['pPmax_line'] = 0.9 * final_data['Smax']
    line_data['pMin_AngDiff'] = math.radians(-parameter['Max_AngDiff'])
    line_data['pMax_AngDiff'] = math.radians(parameter['Max_AngDiff'])
    # Calculate pGLine and pBLine
    z_squared = line_data['pRLine'] ** 2 + line_data['pXLine'] ** 2
    line_data['pGLine'] = line_data['pRLine'] / z_squared
    line_data['pBLine'] = -line_data['pXLine'] / z_squared

    # Final Branches DataFrame
    Branches_df = line_data.rename(columns={
        'i': 'fbus',
        'j': 'tbus'
    })[['fbus', 'tbus', 'pGLine', 'pBLine', 'pSmax_line', 'pPmax_line', 'pMin_AngDiff', 'pMax_AngDiff',
        'pXLine']].reset_index(drop=True)
    return Buses_df, Branches_df, demand_p_df, demand_q_df, node_data


def distribute_storage_capacity(case_study, pBatt_cap, slack_bus, node_ids, zero_capacity_nodes=None, seed=None):
    """
    Randomly distribute total storage capacity among network nodes.

    Parameters
    ----------
    case_study : str
        Name of the case study ('Kanaleneiland', 'Aradas', or 'Annelinn').
    pBatt_cap : float
        Total storage capacity to distribute (MW).
    node_ids : list
        List of node identifiers.
    zero_capacity_nodes : list, optional
        Nodes to exclude from allocation. Defaults to None.
        Slack bus is automatically excluded. Example: [slack_bus, 'Node_67']
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame mapping each node to its assigned storage capacity.
    """
    if zero_capacity_nodes is None:
        zero_capacity_nodes = [slack_bus]

    if seed is not None:
        np.random.seed(seed)

    zero_capacity_nodes = set(zero_capacity_nodes)
    node_ids = list(node_ids)
    non_zero_nodes = [n for n in node_ids if n not in zero_capacity_nodes]

    if len(non_zero_nodes) == 0:
        raise ValueError("No nodes available to assign storage capacity.")

    # Generate random weights and normalize to total_capacity
    weights = np.random.rand(len(non_zero_nodes))
    weights /= weights.sum()
    capacities = weights * float(pBatt_cap)

    # Assemble full list with capacities at correct indices
    node_bat_capacities = {n: 0.0 for n in node_ids}
    for i, n in enumerate(non_zero_nodes):
        node_bat_capacities[n] = capacities[i]
    node_bat_capacities = pd.DataFrame.from_dict(node_bat_capacities, orient='index', columns=['capacity'])
    for n in zero_capacity_nodes:
        node_bat_capacities[n] = 0.0

    return node_bat_capacities

def load_distributed_battery_data():
    """
    Load standard distributed battery parameters from configuration.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame containing:
        - Battery efficiency
        - Depth of discharge
        - C-rate
        - Initial state of charge
    """
    parameter_filepath = os.path.join('case_studies', 'standard_parameter.yml')
    parameter = load_config(parameter_filepath)

    battery_data_df = pd.DataFrame([{
        'eta_ch_batt': parameter['eta_ch_batt'],
        'eta_dch_batt': parameter['eta_dch_batt'],
        'SoC_init_batt': parameter['SoC_init_batt'],
        'DoD_batt': parameter['DoD_batt'],
        'C_rate_batt': parameter['C_rate_batt'],
    }])

    return battery_data_df

