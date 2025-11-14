"""
system_data.py
===============


Utilities for loading and preparing system input data used by the V2G model.


This module provides the `SystemData` class which centralizes all data-loading
and preprocessing steps required to build and populate a Pyomo model of a
distribution network with distributed storage, EVs and renewable generators.


The class wraps calls to `data_helper` (aliased as `dh`) and stores the
results as pandas DataFrames or numpy arrays on the instance for later use.


Usage
-----
Create a SystemData instance and call ``get_data()`` (or individual getters)
before building the model. Example::


sd = SystemData(num_t=24, start_t=0, case_study='Annelinn', enable_pv=True,
enable_wind=True, enable_hydro=False, year=2024,
pBatt_cap=100, pCharge_Invest=50)
sd.get_data()
"""
import numpy as np
import pandas as pd

from submodules import data_helper as dh

np.random.seed(66) # Random seed for reproducibility


class SystemData:
    """
    Container and helper for system input data used in the V2G model.
    
    
    The class collects raw inputs (buses, branches, generators, EV traces,
    price data, capacity data) and exposes them as pandas DataFrames and
    lightweight numpy objects. Each `get_*` method calls a corresponding
    `data_helper` function and keeps the results as attributes for later use.
    
    
    Parameters
    ----------
    num_t : int
    Number of time steps in the simulation horizon.
    start_t : int
    Index of the first time step (commonly 0).
    case_study : str
    Case study identifier used by the data helper loaders.
    enable_pv : bool
    If True, load PV capacity/time-series when available.
    enable_wind : bool
    If True, load wind capacity/time-series when available.
    enable_hydro : bool
    If True, load hydro capacity/time-series when available.
    year : int
    Year used to select seasonal / historical data.
    pBatt_cap : float
    Total battery capacity used for distributing storage across buses.
    pCharge_Invest : float
    Upper bound used when building charging investment constraints.
    availability_matrix : optional
    Optional precalculated availability matrix. If None the matrix is
    created by :meth:`get_availability_matrix` which calls
    ``dh.create_availability_matrix``.
    
    
    Attributes
    ----------
    net : optional
    Placeholder for network object returned by external loaders.
    buses_df : pd.DataFrame
    Table with bus metadata (id, geometry, attributes).
    branches_df : pd.DataFrame
    Table with line/branch metadata.
    generators_df : pd.DataFrame
    Generator table (bus, type, Pmax, Pmin, Qmax, Qmin).
    import_price_df : pd.DataFrame
    Time series of import prices (column 'price').
    evs_df : pd.DataFrame
    Tabular EV trace data enriched with battery specs and sampled variants.
    ev_battery_data : pd.DataFrame
    Lookup table with EV battery parameters by archetype/variant.
    availability_matrix : numpy.ndarray
    Binary availability matrix produced by the helper function.
    """

    def __init__(self, num_t: int, start_t: int, case_study: str, enable_pv: bool, enable_wind: bool,
                 enable_hydro: bool, year: int, pBatt_cap: int, pCharge_Invest: int, slack_bus: int):
        """
                Initializes the SystemData object with configuration parameters and placeholders
                for all relevant datasets used in the case study.

                Parameters
                ----------
                num_t : int
                    Number of time steps in the simulation (e.g., 8760 for hourly data over one year).
                start_t : int
                    Index of the starting time step (used to offset time-series data).
                case_study : str
                    Identifier or name of the case study (e.g., 'Kanaleneiland', 'Aradas').
                enable_pv : bool
                    Whether photovoltaic (PV) generation is included in the system data.
                enable_wind : bool
                    Whether wind generation is included in the system data.
                enable_hydro : bool
                    Whether hydro generation is included in the system data.
                year : int
                    Reference year for which data should be loaded (used for time-dependent datasets).
                pBatt_cap : int
                    Total available battery storage capacity [MWh] to be distributed across buses.
                pCharge_Invest : int
                    Total investment or installed charging capacity [MW] for charging infrastructure.

                Attributes
                ----------
                case_study : str
                    Name of the case study used to locate and load data.
                num_t : int
                    Number of simulation time steps.
                start_t : int
                    Start index for time-series slicing.
                enable_pv : bool
                    Flag for enabling PV data loading.
                enable_wind : bool
                    Flag for enabling wind data loading.
                enable_hydro : bool
                    Flag for enabling hydro data loading.
                year : int
                    Year used to select input datasets.
                pBatt_cap : int
                    Total available battery capacity [MWh].
                pCharge_Invest : int
                    Total installed charging capacity [MW].
                net : None or object
                    Placeholder for network model data (e.g., pandapower net).
                buses_df : pd.DataFrame or None
                    DataFrame containing bus-level information.
                branches_df : pd.DataFrame or None
                    DataFrame containing branch (line) parameters.
                generators_df : pd.DataFrame or None
                    DataFrame containing generator parameters (type, Pmax, Qmax, etc.).
                imports_df : pd.DataFrame or None
                    DataFrame for external imports (if applicable).
                import_price_df : pd.DataFrame or None
                    Time-series DataFrame containing import prices [€/MWh].
                evs_df : pd.DataFrame or None
                    DataFrame containing aggregated electric vehicle information.
                capacity_data_pv_df : pd.DataFrame or None
                    PV generation capacity time series for each bus and time step.
                bus_batt_cap_df : pd.DataFrame or None
                    DataFrame containing distributed battery capacities by bus.
                line_parameters_df : pd.DataFrame or None
                    DataFrame storing electrical line parameters (R, X, thermal limits, etc.).
                availability_matrix : np.ndarray or None
                    Matrix describing EV availability across buses and time steps.

                Returns
                -------
                None
                """
        self.case_study = case_study
        self.num_t = num_t
        self.start_t = start_t
        self.enable_pv = enable_pv
        self.pBatt_cap = pBatt_cap
        self.pCharge_Invest = pCharge_Invest
        self.slack_bus = slack_bus
        self.enable_wind = enable_wind
        self.enable_hydro = enable_hydro
        self.year = year

        self.net = None
        self.buses_df = None
        self.branches_df = None
        self.generators_df = None
        self.imports_df = None
        self.import_price_df = None
        self.evs_df = None
        self.capacity_data_pv_df = None
        self.bus_batt_cap_df = None
        self.line_parameters_df = None
        self.availability_matrix = None

    def get_generator_data(self):
        """
        Load generator metadata and return a cleaned DataFrame.


        The returned ``generators_df`` contains at least the following columns:
        ``['bus', 'type', 'Pmax', 'Pmin', 'Qmax', 'Qmin']``. The ``type``
        column is normalized to lowercase to make downstream filtering robust.


        Returns
        -------
        pd.DataFrame
        The cleaned generator table assigned to ``self.generators_df``.
        """
        self.generator_data_df = dh.load_generator_data(self.case_study, self.year)
        generators_data_df = self.generator_data_df  # Define generators_data_df
        self.generators_df = generators_data_df[['bus', 'type', 'Qmax', 'Qmin', 'Pmax', 'Pmin']]

        return self.generators_df

    def get_price_data(self):
        """
        Load import price (market price) time series for the configured case study.


        The function stores a DataFrame in ``self.import_price_df`` with a single
        column named ``price`` and index corresponding to the simulation time
        steps.
        """
        self.import_price_df = dh.load_price_data(self.case_study, self.num_t, self.start_t, self.year)
        self.import_price_df.rename(columns={'Price data': 'price'}, inplace=True)

    def get_ev_data(self, random_state: int | None = None):
        """
        Load and enrich EV trace data.


        The method performs the following steps:
        - Load raw EV traces and a battery-spec lookup via ``dh``.
        - Compute pre-visit EV energy demand from cumulative ETC values.
        - Assign a sampled vehicle "variant" per vehicle (stochastic mapping)
        using pre-defined archetype probabilities.
        - Merge battery characteristics from the lookup into the trace table.


        Parameters
        ----------
        random_state : int | None, optional
        Seed for reproducible variant sampling. If ``None`` the global
        numpy RNG state is used.


        Returns
        -------
        pd.DataFrame
        Enriched EV trace table stored in ``self.evs_df``.
        """
        self.evs_df = dh.load_ev_data(self.case_study)
        self.ev_battery_data = dh.load_ev_battery_data()
        # Copy & sort
        self.evs_df = self.evs_df.copy()
        self.evs_df = self.evs_df.sort_values(['agent', 'out[h]'])

        # Compute demand from cumulative ETC
        group_col = 'agent' if 'agent' in self.evs_df.columns else (
            'vehicle_name' if 'vehicle_name' in self.evs_df.columns else 'vehicle_id')
        dep_col = 'departure_time [h]' if 'departure_time [h]' in self.evs_df.columns else (
            'out[h]' if 'out[h]' in self.evs_df.columns else 'departure')
        df_sorted = self.evs_df.sort_values([group_col, dep_col]).copy()
        df_sorted['EV_demand [MWh]'] = (
                df_sorted.groupby(group_col)['ETC [MWh]'].shift(-1) - df_sorted['ETC [MWh]']).fillna(0).clip(
            lower=0)

        # Copy results back into self.evs_df while keeping its original row order
        self.evs_df.loc[df_sorted.index, 'EV_demand [MWh]'] = df_sorted['EV_demand [MWh]']

        # Rename columns
        self.evs_df.rename(columns={
            'agent': 'vehicle_name',
            'archetype': 'type',
            'osm_id': 'vehicle_id',
            'in[h]': 'arrival_time [h]',
            'out[h]': 'departure_time [h]'
        }, inplace=True)

        # Define probabilities per group
        ev_probs = {
            'EV1': 0.10,
            'EV2': 0.20,
            'EV3': 0.20,
            'EV4': 0.25,
            'EV5': 0.25
        }
        v2g_probs = {
            'V2G1': 0.10,
            'V2G2': 0.20,
            'V2G3': 0.20,
            'V2G4': 0.25,
            'V2G5': 0.25
        }

        # Helper function to sample variant based on type
        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        def sample_variant_for_type(ev_type):
            if ev_type == "EV":
                return rng.choice(list(ev_probs.keys()), p=list(ev_probs.values()))
            elif ev_type == "V2G":
                return rng.choice(list(v2g_probs.keys()), p=list(v2g_probs.values()))
            else:
                return None

        # Choose the column that identifies a vehicle (prefer vehicle_name then vehicle_id)
        vehicle_id_col = 'vehicle_name' if 'vehicle_name' in self.evs_df.columns else (
            'vehicle_id' if 'vehicle_id' in self.evs_df.columns else group_col)

        # Build a mapping vehicle -> variant (based on that vehicle's type)
        variant_map = {}
        for veh in self.evs_df[vehicle_id_col].unique():
            # pick a representative type for the vehicle (most common / first)
            types = self.evs_df.loc[self.evs_df[vehicle_id_col] == veh, 'type']
            if types.empty:
                chosen = None
            else:
                ev_type = types.mode().iat[0] if not types.mode().empty else types.iloc[0]
                chosen = sample_variant_for_type(ev_type)
            variant_map[veh] = chosen

        # Map the variant back to every row for that vehicle
        self.evs_df['variant'] = self.evs_df[vehicle_id_col].map(variant_map)

        # Merge battery specs (SOC_max, ch, dch) from lookup
        self.evs_df = self.evs_df.merge(
            self.ev_battery_data[['variant', 'SOC_max', 'ch', 'dch', 'eta_ch_EV', 'eta_dch_EV', 'SoC_init_EV']],
            on='variant',
            how='left'
        )
        self.evs_df.rename(columns={'SOC_max': 'EV_SOC_max [MWh]',
                                         'ch': 'Ev_ch_max [MW]',
                                         'dch': 'Ev_dch_max [MW]'}, inplace=True)

        return self.evs_df

    def get_availability_matrix(self):
        """
        Generates and stores the vehicle availability matrix for the simulation.

        This method uses electric vehicle (EV) data and bus data to compute the availability
        of each EV at each bus for all time steps. The result is stored as a matrix indicating
        which EVs are available for charging/discharging at specific times and locations.

        The method internally calls :func:`dh.create_availability_matrix` and stores
        the second returned element (the actual availability matrix) in `self.availability_matrix`.

        Returns
        -------
        None
        """
        self.availability_matrix = dh.create_availability_matrix(self.evs_df, self.buses_df, self.num_t)[1]

    def get_capacity_data(self):
        """
        Loads and stores time-series generation capacity data for different renewable sources.

        This method retrieves PV, wind, and hydro generation capacity profiles for the given
        case study and time period using the :func:`dh.load_capacity_data` function.
        The retrieved data is stored in the attributes:
        - `self.capacity_data_pv_df`
        - `self.capacity_data_wind_df`
        - `self.capacity_data_hydro_df`

        These DataFrames contain generation availability for each node and time step.

        Returns
        -------
        None
        """
        self.capacity_data_pv_df, self.capacity_data_wind_df, self.capacity_data_hydro_df = dh.load_capacity_data(
            self.case_study, self.num_t, self.start_t, self.enable_pv, self.enable_wind, self.enable_hydro)

    def get_bus_branch_demand_data(self):
        """
        Retrieves and loads data related to buses, branches, and demand for the specified case study.
        This method assigns the following attributes:
        - `self.buses_df`: DataFrame containing information about the buses.
        - `self.branches_df`: DataFrame containing information about the branches.
        - `self.demand_p_df`: DataFrame containing active power demand data.
        - `self.demand_q_df`: DataFrame containing reactive power demand data.
        The data is loaded using the `dh.load_bus_branch_demand_data` function, which takes
        the case study identifier and the number of time steps as input.
        Args:
            None
        Returns:
            None
        """
        self.buses_df, self.branches_df, self.demand_p_df, self.demand_q_df, self.node_data = dh.load_bus_branch_demand_data(
            self.case_study, self.num_t, self.start_t, self.slack_bus)

    def get_bus_batt_capacity_data(self):
        """
        Loads distributed battery capacities used for modeling decentralized storage systems.

        This method retrieves the randomly distributed decentralized storage capacity of a region

        The resulting data is stored in `self.distributed_batt_data_df`.

        Returns
        -------
        None
        """
        self.bus_batt_cap_df = dh.distribute_storage_capacity(self.case_study, self.pBatt_cap, self.slack_bus,
                                                              node_ids=self.node_data['node'].values,
                                                              zero_capacity_nodes=None, seed=42)

    def get_distributed_battery_data(self):
        """
        Loads distributed battery parameters used for modeling decentralized storage systems.

        This method retrieves technical parameters (e.g. efficiency, charge/discharge limits)
        for distributed batteries.

        The resulting data is stored in `self.distributed_batt_data_df`.

        Returns
        -------
        None
        """
        self.distributed_batt_data_df = dh.load_distributed_battery_data()

    def get_data(self):
        """
        Retrieves and processes various system data including bus data, branch data, demand data, 
        price data, and electric vehicle (EV) data.
        This method calls internal methods to gather and process the following:
        - Bus data
        - Branch data
        - Demand data
        - Price data
        - Generator data
        - Distributed energy storage data
        - Availability matrix
        - Renewable capacity factors
        - Electric vehicle (EV) data
        Returns:
            None
        """

        self.get_bus_branch_demand_data()
        self.get_price_data()
        self.get_ev_data()
        self.get_availability_matrix()
        self.get_capacity_data()
        self.get_generator_data()
        self.get_bus_batt_capacity_data()
        self.get_distributed_battery_data()

        return None
