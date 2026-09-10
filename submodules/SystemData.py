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
import os

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

    def __init__(self, num_t: int, start_t: int, case_study: str, scenario: str, enable_pv: bool, enable_wind: bool,
                 enable_hydro: bool, year: int, pBatt_cap: int, pCharge_Invest: int, pDoD_V2G: int, slack_bus: int,
                 ev_type: str = "EV"):
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
        self.scenario = scenario
        self.num_t = num_t
        self.start_t = start_t
        self.enable_pv = enable_pv
        self.pBatt_cap = pBatt_cap
        self.pCharge_Invest = pCharge_Invest
        self.pDoD_V2G = pDoD_V2G
        self.slack_bus = slack_bus
        if ev_type not in ("EV", "V2G", "UC"):
            raise ValueError(f"Invalid ev_type '{ev_type}' - must be 'EV', 'V2G', or 'UC'.")
        self.ev_type = ev_type
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

    def _build_uc_dummy_ev(self) -> pd.DataFrame:
        """
        Build a single inert placeholder EV for uncontrolled-charging (UC) runs.

        Under UC, real charging load is pre-baked directly into the nodal demand
        time series (see :meth:`_apply_uc_dumb_charging_demand`), so the optimization
        model must not treat any vehicle as genuinely flexible. This placeholder is
        pinned to the slack bus, available for the entire simulated horizon, and has
        zero trip demand and zero charge/discharge power - the optimizer can never do
        anything with it. It exists purely so downstream code that assumes at least
        one EV (``model.EVs``, the availability matrix, etc.) keeps working.

        Notes
        -----
        ``departure_time [h]`` is set to ``self.num_t`` (one past the last valid
        time index) rather than a fixed constant, so the vehicle is always
        available for the full horizon regardless of ``num_t`` - and, critically,
        never triggers ``create_availability_matrix``'s interactive
        "repeat EV pattern? (y/n)" prompt, which fires whenever
        ``num_t > max(departure_time)`` and would otherwise silently hang an
        automated run.

        Returns
        -------
        pd.DataFrame
        Single-row EV table stored in ``self.evs_df``.
        """
        self.evs_df = pd.DataFrame([{
            'vehicle_name': 'dummy_UC_vehicle',
            'type': self.ev_type,
            'node': self.slack_bus,
            'arrival_time [h]': 0,
            'departure_time [h]': self.num_t,
            'EV_demand [MWh]': 0.0,
            'variant': None,
            'EV_SOC_max [MWh]': 0.0,
            'Ev_ch_max [MW]': 0.0,
            'Ev_dch_max [MW]': 0.0,
            'eta_ch_EV': 1.0,
            'eta_dch_EV': 1.0,
            'SoC_init_EV': 0.0,
        }])
        return self.evs_df

    def _load_real_ev_fleet(self, ev_type_for_sampling: str, random_state: int | None = None) -> pd.DataFrame:
        """
        Load and enrich the real EV trace data for ``self.scenario``.

        The method performs the following steps:
        - Load raw EV traces and a battery-spec lookup via ``dh``.
        - Compute pre-visit EV energy demand from cumulative ETC values.
        - Assign a sampled vehicle "variant" per vehicle (stochastic mapping)
        using pre-defined archetype probabilities.
        - Merge battery characteristics from the lookup into the trace table.
        - Export the enriched table to ``output/ev_configuration_<case>_<scenario>.xlsx``.

        Parameters
        ----------
        ev_type_for_sampling : str
            'EV' or 'V2G' - which archetype pool to sample battery variants from.
            Passed explicitly rather than always using ``self.ev_type`` so this same
            loader can be reused to load the real fleet behind a UC run's
            dumb-charging simulation (see :meth:`get_ev_data`), where ``self.ev_type``
            is ``'UC'`` and would otherwise match neither pool.
        random_state : int | None, optional
        Seed for reproducible variant sampling. If ``None`` the global
        numpy RNG state is used.

        Returns
        -------
        pd.DataFrame
        Enriched EV trace table (not assigned to ``self.evs_df`` here - the caller
        decides where it goes).
        """
        evs_df = dh.load_ev_data(self.case_study, self.scenario)
        self.ev_battery_data = dh.load_ev_battery_data()
        # Copy & sort
        evs_df = evs_df.copy()
        evs_df = evs_df.sort_values(['agent', 'out [h]'])

        # Compute demand from cumulative ETC
        group_col = 'agent' if 'agent' in evs_df.columns else (
            'vehicle_name' if 'vehicle_name' in evs_df.columns else 'vehicle_id')
        dep_col = 'departure_time [h]' if 'departure_time [h]' in evs_df.columns else (
            'out [h]' if 'out [h]' in evs_df.columns else 'departure')
        df_sorted = evs_df.sort_values([group_col, dep_col]).copy()
        df_sorted['EV_demand [MWh]'] = (
                abs(df_sorted.groupby(group_col)['ETC [MWh]'].shift(-1) - df_sorted['ETC [MWh]'])).fillna(0).clip(
            lower=0)

        # Copy results back into evs_df while keeping its original row order
        evs_df.loc[df_sorted.index, 'EV_demand [MWh]'] = df_sorted['EV_demand [MWh]']

        # Rename columns
        evs_df.rename(columns={
            'agent': 'vehicle_name',
            'osm_id': 'vehicle_id',
            'in [h]': 'arrival_time [h]',
            'out [h]': 'departure_time [h]'
        }, inplace=True)
        # Update group/dep column names after rename
        group_col = 'vehicle_name' if 'vehicle_name' in evs_df.columns else group_col

        # Every vehicle in the fleet is the single global type configured for this run -
        # no per-vehicle 'archetype'/'type' column is read from the source data anymore.
        evs_df['type'] = ev_type_for_sampling

        # Define prior probabilities per archetype group
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

        # Build SOC_max lookup from battery data: variant -> SOC_max
        soc_max_lookup = self.ev_battery_data.set_index('variant')['SOC_max'].to_dict()

        # Per-vehicle maximum demand (used to filter eligible variants)
        vehicle_id_col = group_col
        max_demand_per_vehicle = (
            evs_df.groupby(vehicle_id_col)['EV_demand [MWh]'].max()
        )

        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        def sample_variant_demand_aware(ev_type: str, max_demand: float):
            """
            Sample a variant from the prior distribution, restricted to variants
            whose SOC_max can cover the vehicle's maximum observed trip demand.
            Prior probabilities are renormalized over eligible variants.
            Falls back to the variant with the largest SOC_max if no variant
            satisfies the constraint (should not occur with realistic data).
            """
            if ev_type == "EV":
                prior = ev_probs
            elif ev_type == "V2G":
                prior = v2g_probs
            else:
                return None

            eligible = {v: p for v, p in prior.items()
                        if soc_max_lookup.get(v, 0.0) >= max_demand}

            if not eligible:
                # Fallback: pick the variant with the largest SOC_max for this type
                candidates = {v: soc_max_lookup[v] for v in prior if v in soc_max_lookup}
                fallback = max(candidates, key=candidates.get)
                return fallback

            variants = list(eligible.keys())
            weights = np.array([eligible[v] for v in variants], dtype=float)
            weights /= weights.sum()  # renormalize
            return rng.choice(variants, p=weights)

        # Build a mapping vehicle -> variant using demand-aware sampling. Every vehicle
        # shares the same ev_type_for_sampling, so no per-vehicle type lookup is needed.
        variant_map = {}
        for veh in evs_df[vehicle_id_col].unique():
            max_demand = max_demand_per_vehicle.get(veh, 0.0)
            variant_map[veh] = sample_variant_demand_aware(ev_type_for_sampling, max_demand)

        # Map the variant back to every row for that vehicle
        evs_df['variant'] = evs_df[vehicle_id_col].map(variant_map)

        # Merge battery specs (SOC_max, ch, dch) from lookup
        evs_df = evs_df.merge(
            self.ev_battery_data[['variant', 'SOC_max', 'ch', 'dch', 'eta_ch_EV', 'eta_dch_EV', 'SoC_init_EV']],
            on='variant',
            how='left'
        )
        evs_df.rename(columns={'SOC_max': 'EV_SOC_max [MWh]',
                                'ch': 'Ev_ch_max [MW]',
                                'dch': 'Ev_dch_max [MW]'}, inplace=True)

        # Sanity check: log any vehicles where demand still exceeds SOC_max
        violations = evs_df[evs_df['EV_demand [MWh]'] > evs_df['EV_SOC_max [MWh]']]
        if not violations.empty:
            print(f"WARNING: {violations[vehicle_id_col].nunique()} vehicle(s) still have "
                  f"EV_demand > EV_SOC_max after demand-aware assignment. "
                  f"Check battery data coverage.")
        else:
            print("Demand-aware variant assignment: all vehicles satisfy SOC_max >= max demand.")

        # Export enriched EV configuration to Excel
        output_cols = [
            'vehicle_name', 'type', 'node', 'arrival_time [h]', 'departure_time [h]',
            'EV_demand [MWh]', 'variant', 'EV_SOC_max [MWh]', 'Ev_ch_max [MW]',
            'Ev_dch_max [MW]', 'eta_ch_EV', 'eta_dch_EV', 'SoC_init_EV'
        ]
        os.makedirs('output', exist_ok=True)
        evs_df[output_cols].to_excel(
            os.path.join('output', f'ev_configuration_{self.case_study}_{self.scenario}.xlsx'),
            sheet_name='adjusted_intervals_hrs',
            index=False
        )

        print('Added EV demand and variant data. Sample of enriched EV table:')
        return evs_df

    def _apply_uc_dumb_charging_demand(self, real_fleet_df: pd.DataFrame) -> None:
        """
        Simulate uncontrolled ("dumb") charging for the real fleet behind this UC run,
        add the resulting nodal demand directly into ``self.demand_p_df`` in memory,
        and stage the known per-EV/per-node/cost results for later export.

        Ported from the former standalone ``Calculate_dumb_charging.py`` script -
        no separate script invocation or manual demand-CSV swap is needed anymore.
        Writes a debug CSV of the resulting demand for inspection only; nothing in
        the pipeline reads it back.

        Since the optimizer never sees the real fleet under UC (see
        :meth:`_build_uc_dummy_ev`), this precomputed per-EV/per-node/cost detail is
        the only place that data exists. It's staged here as instance attributes
        (``self.uc_ev_charging_df``, ``self.uc_node_charging_total_df``,
        ``self.uc_charging_cost_df``) rather than written to ``result.sqlite``
        directly, since that database doesn't exist yet at this point in the
        pipeline (data loading happens before the model is even built, let alone
        solved) - ``V2G.export_results()`` writes them out after the solved model's
        own tables, once the file actually exists.

        Parameters
        ----------
        real_fleet_df : pd.DataFrame
        Enriched real EV fleet table, as returned by :meth:`_load_real_ev_fleet`.

        Returns
        -------
        None
        """
        charging_demand, per_vehicle_charging = dh.simulate_dumb_charging(real_fleet_df, self.num_t)
        self.demand_p_df = dh.add_dumb_charging_to_demand(self.demand_p_df, charging_demand)

        total_added = charging_demand.values.sum()
        peak = charging_demand.values.max()
        peak_node = charging_demand.max().idxmax()
        peak_hour = charging_demand.max(axis=1).idxmax()
        print(f"  UC dumb-charging demand added: {total_added:.4f} MWh total, "
              f"peak {peak:.4f} MW (node {peak_node}, hour {peak_hour})")

        os.makedirs('output', exist_ok=True)
        debug_path = os.path.join('output', f'demand_data_uc_{self.case_study}_{self.scenario}.csv')
        self.demand_p_df.to_csv(debug_path, index=False)
        print(f"  Debug export (not read back by the pipeline): {debug_path}")

        # Per-EV charging detail (vehicle_name, node, time, charge_MW) - known exactly,
        # since it comes straight out of the simulation rather than a solved model.
        self.uc_ev_charging_df = per_vehicle_charging

        # Per-node charging total, long format matching the Pyomo-exported tables'
        # own convention (index columns + 'values'), for consistency in result.sqlite.
        self.uc_node_charging_total_df = (
            charging_demand.rename_axis('time').reset_index()
            .melt(id_vars='time', var_name='node', value_name='values')
            [['node', 'time', 'values']]
        )

        # System-wide charging cost per timestep: total charge already known, multiplied
        # by the already-loaded price series - the overall total is SUM(cost) over this.
        price = self.import_price_df['price'].to_numpy()
        total_charge_mw = charging_demand.sum(axis=1).to_numpy()
        if len(price) != len(total_charge_mw):
            raise ValueError(
                f"Price series has {len(price)} timesteps but charging demand has "
                f"{len(total_charge_mw)} - cannot align for cost calculation."
            )
        self.uc_charging_cost_df = pd.DataFrame({
            'time': range(self.num_t),
            'total_charge_MW': total_charge_mw,
            'price': price,
            'cost': total_charge_mw * price,
        })
        print(f"  UC charging cost: {self.uc_charging_cost_df['cost'].sum():.2f} total "
              f"(price x known charging demand, precomputed)")

    def get_ev_data(self, random_state: int | None = None):
        """
        Load and enrich EV trace data, or build the UC placeholder.

        For ``ev_type`` 'EV'/'V2G', loads and enriches the real fleet for
        ``self.scenario`` via :meth:`_load_real_ev_fleet`.

        For ``ev_type`` 'UC', loads the real fleet under the 'EV' archetype pool
        purely to simulate the resulting dumb-charging demand (added into
        ``self.demand_p_df`` via :meth:`_apply_uc_dumb_charging_demand`), then
        replaces ``self.evs_df`` with a single inert placeholder vehicle from
        :meth:`_build_uc_dummy_ev`. The archetype pool choice doesn't affect the
        simulated demand - it never reads discharge-related columns - it's just a
        valid pool for the variant-sampling logic to draw from.

        Parameters
        ----------
        random_state : int | None, optional
        Seed for reproducible variant sampling. If ``None`` the global
        numpy RNG state is used.

        Returns
        -------
        pd.DataFrame
        Enriched EV trace table (or the UC placeholder), stored in ``self.evs_df``.
        """
        if self.ev_type == "UC":
            real_fleet_df = self._load_real_ev_fleet(ev_type_for_sampling="EV", random_state=random_state)
            self._apply_uc_dumb_charging_demand(real_fleet_df)
            self.evs_df = self._build_uc_dummy_ev()
            return self.evs_df

        self.evs_df = self._load_real_ev_fleet(ev_type_for_sampling=self.ev_type, random_state=random_state)
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
