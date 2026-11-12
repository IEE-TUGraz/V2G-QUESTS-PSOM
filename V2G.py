import math
import os
import sqlite3
import subprocess
import time

import pandas as pd
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, minimize, Constraint, Param, value, \
    SolverFactory, TransformationFactory, Binary, Reals, Set, Any, Expression
from pyomo.opt import SolverStatus, TerminationCondition
from collections import defaultdict

from submodules import SystemData as sd
from submodules import data_helper as dh
from submodules import helper_functions as h


class V2G:
    """
    Vehicle-to-Grid (V2G) optimization model.

    This class handles the complete workflow for a V2G system, including
    data loading, preprocessing, Pyomo model creation, solving, and exporting
    results. It supports both DC-OPF and SOCP formulations.

    Overview
    --------------
    - **Data processing:** load and preprocess grid, EV, generator, and price data
    - **Model creation:** define sets, parameters, variables, constraints, and objective
    - **Solving:** run the Pyomo model using Gurobi
    - **Exporting:** write results to structured CSV files
    - **Visualization:** optional network plotting of inputs or results
    """

    def __init__(self):
        """
        Initialize the V2G class and load configuration parameters from
        ``config.yml``.

        This constructor sets up all configuration options for the case study,
        simulation horizon, renewable settings, solver parameters, and paths
        to external tools. It also initializes placeholders for the Pyomo model
        and solver results.

        Attributes
        ----------
        enable_rMIP : bool
            Whether to solve the model as a relaxed Mixed Integer Problem (rMIP).
        MIP_Gap : float
            Acceptable optimality gap for the Mixed Integer Programming solver.
        case_study : str
            Name of the selected case study loaded from the configuration file.
        dc_opf : bool
            If ``True``, use DC-OPF; otherwise use SOCP as the convex AC-OPF formulation.
        start_t : int
            Index of the first time step considered in the optimization.
        num_t : int
            Number of time steps included in the simulation.
        enable_pv : bool
            Whether photovoltaic generation is included in the model.
        enable_wind : bool
            Whether wind generation is included in the model.
        enable_hydro : bool
            Whether hydro generation is included in the model.
        lego_gpt_path : str
            Filesystem path to the LEGO-GPT script used for plotting.
        lego_gpt_env_path : str
            Path to the virtual environment for LEGO-GPT.
        slack_bus : int
            ID of the slack/reference bus in the power grid.
        year : int
            Simulation year (used for selecting time series and input data).
        pBatt_cap : float
            Installed decentralized storage capacity in the system (excluding V2G).
        pCharge_Invest : float
            Investment cost for charging infrastructure.
        pQfactor : float
            Reactive power factor applied to active power demand.
        pV_slack : float
            Slack-bus voltage magnitude used in the SOCP formulation.
        pCh_Inf_Cost : float
            Cost of charging infrastructure per MW.
        pPenality : float
            Penalty factor for excess or missing energy at EVs or buses.
        model : pyomo.environ.ConcreteModel or None
            Pyomo optimization model (created later in :meth:`create_model`).
        results : pyomo.opt.results.SolverResults or None
            Solver results returned after :meth:`solve_model`.
        """

        # Load configuration parameters from the `config.yml` file
        config = h.load_config("config.yml")
        self.enable_rMIP = bool(config['enable_rMIP'])
        self.MIP_Gap = config['MIP_Gap']
        self.case_study = config['case_study']
        self.scenario = config['scenario']
        self.dc_opf = bool(config['dc_opf'])
        self.start_t = config['start_t']
        self.num_t = config['num_t']
        self.enable_pv = config['enable_pv']
        self.enable_wind = config['enable_wind']
        self.enable_hydro = config['enable_hydro']
        self.lego_gpt_path = config['LEGO_GPT_path']
        self.lego_gpt_env_path = config['LEGO_GPT_env_path']
        self.slack_bus = config['slack_bus']
        self.year = config['year']
        self.pBatt_cap = config['pBatt_cap']
        self.pCharge_Invest = config['pCharge_Invest']
        self.pDoD_V2G = config['pDoD_V2G']
        self.ev_type = config.get('ev_type', 'EV')
        if self.ev_type not in ('EV', 'V2G', 'UC'):
            raise ValueError(f"Invalid ev_type '{self.ev_type}' in config.yml - must be 'EV', 'V2G', or 'UC'.")
        self.pQfactor = config['pQfactor']
        self.pV_slack = config['pV_slack']
        self.pCh_Inf_Cost = config['pCh_Inf_Cost']
        self.pPenality = config['pPenality']
        self.pEnable_Soft_Line_Limits = config.get('Enable_Soft_Line_Limits', False)
        self.pEnable_Soft_Voltage_Limits = config.get('Enable_Soft_Voltage_Limits', False)
        self.pSoft_Line_Limit = config.get('Line_soft_limit')
        self.pSoft_Voltage_Limit = config.get('V_soft_limit')
        self.pEnable_SOCP_Exactness_Penalty = config.get('Enable_SOCP_Exactness_Penalty', False)
        self.pSOCP_Exactness_Epsilon = float(config.get('SOCP_exactness_epsilon', 1e-4))
        # Penalty convex-concave procedure (Lipp & Boyd 2016) schedule for the exactness tightening
        self.pCCP_Tau_Init = float(config.get('CCP_tau_init', 1.0e3))
        self.pCCP_Tau_Mult = float(config.get('CCP_tau_mult', 10.0))
        self.pCCP_Tau_Max = float(config.get('CCP_tau_max', 1.0e7))
        self.pCCP_Max_Iter = int(config.get('CCP_max_iter', 20))
        # Modelling and solver constants (defaults are the values used for the published results)
        self.pBigM = float(config.get('Big_M', 1.0e5))
        self.pExport_Reward = float(config.get('pExport_Reward', 1.0e-3))
        self.pSoft_Line_Penalty_Factor = float(config.get('Line_soft_penalty_factor', 0.1))
        self.pSoft_Voltage_Penalty_Factor = float(config.get('V_soft_penalty_factor', 0.052))
        self.gurobi_options = config.get('Gurobi_options') or {'BarHomogeneous': 1, 'NumericFocus': 3,
                                                               'BarConvTol': 1.0e-6}

        # Initialize placeholders for the Pyomo model and solver results
        self.model = None  # Pyomo model object
        self.results = None  # Pyomo results object
        self.solution_loaded = False  # True once a (possibly non-optimal) solution is loaded
        self.run_status = 'not_solved'  # Overall solve outcome, set by solve_model

    def get_data(self):
        """
        Load system data using the SystemData helper and assign it to instance attributes.

        This method initializes a :class:`sd.SystemData` object using the configuration
        values stored in this instance, calls its ``get_data()`` method to populate
        internal dataframes and matrices, and then maps these results to the
        V2G instance attributes.

        Side Effects
        ------------
        - Sets attributes used by later preprocessing and model construction:
            - ``buses_df``, ``branches_df``, ``loads_p_df``, ``evs_df``, ``price_df``
            - ``capacity_pv_df``, ``capacity_wind_df``, ``capacity_hydro_df``
            - ``generators_df``, ``bus_batt_cap_df``, ``pCharge_Invest``
            - ``availability_matrix``
        - Prints progress messages and total load time to stdout.

        Dependencies
        ------------
        - Relies on ``sd.SystemData`` to provide the data attributes listed above.
        - Uses ``time`` for timing.
        - Requires that the following instance attributes are already set:
            - ``num_t``, ``start_t``, ``case_study``, ``scenario``, ``enable_pv``, ``enable_wind``,
              ``enable_hydro``, ``year``, ``pBatt_cap``, ``pCharge_Invest``, ``pDoD_V2G``, ``slack_bus``, ``ev_type``

        Returns
        -------
        None
        """

        # measure start time for loading
        start_time = time.time()
        print('Loading system data')

        # instantiate SystemData with configuration values and load the data
        case_study = sd.SystemData(self.num_t, self.start_t, self.case_study, self.scenario, self.enable_pv,
                                   self.enable_wind, self.enable_hydro, self.year, self.pBatt_cap, self.pCharge_Invest, self.pDoD_V2G, self.slack_bus,
                                   ev_type=self.ev_type)
        case_study.get_data()

        # map loaded data to instance attributes for downstream usage
        self.buses_df = case_study.buses_df
        self.branches_df = case_study.branches_df
        self.loads_p_df = case_study.demand_p_df
        self.evs_df = case_study.evs_df
        self.price_df = case_study.import_price_df
        self.capacity_pv_df = case_study.capacity_data_pv_df
        self.capacity_wind_df = case_study.capacity_data_wind_df
        self.capacity_hydro_df = case_study.capacity_data_hydro_df
        self.generators_df = case_study.generators_df
        self.bus_batt_cap_df = case_study.bus_batt_cap_df
        self.pCharge_Invest = case_study.pCharge_Invest
        self.distributed_batt_data_df = case_study.distributed_batt_data_df
        self.evs_df = case_study.evs_df
        self.availability_matrix = case_study.availability_matrix
        # Precomputed UC EV-charging results, only populated when ev_type == "UC"
        # (see SystemData._apply_uc_dumb_charging_demand); None otherwise.
        self.uc_ev_charging_df = getattr(case_study, 'uc_ev_charging_df', None)
        self.uc_node_charging_total_df = getattr(case_study, 'uc_node_charging_total_df', None)
        self.uc_charging_cost_df = getattr(case_study, 'uc_charging_cost_df', None)

        # report elapsed time
        print('Data Loaded in', time.time() - start_time, 'seconds')

    def extract_generator_params(self):
        """
        Extract generator parameters from ``generators_df`` and assign them as instance attributes.

        The method processes generator data for predefined types (``pv``, ``hydro``, ``wind``) and
        parameters (``Pmax``, ``Pmin``, ``Qmax``, ``Qmin``).
        For every (type, parameter) combination, it creates a dictionary mapping bus IDs to values
        and stores it in an attribute such as ``pv_Pmax`` or ``wind_Qmin``.

        Attributes Created
        ------------------
        The following attributes are dynamically created and attached to the instance:

        +--------------+-------------------------------+
        | Generator    | Attributes Created            |
        | Type         |                               |
        +==============+===============================+
        | ``pv``       | ``pv_Pmax``, ``pv_Pmin``,     |
        |              | ``pv_Qmax``, ``pv_Qmin``      |
        +--------------+-------------------------------+
        | ``hydro``    | ``hydro_Pmax``, ``hydro_Pmin``,|
        |              | ``hydro_Qmax``, ``hydro_Qmin`` |
        +--------------+-------------------------------+
        | ``wind``     | ``wind_Pmax``, ``wind_Pmin``, |
        |              | ``wind_Qmax``, ``wind_Qmin``  |
        +--------------+-------------------------------+

        Example
        -------
        If ``generators_df`` contains:

        +---------+--------+-------+-------+-------+-------+
        | bus     | type   | Pmax  | Pmin  | Qmax  | Qmin  |
        +=========+========+=======+=======+=======+=======+
        | Node_1  | pv     | 100   | 50    | 30    | 10    |
        +---------+--------+-------+-------+-------+-------+
        | Node_2  | wind   | 200   | 100   | 60    | 20    |
        +---------+--------+-------+-------+-------+-------+
        | Node_3  | hydro  | 150   | 75    | 45    | 15    |
        +---------+--------+-------+-------+-------+-------+

        Then the method will produce attributes such as:

        - ``self.pv_Pmax   = {"Node_1": 100}``
        - ``self.wind_Pmin = {"Node_2": 100}``
        - ``self.hydro_Qmax = {"Node_3": 45}``

        Raises
        ------
        AttributeError
            If ``generators_df`` is missing required columns or not properly defined.
        """
        types = ['pv', 'hydro', 'wind']
        params = ['Pmax', 'Pmin', 'Qmax', 'Qmin']
        self.generators_df['type'] = self.generators_df['type'].str.lower()
        for gen_type in types:
            gen_df = self.generators_df[self.generators_df['type'] == gen_type]

            for param in params:
                param_dict = dict(zip(gen_df['bus'], gen_df[param]))
                setattr(self, f"{gen_type}_{param.lower()}", param_dict)

    def preprocess_data(self):
        """
        Preprocess the data required for the V2G (Vehicle-to-Grid) system.

        This method performs the following steps:

        1. Creates a mirrored copy of each branch for the SOCP formulation
           (swapping ``fbus`` and ``tbus``) and appends it to ``branches_df``.
        2. Sets a MultiIndex on the ``branches_df`` DataFrame using the ``fbus`` and
           ``tbus`` columns.
        3. Aligns the index of ``loads_p_df`` with the index of ``price_df`` to ensure
           consistency in time indexing.
        4. Converts the ``loads_p_df`` DataFrame to a float type.

        Attributes
        ----------
        branches_df : pd.DataFrame
            DataFrame containing branch data, modified during preprocessing.
        loads_p_df : pd.DataFrame
            DataFrame containing load power data, aligned with the time index of
            ``price_df``.
        price_df : pd.DataFrame
            DataFrame containing price data used for time index alignment.

        Returns
        -------
        None
        """

        start_time = time.time()
        print('Preprocessing data')

        # Copies the original branches_df to avoid modifying it directly
        branches_df = self.branches_df.copy()

        # Create mirrored branches by swapping 'fbus' and 'tbus'
        branches_df = pd.concat([branches_df, branches_df.rename(columns={'fbus': 'tbus', 'tbus': 'fbus'})], axis=0)

        # Multilevel index for branches_df
        multi_index = pd.MultiIndex.from_tuples(
            [(fbus, tbus) for fbus, tbus in zip(branches_df['fbus'], branches_df['tbus'])], names=['fbus', 'tbus'])
        branches_df.index = multi_index
        self.branches_df = branches_df  # Actualize the branches_df attribute

        # Correct index of loads_p_df to match price_df
        self.loads_p_df.index = self.price_df.index

        # Convert loads_p_df to float
        self.loads_p_df.astype(float)

        # create availability dict for EVs at nodes over time
        self.availability_dict = self.availability_matrix.set_index(['vehicle', 'node', 'time'])['available'].astype(
            int).to_dict()

        self.evs_at_node_time = defaultdict(list)
        for (ev, bus, t) in self.availability_matrix.set_index(['vehicle', 'node', 'time']).index:
            self.evs_at_node_time[(bus, t)].append(ev)

        print('Data preprocessed in', time.time() - start_time, 'seconds')

    def create_model(self):
        """
        Create the Pyomo optimization model for the Vehicle-to-Grid (V2G) system.

        This method defines all sets, parameters, variables, and constraints
        required for the optimization model. It supports both DC Optimal Power Flow
        (DC-OPF) and Second-Order Cone Programming (SOCP) formulations.

        Steps
        -----
        1. **Define sets**:
           - Time steps
           - Grid buses
           - Branches
           - Electric Vehicles (EVs)
           - Generators (PV, wind, hydro)
        2. **Define parameters** for:
           - Grid topology and branch characteristics
           - EV availability, charging/discharging limits
           - Renewable generator capacities
           - Storage capacities and battery limits
        3. **Define variables**:
           - Active and reactive power flows
           - EV charging and discharging powers
           - Storage state of charge and voltage magnitudes
        4. **Define constraints**:
           - Power balance at each bus and time step
           - Generator and EV operational limits
           - Battery operation and state constraints
           - Second-order cone constraints (if SOCP formulation)
        5. **Define objective function** to minimize:
           - Electricity import costs
           - EV energy costs
           - Penalty for slack bus deviations
           - Charging infrastructure investment costs

        Returns
        -------
        None
        """

        start_time = time.time()
        print('Creating Pyomo model')
        model = ConcreteModel()

        # Define sets
        model.Time = Set(initialize=list(self.loads_p_df.index),
                         doc='Time steps')

        model.Buses = Set(initialize=list(self.buses_df.id.values),
                          doc='Grid-buses')

        model.Branches = Set(dimen=2, initialize=self.branches_df.index,
                             doc='Grid-branches')

        model.slack_bus = self.slack_bus

        model.EVs = Set(initialize=list(self.evs_df['vehicle_name'].unique()),
                        doc="Electric Vehicles")

        model.Generators = Set(initialize=self.generators_df['type'].dropna().unique().tolist(),
                               doc='Generator types')

        model.PV_Buses = Set(within=model.Buses, initialize=self.pv_pmax.keys(),
                             doc='Buses with PV generators')

        model.Wind_Buses = Set(within=model.Buses, initialize=self.wind_pmax.keys(),
                               doc='Buses with Wind generators')

        model.Hydro_Buses = Set(within=model.Buses, initialize=self.hydro_pmax.keys(),
                                doc='Buses with Hydro generators')

        model.EV_Available_Locations = Set(dimen=3, initialize=[(row['vehicle'], row['node'], row['time']) for _, row in
                                                                self.availability_matrix.iterrows()],
                                           doc="Sparse set of (EV, bus, time) where EV is actually available")

        print('Sets sucessfully defined')
        # ============================================
        # Define Parameters
        # ============================================

        # --------------------------------------------
        # General Parameters
        # --------------------------------------------
        model.add_component('pBigM',
                            Param(initialize=self.pBigM, domain=NonNegativeReals, doc='Big M constant for linearization'))

        for c in self.buses_df.columns:
            if c not in ['id']:
                model.add_component(c, Param(model.Buses,
                                             initialize=self.buses_df.set_index('id')[c].astype(float).to_dict(),
                                             doc='Bus parameters'))

        for c in self.branches_df.columns:
            if c not in ['fbus', 'tbus']:
                model.add_component(c, Param(model.Branches, initialize=self.branches_df[c].astype(float).to_dict(),
                                             doc='Branch parameters'))

        model.add_component('pDemand',
                            Param(model.Time, model.Buses, initialize=self.loads_p_df.astype(float).stack().to_dict(),
                                  domain=Reals, doc='Active power demand at buses over time'))

        model.add_component('pElPrice',
                            Param(model.Time, initialize=self.price_df['price'].astype(float).to_dict(), domain=Reals))

        model.add_component('pCf_Pv',
                            Param(model.Time, initialize=self.capacity_pv_df['electricity'].astype(float).to_dict(),
                                  domain=Reals, doc='Capacity Factors for PV generation'))

        model.add_component('pCf_Wind',
                            Param(model.Time, initialize=self.capacity_wind_df['electricity'].astype(float).to_dict(),
                                  domain=Reals, doc='Capacity Factors for Wind generation'))

        model.add_component('pCf_Hydro',
                            Param(model.Time, initialize=self.capacity_hydro_df['electricity'].astype(float).to_dict(),
                                  domain=Reals, doc='Capacity Factors for Hydro generation'))

        model.add_component('pPenality', Param(initialize=self.pPenality, domain=NonNegativeReals,
                                               doc='Penalty factor for exes and lack of energy in EVs and nodes'))
        model.add_component('pSoft_Line_Limit', Param(initialize=self.pSoft_Line_Limit, domain=NonNegativeReals,
                                                      doc='Soft limit for the thermal line constraints'))

        # AC-OPF specific parameters
        if not self.dc_opf:
            model.add_component('pQfactor', Param(initialize=self.pQfactor, domain=NonNegativeReals,
                                                  doc='Reactive power cost factor for SOCP formulation'))
            model.add_component('pV_slack', Param(initialize=self.pV_slack, domain=NonNegativeReals,
                                                  doc='Slack bus voltage magnitude for SOCP formulation'))
            model.add_component('pVmin', Param(model.Buses, initialize=model.Vmin, domain=NonNegativeReals,
                                               doc='Minimum voltage magnitude limit for SOCP formulation'))
            model.add_component('pVmax', Param(model.Buses, initialize=model.Vmax, domain=NonNegativeReals,
                                               doc='Maximum voltage magnitude limit for SOCP formulation'))

            if self.pEnable_Soft_Voltage_Limits:
                model.add_component('pSoft_Voltage_Limit',
                                    Param(initialize=self.pSoft_Voltage_Limit, domain=NonNegativeReals,
                                          doc='Soft limit for the SOCP voltage magnitude constraints'))
            else:
                model.add_component('pSoft_Voltage_Limit', Param(initialize=0, domain=NonNegativeReals,
                                                                 doc='Soft limit for the SOCP voltage magnitude constraints (set to 0 if soft limits are disabled)'))

            if self.pEnable_SOCP_Exactness_Penalty:
                # Canonical (undirected) branch set: one entry per node pair, used to avoid
                # imposing the exactness tightening constraint twice on mirrored branches.
                model.CanonicalBranches = Set(dimen=2, initialize=[
                    (i, j) for (i, j) in dict.fromkeys(self.branches_df.index) if i < j
                ], doc='One representative direction per undirected branch, used for SOCP exactness tightening')

                model.add_component('pSOCP_Exactness_Epsilon',
                                    Param(initialize=self.pSOCP_Exactness_Epsilon, domain=NonNegativeReals,
                                          doc='Absolute tolerance on the SOCP relaxation gap'
                                              ' (vCii_i*vCii_j - vCij^2 - vSij^2), in p.u.^2'))

                # Linearization point (previous solved values) for the convex majorant of the
                # bilinear gap term. Populated from a first, untightened solve and then used to
                # build a global convex over-estimator of the (nonconvex) SOCP relaxation gap.
                model.pCii_lin = Param(model.Buses, model.Time, initialize=float(self.pV_slack) ** 2,
                                       mutable=True,
                                       doc='Linearization point for vCii used in the SOCP exactness majorant')
                model.pCij_lin = Param(model.CanonicalBranches, model.Time, initialize=float(self.pV_slack) ** 2,
                                       mutable=True,
                                       doc='Linearization point for vCij used in the SOCP exactness majorant')
                model.pSij_lin = Param(model.CanonicalBranches, model.Time, initialize=0.0, mutable=True,
                                       doc='Linearization point for vSij used in the SOCP exactness majorant')

                # Penalty weight on the majorant slack, raised every CCP iteration.
                model.pCCP_Tau = Param(initialize=self.pCCP_Tau_Init, mutable=True, domain=NonNegativeReals,
                                       doc='Penalty-CCP weight on the SOCP exactness slack in EUR/p.u.^2')
        # --------------------------------------------
        # Generator Parameters
        # --------------------------------------------
        # Define the types of generators and their parameters
        generator_types = ['PV', 'Wind', 'Hydro']  # List of generator types (Photovoltaic, Wind, Hydro)
        params = ['Pmax', 'Pmin', 'Qmax',
                  'Qmin']  # List of generator parameters (maximum/minimum active and reactive power)
        # Loop through each generator type and parameter to add them as model components
        for gen_type in generator_types:
            for param in params:
                param_name = f"{gen_type}_{param}"  # Construct the parameter name (e.g., PV_Pmax)
                param_data = getattr(self, f"{gen_type.lower()}_{param.lower()}")  # Retrieve the corresponding data
                bus_set = getattr(model, f"{gen_type}_Buses")  # Retrieve the set of buses for the generator type
                model.add_component(
                    param_name,
                    Param(bus_set, initialize=param_data, domain=NonNegativeReals, doc='Generator power-parameters')
                    # Add the parameter to the model
                )
        # --------------------------------------------
        # EV Parameters
        # --------------------------------------------
        model.add_component('pEV_SoC_max', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'EV_SOC_max [MWh]'].astype(float).to_dict(), domain=NonNegativeReals,
                                                 doc='Maximum SoC of EVs'))

        model.add_component('pEta_ch', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'eta_ch_EV'].astype(float).to_dict(), domain=NonNegativeReals,
                                             doc='Charging efficiency of EVs'))

        model.add_component('pEta_dch', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'eta_dch_EV'].astype(float).to_dict(), domain=NonNegativeReals,
                                              doc='Discharge efficiency of EVs'))

        model.add_component('pEVCh_max', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'Ev_ch_max [MW]'].astype(float).to_dict(), domain=NonNegativeReals,
                                               doc='Maximum charging power of EVs'))

        model.add_component('pEVDch_max', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'Ev_dch_max [MW]'].astype(float).to_dict(), domain=NonNegativeReals,
                                                doc='Maximum discharging power of EVs'))

        model.add_component('pEV_SoC_init', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'SoC_init_EV'].astype(float).to_dict(),
                                                  doc='Initial state of charge of EVs'))

        model.add_component('pCh_Inf_Cost', Param(initialize=self.pCh_Inf_Cost, domain=Reals,
                                                  doc='Charging infrastructure cost per MW'))

        model.add_component('pDoD_V2G', Param(initialize=self.pDoD_V2G, domain=NonNegativeReals,
                                              doc='Depth of discharge for V2G usage'))

        # Create a dictionary mapping each EV's name and departure time to its energy demand.
        # The `evs_df` DataFrame is indexed by 'vehicle_name' and 'departure_time [h]',
        # and the 'EV_demand [MWh]' column is converted to float and then to a dictionary.
        soc_dict = self.evs_df.set_index(['vehicle_name', 'departure_time [h]'])['EV_demand [MWh]'].astype(
            float).to_dict()

        model.add_component('pSoC_final',
                            Param(model.EVs, model.Time, initialize=lambda m, ev, t: soc_dict.get((ev, t), 0),
                                  domain=NonNegativeReals,
                                  doc='Required SoC at departure of EVs'))

        # Add availability matrix as a parameter that indicates if a EV is available at a certain node at a certain time
        model.pAvailable = Param(model.EVs, model.Buses, model.Time,
                                 initialize=lambda m, ev, bus, t: self.availability_dict.get((ev, bus, t), 0),
                                 domain=Binary,
                                 doc="Binary availability of EVs at nodes over time")

        model.add_component('pCharge_Invest', Param(initialize=float(self.pCharge_Invest), domain=Reals,
                                                    doc='Charging infrastructure investment cost per MW per EV'))
        # --------------------------------------------
        # Decentralized Battery Parameters
        # --------------------------------------------
        model.add_component('pEta_ch_batt', Param(initialize=float(self.distributed_batt_data_df['eta_ch_batt'].iat[0]),
                                                  doc='Charging efficiency of decentralized batteries'))

        model.add_component('pEta_dch_batt',
                            Param(initialize=float(self.distributed_batt_data_df['eta_dch_batt'].iat[0]),
                                  doc='Discharging efficiency of decentralized batteries'))

        model.add_component('pBatt_SoC_init',
                            Param(initialize=float(self.distributed_batt_data_df['SoC_init_batt'].iat[0]),
                                  doc='Initial state of charge of decentralized batteries'))

        model.add_component('pBatt_SoC_min', Param(initialize=float(1-self.distributed_batt_data_df['DoD_batt'].iat[0]),
                                                   doc='Depth of discharge of decentralized batteries'))

        model.add_component('pC_rate_batt', Param(initialize=float(self.distributed_batt_data_df['C_rate_batt'].iat[0]),
                                                  doc='C-rate of decentralized batteries'))

        model.add_component('pBus_batt_cap',
                            Param(model.Buses, initialize=self.bus_batt_cap_df['capacity'].astype(float).to_dict(),
                                  domain=Reals,
                                  doc='Battery capacity per bus'))
        # ============================================
        # Define Variables
        # ============================================

        # --------------------------------------------
        # General Variables
        # --------------------------------------------
        model.vImpP = Var(model.Time, domain=NonNegativeReals,
                          doc='Active Energy Import')
        model.vExpP = Var(model.Time, domain=NonNegativeReals,
                          doc='Active Energy Export')
        model.vLineP = Var(model.Branches, model.Time, domain=Reals,
                           doc='Active Power Flow on Lines')
        model.vPNS = Var(model.Buses, model.Time, domain=NonNegativeReals,
                         doc='Slack variable for active power balance at buses')
        model.vEPS = Var(model.Buses, model.Time, domain=NonNegativeReals,
                         doc='Slack variable for excess active power at buses')
        if self.pEnable_Soft_Line_Limits:
            # Overload above the soft limit is capped at the hard line limit
            model.vLineOverload = Var(model.Branches, model.Time, domain=NonNegativeReals,
                                      bounds=lambda model, i, j, t: (0, model.pPmax_line[i, j] * (1 - model.pSoft_Line_Limit)),
                                      doc='Slack variable for line overloads')
        # --------------------------------------------
        # EV Variables
        # --------------------------------------------
        model.vSoC_EV = Var(model.EVs, model.Time, domain=NonNegativeReals,
                            doc='State of Charge of EVs')
        model.vCh_EV = Var(model.EVs, model.Time, domain=NonNegativeReals,
                           doc='Charging Amount of EVs')
        model.vDch_EV = Var(model.EVs, model.Time, domain=NonNegativeReals,
                            doc='Discharging Amount of EVs')
        model.vSoC_n = Var(model.EVs, model.Time, domain=NonNegativeReals,
                           doc='Slack variable for undercharging EVs')
        model.vSoC_p = Var(model.EVs, model.Time, domain=NonNegativeReals,
                           doc='Slack variable for overcharging EVs')
        model.vbCh_EV = Var(model.EVs, model.Time, domain=Binary,
                            doc='Binary variable to avoid charging and discharging of EVs at the same time')
        model.vCh_cap_EV = Var(model.Buses, domain=NonNegativeReals,
                               doc='Charging capacity installed at each bus')
        # --------------------------------------------
        # Renewable Generator Variables
        # --------------------------------------------
        model.vProd_PV = Var(model.PV_Buses, model.Time, domain=NonNegativeReals,
                             doc='PV Production at Buses')
        model.vProd_Wind = Var(model.Wind_Buses, model.Time, domain=NonNegativeReals,
                               doc='Wind Production at Buses')
        model.vProd_Hydro = Var(model.Hydro_Buses, model.Time, domain=NonNegativeReals,
                                doc='Hydro Production at Buses')
        # --------------------------------------------
        # Storage Variables
        # --------------------------------------------
        model.vSoC_Batt = Var(model.Buses, model.Time, domain=NonNegativeReals,
                              doc='State of Charge of Batteries at Buses')

        model.vCh_Batt = Var(model.Buses, model.Time, domain=NonNegativeReals,
                             doc='Charging Amount of Buse Batteries')

        model.vDch_Batt = Var(model.Buses, model.Time, domain=NonNegativeReals,
                              doc='Discharging Amount of Bus Batteries')

        model.vbCh_Batt = Var(model.Buses, model.Time, domain=Binary,
                              doc='Binary variable to avoid charging and discharging of Bus Batteries at the same time')
        # --------------------------------------------
        # SOCP Variables
        # --------------------------------------------
        if not self.dc_opf:
            def voltage_bounds(model, i, t):
                """
                Return voltage bounds for bus i at time t.
                Slack bus has no bounds; other buses use squared voltage limits.
                """
                if i == model.slack_bus:
                    return (None, None)  # No bounds for slack bus
                else:
                    return (model.pVmin[i] ** 2, model.pVmax[i] ** 2)

            model.vCii = Var(model.Buses, model.Time, domain=NonNegativeReals, bounds=voltage_bounds,
                             doc='Squared voltage magnitude at bus i, i.e. v_{C,ii} = V_{i,re}^2 + V_{i,im}^2')

            model.vCij = Var(model.Branches, model.Time, domain=NonNegativeReals,
                             doc='Real part of the voltage product between buses'
                                 ' :math:`i` and :math:`j`, defined as :math:`v_{C,ij} = V_{i,re}V_{j,re} + V_{i,im}V_{j,im}`.')

            model.vSij = Var(model.Branches, model.Time, domain=Reals,
                             doc='Imaginary part of the voltage product between buses'
                                 ' :math:`i` and :math:`j`, defined as :math:`v_{S,ij} = V_{i,re}V_{j,im} - V_{i,re}V_{j,im}`.')
            model.vLineQ = Var(model.Branches, model.Time, domain=Reals,
                               doc='Reactive Power Flow on Lines')
            model.vImpQ = Var(model.Time, domain=Reals,
                              doc='Reactive Energy Import')
            # Note: No reactive power export variable since the is no reactive power production in the model

            def undershoot_bounds(model, i, t):
                # Gap between the preferred target (Vmin+delta)^2 and the true hard floor
                # Vmin^2 - NOT delta^2, which only measures the gap in linear voltage, not
                # in the squared quantity vCii actually lives in.
                if i == model.slack_bus:
                    return (0, 0)
                return (0, (model.pVmin[i] + model.pSoft_Voltage_Limit) ** 2 - model.pVmin[i] ** 2)

            def overshoot_bounds(model, i, t):
                # Gap between the true hard ceiling Vmax^2 and the preferred target
                # (Vmax-delta)^2, in the same squared units vCii uses.
                if i == model.slack_bus:
                    return (0, 0)
                return (0, model.pVmax[i] ** 2 - (model.pVmax[i] - model.pSoft_Voltage_Limit) ** 2)

            model.vVoltage_Undershoot = Var(model.Buses, model.Time, domain=NonNegativeReals,
                                            bounds=undershoot_bounds,
                                            doc='Slack variable for voltage magnitude undershoot')
            model.vVoltage_Overshoot = Var(model.Buses, model.Time, domain=NonNegativeReals,
                                           bounds=overshoot_bounds,
                                           doc='Slack variable for voltage magnitude overshoot')

            if self.pEnable_SOCP_Exactness_Penalty:
                model.vSOCP_Slack = Var(model.CanonicalBranches, model.Time, domain=NonNegativeReals,
                                        doc='Penalty-CCP slack on the SOCP exactness majorant in p.u.^2')

        elif self.dc_opf:
            def theta_bounds(model, i, t):
                return (model.pMin_Ang[i], model.pMax_Ang[i])

            model.vTheta = Var(model.Buses, model.Time, domain=Reals, bounds=theta_bounds,
                               doc='Voltage phase angle at each bus in the DC-OPF formulation')
        # ============================================
        # Define equations
        # ============================================

        # --------------------------------------------
        # General equations
        # --------------------------------------------
        if self.pEnable_Soft_Line_Limits:
            @model.Constraint(model.Branches, model.Time)
            def thermal_constraint_upper(model, i, j, t):
                r"""
                Soft thermal limit for active power flow on branch :math:`(i, j)` in flow direction.

                .. math::

                    vLineP_{ij,t} \leq P^{\max}_{ij} \cdot soft + vLineOverload_{ij,t}

                """
                return model.vLineP[(i, j), t] <= model.pPmax_line[i, j] * model.pSoft_Line_Limit + model.vLineOverload[
                    (i, j), t]

            @model.Constraint(model.Branches, model.Time)
            def thermal_constraint_lower(model, i, j, t):
                r"""
                Soft thermal limit for active power flow on branch :math:`(i, j)` against flow direction.

                .. math::

                    vLineP_{ij,t} \geq -(P^{\max}_{ij} \cdot soft + vLineOverload_{ij,t})

                """
                return model.vLineP[(i, j), t] >= -(
                        model.pPmax_line[i, j] * model.pSoft_Line_Limit + model.vLineOverload[(i, j), t])
        else:
            @model.Constraint(model.Branches, model.Time)
            def thermal_constraint_upper(model, i, j, t):
                r"""
                Thermal limit constraint for active power flow on branch :math:`(i, j)`.

                Ensures that the active power flow does not exceed the thermal capacity of the line:

                .. math::

                    vLineP_{ij,t} \leq P^{\max}_{ij}

                """
                return model.vLineP[(i, j), t] <= model.pPmax_line[i, j]

            @model.Constraint(model.Branches, model.Time)
            def thermal_constraint_lower(model, i, j, t):
                r"""
                Thermal limit constraint for active power flow against flow direction.

                .. math::

                    vLineP_{ij,t} \geq -P^{\max}_{ij}

                """
                return model.vLineP[(i, j), t] >= -model.pPmax_line[i, j]

        @model.Constraint(model.Time)
        def energy_import_constraint(model, t):
            r"""
            Power balance at the slack bus.

            Ensures that the net import equals the sum of outgoing branch flows:

            .. math::

                vImpP_t - vExpP_t = \sum_{j \in \delta(slack)} vLineP_{slack,j,t}

            """
            return model.vImpP[t] - model.vExpP[t] == sum(
                model.vLineP[(model.slack_bus, j), t] for j in h.delta(model.slack_bus, model)
            )

        # --------------------------------------------
        # Battery equations
        # --------------------------------------------
        @model.Constraint(model.Buses, model.Time)
        def soc_batt_combined(model, i, t):
            r"""
            Battery state-of-charge (SoC) evolution.

            - At the first time step, the SoC is initialized as a fraction of battery capacity.
            - For subsequent time steps, the SoC evolves based on charging and discharging.

            .. math::

                SoC_{i,0} = pBus\_batt\_cap_i \cdot pBatt\_SoC\_init

                SoC_{i,t} = SoC_{i,t-1}
                           + \eta_{ch} \cdot Ch_{i,t-1}
                           - \frac{Dch_{i,t-1}}{\eta_{dch}}, \quad t > 0
            """
            if t == model.Time.first():
                return model.vSoC_Batt[i, t] == model.pBus_batt_cap[i] * model.pBatt_SoC_init
            else:
                return (
                        model.vSoC_Batt[i, t] ==
                        model.vSoC_Batt[i, t - 1]
                        + model.vCh_Batt[i, t - 1] * model.pEta_ch_batt
                        - model.vDch_Batt[i, t - 1] / model.pEta_dch_batt
                )

        # Note: The following constraints ensure that batteries can only charge or discharge at a given time but the addition
        # of integer variables may highly increase computational time. Consider disabling them for larger case studies or solving as an rMIP.
        @model.Constraint(model.Buses, model.Time)
        def discharge_batt_mode_link(model, i, t):
            r"""
            Prevent simultaneous charging and discharging (Big-M link for discharging mode).
            Consider disabling for larger case studies or solving as an rMIP if computational time is too high.

            .. math::

                Dch_{i,t} \leq M \cdot vbCh_{i,t} \cdot pBus\_batt\_cap_i
            """
            return model.vDch_Batt[i, t] <= model.pBigM * model.vbCh_Batt[i, t] * model.pBus_batt_cap[i]

        @model.Constraint(model.Buses, model.Time)
        def charge_batt_mode_link(model, i, t):
            r"""
            Prevent simultaneous charging and discharging (Big-M link for charging mode).
            Consider disabling for larger case studies or solving as an rMIP if computational time is too high.

            .. math::

                Ch_{i,t} \leq M \cdot (1 - vbCh_{i,t}) \cdot pBus\_batt\_cap_i
            """
            return model.vCh_Batt[i, t] <= model.pBigM * (1 - model.vbCh_Batt[i, t]) * model.pBus_batt_cap[i]

        @model.Constraint(model.Buses, model.Time)
        def soc_batt_max_constraint(model, i, t):
            r"""
            Maximum SoC constraint (cannot exceed battery capacity).

            .. math::

                SoC_{i,t} \leq pBus\_batt\_cap_i
            """
            return model.vSoC_Batt[i, t] <= model.pBus_batt_cap[i]

        @model.Constraint(model.Buses, model.Time)
        def soc_batt_min_constraint(model, i, t):
            r"""
            Minimum SoC constraint (cannot fall below allowed depth of discharge).

            .. math::

                SoC_{i,t} \geq pBatt\_SoC\_min \cdot pBus\_batt\_cap_i
            """
            return model.vSoC_Batt[i, t] >= model.pBatt_SoC_min * model.pBus_batt_cap[i]

        @model.Constraint(model.Buses, model.Time)
        def charge_batt_limit_upper_constraint(model, i, t):
            r"""
            Upper limit on charging power.

            .. math::

                Ch_{i,t} \leq C\_rate \cdot pBus\_batt\_cap_i
            """
            return model.vCh_Batt[i, t] <= model.pC_rate_batt * model.pBus_batt_cap[i]

        @model.Constraint(model.Buses, model.Time)
        def dcharge_batt_limit_upper_constraint(model, i, t):
            r"""
            Upper limit on discharging power.

            .. math::

                Dch_{i,t} \leq C\_rate \cdot pBus\_batt\_cap_i
            """
            return model.vDch_Batt[i, t] <= model.pC_rate_batt * model.pBus_batt_cap[i]

        @model.Constraint(model.Buses)
        def soc_batt_cyclic_constraint(model, i):
            r"""
            Cyclic SoC constraint for decentralized batteries.

            Forces the battery state of charge at the end of the horizon
            to equal the initial SoC, preventing end-of-horizon discharging.

            .. math::

                SoC_{i,T} + \eta_{ch} \cdot Ch_{i,T} - \frac{Dch_{i,T}}{\eta_{dch}}
                = pBus\_batt\_cap_i \cdot pBatt\_SoC\_init
            """
            t_last = model.Time.last()
            return (model.vSoC_Batt[i, t_last]
                    + model.vCh_Batt[i, t_last] * model.pEta_ch_batt
                    - model.vDch_Batt[i, t_last] / model.pEta_dch_batt
                    == model.pBus_batt_cap[i] * model.pBatt_SoC_init)

        # --------------------------------------------
        # EV equations
        # --------------------------------------------
        @model.Constraint(model.EVs, model.Time)
        def soc_departure_constraint(model, ev, t):
            r"""
            Ensure EV SoC meets required departure SoC.

            .. math::

                SoC_{ev,t} \geq SoC^{final}_{ev,t}
            """
            return model.vSoC_EV[ev, t] >= model.pSoC_final[ev, t]

        @model.Constraint(model.EVs, model.Time)
        def soc_charge_constraint(model, ev, t):
            r"""
            EV SoC evolution over time.

            - At the first time step, SoC is initialized based on initial SoC fraction:

            .. math::

                SoC_{ev,0} = pEV\_SoC\_init_{ev} \cdot pEV\_SoC\_max_{ev}

            - For subsequent time steps, SoC evolves considering charging, discharging,
              and slack variables to guarantee solvability:

            .. math::

                SoC_{ev,t} = SoC_{ev,t-1}
                            + \eta_{ch,ev} \cdot Ch_{ev,t-1}
                            - \frac{Dch_{ev,t-1}}{\eta_{dch,ev}}
                            + SoC^n_{ev,t-1} - SoC^p_{ev,t-1} - SoC^{final}_{ev,t-1}, \quad t > 0
            """
            if t == model.Time.first():
                return model.vSoC_EV[ev, t] == model.pEV_SoC_init[ev] * model.pEV_SoC_max[ev]
            else:
                return (
                        model.vSoC_EV[ev, t] ==
                        model.vSoC_EV[ev, t - 1]
                        + model.vCh_EV[ev, t - 1] * model.pEta_ch[ev]
                        - model.vDch_EV[ev, t - 1] / model.pEta_dch[ev]
                        + model.vSoC_n[ev, t - 1]
                        - model.vSoC_p[ev, t - 1]
                        - model.pSoC_final[ev, t - 1]
                )

        @model.Constraint(model.EVs, model.Time)
        def charge_limit_upper_constraint(model, ev, t):
            r"""
            Limit EV charging power at each time and bus.

            .. math::

                Ch_{ev,t} \leq Ch^{max}_{ev} \cdot \sum_{i \in Buses} Available_{ev,i,t}
            """
            return model.vCh_EV[ev, t] <= model.pEVCh_max[ev] * sum(model.pAvailable[ev, i, t] for i in model.Buses)

        @model.Constraint(model.EVs, model.Time)
        def discharge_limit_upper_constraint(model, ev, t):
            r"""
            Limit EV discharging power at each time and bus.

            .. math::

                Dch_{ev,t} \leq Dch^{max}_{ev} \cdot \sum_{i \in Buses} Available_{ev,i,t}
            """
            return model.vDch_EV[ev, t] <= model.pEVDch_max[ev] * sum(model.pAvailable[ev, i, t] for i in model.Buses)

        @model.Constraint(model.EVs, model.Time)
        def soc_min_constraint(model, ev, t):
            return model.vDch_EV[ev, t] <= model.pEta_dch[ev] * (model.vSoC_EV[ev, t] - (1-model.pDoD_V2G) * model.pEV_SoC_max[ev])

        @model.Constraint(model.EVs, model.Time)
        def soc_max_constraint(model, ev, t):
            r"""
            Limit maximum SoC of EVs.

            .. math::

                SoC_{ev,t} \leq SoC^{max}_{ev}
            """
            return model.vSoC_EV[ev, t] <= model.pEV_SoC_max[ev]

        @model.Constraint()
        def ev_charging_invest_constraint(model):
            r"""
            Limit total investment in EV charging capacities.

            .. math::

                \sum_{i \in Buses} ChCap_{i} \leq pCharge\_Invest \cdot |EVs|
            """
            return sum(model.vCh_cap_EV[i] for i in model.Buses) <= model.pCharge_Invest * len(model.EVs)

        @model.Constraint(model.Buses, model.Time)
        def charging_capacity_constraint(model, i, t):
            r"""
            Ensure node-level charging capacity is respected.

            .. math::

                \sum_{ev \in EVs} (Ch_{ev,t} + Dch_{ev,t}) \cdot Available_{ev,i,t} \leq ChCap_i
            """
            available_evs = self.evs_at_node_time.get((i, t), [])
            if not available_evs:
                return Constraint.Skip
            return sum(model.vCh_EV[ev, t] + model.vDch_EV[ev, t] for ev in available_evs) <= model.vCh_cap_EV[i]

        # Note: The following constraints ensure that EVs can only charge or discharge at a given time but the addition
        # of integer variables may highly increase computational time. Consider disabling them for larger case studies or solving as an rMIP.
        @model.Constraint(model.EVs, model.Time)
        def discharge_mode_link_per_bus(model, ev, t):
            r"""
            Prevent simultaneous charging and discharging for EVs (Big-M link for discharging mode).
            Consider disabling for larger case studies or solving as an rMIP if computational time is too high.

            .. math::

                Dch_{ev,t} \leq M \cdot vbCh_{ev,t}
            """
            return model.vDch_EV[ev, t] <= model.pBigM * model.vbCh_EV[ev, t]

        @model.Constraint(model.EVs, model.Time)
        def charge_mode_link_per_bus(model, ev, t):
            r"""
            Prevent simultaneous charging and discharging for Evs (link for charging mode).
            Consider disabling for larger case studies or solving as an rMIP if computational time is too high.

            .. math::

                Ch_{ev,t} \leq M \cdot (1 - vbCh_{ev,t})
            """
            return model.vCh_EV[ev, t] <= model.pBigM * (1 - model.vbCh_EV[ev, t])

        if self.ev_type == "V2G":

            @model.Constraint(model.EVs)
            def soc_v2g_cyclic_constraint(model, ev):
                r"""
                Cyclic SoC constraint for V2G-capable EVs.

                Forces each EV's state of charge at the end of the horizon to equal its
                initial SoC, preventing end-of-horizon discharging.

                ``vSoC_EV[ev, t_last]`` reflects the SoC evolution only up through
                ``t_last - 1``'s charge/discharge (see ``soc_charge_constraint``) - there is
                no ``t_last + 1`` for the recursion to debit ``t_last``'s own action against.
                Without including that last action here explicitly, it would be entirely free
                (helps the objective, invisible to every SoC constraint), defeating the
                constraint's purpose.

                .. math::

                    SoC_{ev,T} + \eta_{ch,ev} \cdot Ch_{ev,T} - \frac{Dch_{ev,T}}{\eta_{dch,ev}}
                    = pEV\_SoC\_init_{ev} \cdot pEV\_SoC\_max_{ev}
                """
                t_last = model.Time.last()
                return (model.vSoC_EV[ev, t_last]
                        + model.vCh_EV[ev, t_last] * model.pEta_ch[ev]
                        - model.vDch_EV[ev, t_last] / model.pEta_dch[ev]
                        == model.pEV_SoC_init[ev] * model.pEV_SoC_max[ev])

        # --------------------------------------------
        # Generator Equations
        # --------------------------------------------
        @model.Constraint(model.PV_Buses, model.Time)
        def pv_max_active_production_constraint(model, PV_Buses, t):
            r"""
                Maximum active power production of PV generators.

                The output is limited by the installed capacity and the time-dependent capacity factor:

                .. math::

                    P^{PV}_{bus,t} = P^{PV,max}_{bus} \cdot CF^{PV}_t
                """
            return model.vProd_PV[PV_Buses, t] == model.PV_Pmax[PV_Buses] * model.pCf_Pv[t]

        @model.Constraint(model.Wind_Buses, model.Time)
        def wind_max_active_production_constraint(model, Wind_Buses, t):
            r"""
                Maximum active power production of wind generators.

                The output is limited by the installed capacity and the time-dependent capacity factor:

                .. math::

                    P^{Wind}_{bus,t} = P^{Wind,max}_{bus} \cdot CF^{Wind}_t
                """
            return model.vProd_Wind[Wind_Buses, t] == model.Wind_Pmax[Wind_Buses] * model.pCf_Wind[t]

        @model.Constraint(model.Hydro_Buses, model.Time)
        def hydro_max_active_production_constraint(model, Hydro_Buses, t):
            r"""
                Maximum active power production of hydro generators.

                The output is limited by the installed capacity and the time-dependent capacity factor:

                .. math::

                    P^{Hydro}_{bus,t} = P^{Hydro,max}_{bus} \cdot CF^{Hydro}_t
                """
            return model.vProd_Hydro[Hydro_Buses, t] == model.Hydro_Pmax[Hydro_Buses] * model.pCf_Hydro[t]

        # --------------------------------------------
        # DC-OPF equations
        # --------------------------------------------
        if self.dc_opf:
            @model.Constraint(model.Buses, model.Time)
            def eDC_power_balance_active_constraint(model, i, t):
                r"""
                    Active power balance at each bus (DC-OPF) including slack variables.

                    For all non-slack buses, the sum of generation, storage, EV charging/discharging,
                    and net branch flows, plus slack variables, equals the demand:

                    .. math::

                        - P^{demand}_{i,t}
                        - \sum_{ev} Available_{ev,i,t} \cdot Ch_{ev,t}
                        + \sum_{ev} Available_{ev,i,t} \cdot Dch_{ev,t}
                        + P^{PV}_{i,t} + P^{Wind}_{i,t} + P^{Hydro}_{i,t}
                        - Ch^{Batt}_{i,t} + Dch^{Batt}_{i,t}
                        + P^{NS}_{i,t} - P^{ES}_{i,t}
                        = \sum_{j \in \delta(i)} P_{ij,t}

                    Here:

                    - `P^{NS}_{i,t}` = positive slack (under-supply)
                    - `P^{ES}_{i,t}` = negative slack (over-supply)

                    The slack bus is skipped to allow angle reference.
                    """
                if i == value(model.slack_bus):
                    return Constraint.Skip

                available_evs = self.evs_at_node_time.get((i, t), [])

                return (
                        - model.pDemand[t, i]
                        - sum(model.vCh_EV[ev, t] for ev in available_evs)
                        + sum(model.vDch_EV[ev, t] for ev in available_evs)
                        + (model.vProd_PV[i, t] if i in model.PV_Buses else 0)
                        + (model.vProd_Wind[i, t] if i in model.Wind_Buses else 0)
                        + (model.vProd_Hydro[i, t] if i in model.Hydro_Buses else 0)
                        - model.vCh_Batt[i, t]
                        + model.vDch_Batt[i, t]
                        + model.vPNS[i, t]
                        - model.vEPS[i, t]
                        == sum(model.vLineP[(i, j), t] for j in
                               h.delta(i, model)))  # vLineP is Positive for outgoing flow from i to j

            @model.Constraint(model.Branches, model.Time)
            def eDC_flow_equation(model, i, j, t):
                r"""
                Active power flow on a line (DC approximation).

                Power flow is proportional to the voltage angle difference over line reactance:

                .. math::

                    P_{ij,t} = \frac{1}{X_{ij}} \left( \theta_{i,t} - \theta_{j,t} \right)
                """
                return model.vLineP[(i, j), t] == 1 / model.pXLine[(i, j)] * (model.vTheta[i, t] - model.vTheta[j, t])

            @model.Constraint(model.Time)
            def eDC_fix_angle_constraint(model, t):
                r"""
                Fix voltage angle at the slack bus to 0.

                .. math::

                    \theta_{slack,t} = 0
                """
                return model.vTheta[model.slack_bus, t] == 0
        # --------------------------------------------
        # SOCP equations
        # --------------------------------------------
        if not self.dc_opf:

            @model.Constraint(model.Time)
            def energy_import_reactive_constraint(model, t):
                r"""
                Reactive power balance at the slack bus.

                Ensures that the net reactive power import equals the sum of outgoing branch reactive flows:

                .. math::

                    vImpQ_t = \sum_{j \in \delta(slack)} vLineQ_{slack,j,t}

                """
                return model.vImpQ[t] == sum(
                    model.vLineQ[(model.slack_bus, j), t] for j in h.delta(model.slack_bus, model)
                )

            @model.Constraint(model.Buses, model.Time)
            def power_balance_active_constraint(model, i, t):
                r"""
                Active power balance at each bus including battery, EVs, generation, and slack variables.

                .. math::

                    - P^{demand}_{i,t}
                    - \sum_{ev} Available_{ev,i,t} \cdot Ch_{ev,t}
                    + \sum_{ev} Available_{ev,i,t} \cdot Dch_{ev,t}
                    + P^{PV}_{i,t} + P^{Wind}_{i,t} + P^{Hydro}_{i,t}
                    + G_s[i] V_{ii,t}
                    - Ch^{Batt}_{i,t} + Dch^{Batt}_{i,t}
                    + P^{NS}_{i,t} - P^{ES}_{i,t}
                    = \sum_{j \in \delta(i)} P_{ij,t}
                """
                if i == value(model.slack_bus):
                    return Constraint.Skip

                available_evs = self.evs_at_node_time.get((i, t), [])
                return (
                        - model.pDemand[t, i]
                        - sum(model.vCh_EV[ev, t] for ev in available_evs)
                        + sum(model.vDch_EV[ev, t] for ev in available_evs)
                        + (model.vProd_PV[i, t] if i in model.PV_Buses else 0)
                        + (model.vProd_Wind[i, t] if i in model.Wind_Buses else 0)
                        + (model.vProd_Hydro[i, t] if i in model.Hydro_Buses else 0)
                        + model.Gs[i] * model.vCii[i, t]
                        - model.vCh_Batt[i, t]
                        + model.vDch_Batt[i, t]
                        + model.vPNS[i, t]
                        - model.vEPS[i, t]
                        == sum([model.vLineP[(i, j), t] for j in h.delta(i, model)])
                )

            @model.Constraint(model.Buses, model.Time)
            def power_balance_reactive_constraint(model, i, t):
                r"""
                Reactive power balance at each bus.

                .. math::

                    - Q^{demand}_{i,t}
                    + B_s[i] \cdot V_{ii,t}
                    = \sum_{j \in \delta(i)} Q_{ij,t}
                """
                if i != value(model.slack_bus):
                    return (
                            - model.pQfactor * model.pDemand[t, i]
                            + model.Bs[i] * model.vCii[i, t]
                            == sum(model.vLineQ[(i, j), t]
                                   for j in h.delta(i, model)))
                else:
                    return Constraint.Skip

            # add voltage and angle limits
            @model.Constraint(model.Branches, model.Time)
            def phase_angle_diff_min_constraint(model, i, j, t):
                r"""
                Minimum phase angle difference across a branch.

                .. math::

                    S_{ij,t} \leq C_{ij,t} \cdot \tan(\theta^{max}_{ij})
                """
                return model.vSij[i, j, t] <= model.vCij[i, j, t] * math.tan(
                    value(model.pMax_AngDiff[i, j]))

            @model.Constraint(model.Branches, model.Time)
            def phase_angle_diff_max_constraint(model, i, j, t):
                r"""
                 Maximum phase angle difference across a branch.

                 .. math::

                     S_{ij,t} \geq - C_{ij,t} \cdot \tan(\theta^{max}_{ij})
                 """
                return model.vSij[i, j, t] >= -model.vCij[i, j, t] * math.tan(
                    value(model.pMax_AngDiff[i, j]))

            if self.pEnable_Soft_Voltage_Limits:

                @model.Constraint(model.Buses, model.Time)
                def voltage_soft_limit_min(model, i, t):
                    if i == model.slack_bus: return Constraint.Skip
                    # vVoltage_Undershoot >= (Vmin+delta)^2 - vCii  (zero if not violated)
                    return model.vVoltage_Undershoot[i, t] >= (model.pVmin[i] + model.pSoft_Voltage_Limit) ** 2 - model.vCii[i, t]

                @model.Constraint(model.Buses, model.Time)
                def voltage_soft_limit_max(model, i, t):
                    if i == model.slack_bus: return Constraint.Skip
                    # vVoltage_Overshoot >= vCii - (Vmax-delta)^2  (zero if not violated)
                    return model.vVoltage_Overshoot[i, t] >= model.vCii[i, t] - (
                            model.pVmax[i] - model.pSoft_Voltage_Limit) ** 2
            else:
                @model.Constraint(model.Buses, model.Time)
                def voltage_limit_min(model, i, t):
                    r"""
                    Minimum squared voltage magnitude at a bus.

                    .. math::

                        V_{min,i}^2 \leq V_{ii,t}
                    """
                    if i != value(model.slack_bus):
                        return (model.pVmin[i] ** 2 - model.vVoltage_Undershoot[i, t] <= model.vCii[i, t])
                    else:
                        return Constraint.Skip

            @model.Constraint(model.Buses, model.Time)
            def voltage_limit_max(model, i, t):
                r"""
                Maximum squared voltage magnitude at a bus.

                .. math::

                    V_{ii,t} \leq V_{max,i}^2
                """
                if i != value(model.slack_bus):
                    return (model.vCii[i, t] <= model.pVmax[i] ** 2 + model.vVoltage_Overshoot[i, t])
                else:
                    return Constraint.Skip

            @model.Constraint(model.Branches, model.Time)
            def power_flow_active_constraint(model, i, j, t):
                r"""
                Active power flow on a branch (SOCP formulation).

                .. math::

                    P_{ij,t} = G_{ij} \cdot V_{ii,t} - G_{ij} \cdot C_{ij,t} + B_{ij} \cdot S_{ij,t}
                """
                return (model.vLineP[(i, j), t] == model.pGLine[i, j]
                        * model.vCii[i, t] - model.pGLine[i, j] * model.vCij[i, j, t]
                        + model.pBLine[i, j] * model.vSij[i, j, t])

            @model.Constraint(model.Branches, model.Time)
            def power_flow_reactive_constraint(model, i, j, t):
                r"""
                Reactive power flow on a branch (SOCP formulation).

                .. math::

                    Q_{ij,t} = - B_{ij} \cdot V_{ii,t} + B_{ij} \cdot C_{ij,t} + G_{ij} \cdot S_{ij,t}
                """
                return (model.vLineQ[(i, j), t] == - model.pBLine[i, j]
                        * model.vCii[i, t] + model.pBLine[i, j] * model.vCij[i, j, t]
                        + model.pGLine[i, j] * model.vSij[i, j, t])

            @model.Constraint(model.Time)
            def fix_voltage_constraint(model, t):
                r"""
                Fix the quadratic voltage magnitude at the slack bus.

                .. math::

                    V_{slack,t} = V_{slack}^{set}
                """
                return model.vCii[model.slack_bus, t] == (value(model.pV_slack)) ** 2

            # NOTE: These SOCP constraints are defined for the branches, not the buses

            # Second-order cone programming (SOCP) constraint
            @model.Constraint(model.Branches, model.Time)
            def socp_constraint(model, i, j, t):
                r"""
                 Second-order cone constraint for branch voltages (SOCP).

                 .. math::

                     C_{ij,t}^2 + S_{ij,t}^2 \leq V_{ii,t} \cdot V_{jj,t}
                 """
                return (model.vCij[i, j, t] ** 2 + model.vSij[i, j, t] ** 2
                        <= model.vCii[i, t] * model.vCii[j, t])

            # Symmetry and asymmetry constraints
            @model.Constraint(model.Branches, model.Time)
            def cij_symmetry_constraint(model, i, j, t):
                r"""
                Symmetry of the C_{ij} term across branches.

                .. math::

                    C_{ij,t} = C_{ji,t}
                """
                return model.vCij[i, j, t] == model.vCij[j, i, t]

            @model.Constraint(model.Branches, model.Time)
            def sij_asymmetry_constraint(model, i, j, t):
                r"""
                Anti-symmetry of the S_{ij} term across branches.

                .. math::

                    S_{ij,t} = -S_{ji,t}
                """
                return model.vSij[i, j, t] == -model.vSij[j, i, t]

            if self.pEnable_SOCP_Exactness_Penalty:
                @model.Constraint(model.CanonicalBranches, model.Time)
                def socp_exactness_constraint(model, i, j, t):
                    r"""
                    Convex majorant tightening the SOCP relaxation gap toward exactness.

                    The relaxation gap :math:`vCii_{i,t} \cdot vCii_{j,t} - vCij_{ij,t}^2 - vSij_{ij,t}^2`
                    (always :math:`\geq 0` by ``socp_constraint``) is bilinear and therefore nonconvex.
                    Using :math:`xy = \tfrac{1}{4}[(x+y)^2-(x-y)^2]`, the gap decomposes as a convex
                    term minus a convex term; linearizing the subtracted (convex) term at the previous
                    solved point :math:`(x_0, y_0, c_0, s_0)` yields a convex function that is a
                    *global* upper bound on the true gap (a first-order tangent plane always lies
                    below a convex function). Wherever the slack :math:`\sigma` is zero, the true
                    relaxation gap is therefore also :math:`\leq \varepsilon`.

                    The majorant equals the true gap plus a squared distance to the linearization
                    point, so a hard cap acts as a trust region and can be infeasible far from an
                    exact point. The slack :math:`\sigma \geq 0` (penalized with ``pCCP_Tau`` in the
                    objective) keeps every iteration feasible; :meth:`solve_model` re-linearizes at
                    each new solution and raises ``pCCP_Tau`` (penalty CCP, Lipp & Boyd 2016) until
                    the true gap is within :math:`\varepsilon` everywhere.

                    .. math::

                        \frac{(x+y)^2}{4} - \left[
                            \frac{(x_0-y_0)^2}{4} + c_0^2 + s_0^2
                            + \frac{x_0-y_0}{2}(x-x_0) - \frac{x_0-y_0}{2}(y-y_0)
                            + 2c_0(c-c_0) + 2s_0(s-s_0)
                        \right] \leq \varepsilon + \sigma_{ij,t}
                    """
                    x, y = model.vCii[i, t], model.vCii[j, t]
                    c, s = model.vCij[i, j, t], model.vSij[i, j, t]
                    x0, y0 = model.pCii_lin[i, t], model.pCii_lin[j, t]
                    c0, s0 = model.pCij_lin[i, j, t], model.pSij_lin[i, j, t]

                    b_lin = ((x0 - y0) ** 2 / 4 + c0 ** 2 + s0 ** 2
                             + (x0 - y0) / 2 * (x - x0) - (x0 - y0) / 2 * (y - y0)
                             + 2 * c0 * (c - c0) + 2 * s0 * (s - s0))

                    return (x + y) ** 2 / 4 - b_lin <= model.pSOCP_Exactness_Epsilon + model.vSOCP_Slack[i, j, t]

                # Inactive until a first (untightened) solve provides the linearization point.
                model.socp_exactness_constraint.deactivate()
        # --------------------------------------------
        # Cost expressions and Objective-function
        # --------------------------------------------
        # The small export reward makes exporting strictly better than dissipating surplus
        # power in relaxed line losses, which would otherwise be free.
        model.cost_function = Expression(
            expr=sum(model.vImpP[t] * model.pElPrice[t] - model.vExpP[t] * self.pExport_Reward for t in model.Time))

        model.ev_energy_cost = Expression(
            expr=sum(model.pElPrice[t] * (model.vCh_EV[ev, t] - model.vDch_EV[ev, t])
                     for ev in model.EVs for t in model.Time))

        model.ens_cost = Expression(
            expr=(sum(model.vSoC_n[ev, t] + model.vSoC_p[ev, t] for ev in model.EVs for t in model.Time)
                  + sum(model.vPNS[i, t] + model.vEPS[i, t] for i in model.Buses for t in model.Time)
                  ) * model.pPenality)

        model.charging_investment_cost = Expression(
            expr=sum(model.vCh_cap_EV[i] for i in model.Buses) * model.pCh_Inf_Cost)

        if self.pEnable_Soft_Line_Limits:
            model.soft_line_limits_penalty = Expression(
                expr=(
                        sum(model.vLineOverload[(i, j), t] for (i, j) in model.Branches for t in
                            model.Time) * self.pSoft_Line_Penalty_Factor * model.pPenality
                )
            )
        else:
            model.soft_line_limits_penalty = Expression(
                expr=0
            )

        if self.pEnable_Soft_Voltage_Limits:
            model.soft_voltage_limits_penalty = Expression(
                expr=(
                        sum(model.vVoltage_Undershoot[i, t] + model.vVoltage_Overshoot[i, t]
                            for i in model.Buses for t in model.Time) * self.pSoft_Voltage_Penalty_Factor * model.pPenality
                )
            )
        else:
            model.soft_voltage_limits_penalty = Expression(
                expr=0
            )

        # Penalty-CCP term; zero at the optimum while socp_exactness_constraint is inactive.
        # Not a system cost - subtract it from obj when reporting costs.
        if not self.dc_opf and self.pEnable_SOCP_Exactness_Penalty:
            model.socp_ccp_penalty = Expression(
                expr=model.pCCP_Tau * sum(model.vSOCP_Slack[i, j, t]
                                          for (i, j) in model.CanonicalBranches for t in model.Time)
            )
        else:
            model.socp_ccp_penalty = Expression(
                expr=0
            )

        model.obj = Objective(
            expr=model.cost_function
                 + model.ens_cost
                 + model.charging_investment_cost
                 + model.ev_energy_cost
                 + model.soft_line_limits_penalty
                 + model.soft_voltage_limits_penalty
                 + model.socp_ccp_penalty
            , sense=minimize,
            doc=r"""
                Total system cost objective, including electricity import, EV energy costs,
                Energy Not Served (ENS), and charging infrastructure investment.

                The objective function is the sum of four components:

                1. **Electricity import cost minus export reward:**

                    .. math::

                        Cost_{import} = \sum_{t \in T} \left( P^{imp}_t \cdot Price_t - P^{exp}_t \cdot r_{exp} \right)

                    where:
                    - \(P^{imp}_t\) / \(P^{exp}_t\) = imported / exported electricity at time t
                    - \(Price_t\) = electricity price at time t
                    - \(r_{exp}\) = small export reward (``pExport_Reward``) that makes exporting
                      strictly better than dissipating surplus power in relaxed line losses

                2. **EV energy cost (charging minus discharging):**

                    .. math::

                        Cost_{EV} = \sum_{ev \in EVs} \sum_{t \in T} Price_t \cdot (Ch_{ev,t} - Dch_{ev,t})

                    where:
                    - \(Ch_{ev,t}\) = charging power of EV ev at time t
                    - \(Dch_{ev,t}\) = discharging power of EV ev at time t
                    - \(Price_t\) = electricity price at time t

                3. **Power Not Served (PNS) and Excess Power Served (EPS) cost:**

                    .. math::

                        Cost_{ENS} = p_{pen} \cdot \Bigg( 
                            \sum_{ev \in EVs} \sum_{t \in T} (SoC^{neg}_{ev,t} + SoC^{pos}_{ev,t}) +
                            \sum_{i \in Buses} \sum_{t \in T} (P^{NS}_{i,t} + P^{ES}_{i,t})
                        \Bigg)

                    where:
                    - \(SoC^{neg}_{ev,t}\) / \(SoC^{pos}_{ev,t}\) = negative/positive EV slack at time t
                    - \(P^{NS}_{i,t}\) / \(P^{ES}_{i,t}\) = negative/positive power slack at bus i, time t
                    - \(SoC^{neg}_{ev,t}) / \ SoC^{pos}_{ev,t}) = negative/positive EV slack at time t
                    - \(p_{pen}\) = penalty factor for ENS

                4. **Charging station investment cost:**

                    .. math::

                        Cost_{EV\_invest} = p_{Ch\_cost} \cdot \sum_{i \in Buses} ChCap_i

                    where:
                    - \(ChCap_i\) = installed charging capacity at bus i
                    - \(p_{Ch\_cost}\) = cost per unit of charging capacity in MW

                5. **Soft line loading limit violation penalty (optional):**

                .. math::

                    Cost_{line} =
                    \alpha_{line} \cdot p_{pen}
                    \cdot \sum_{(i,j) \in Branches} \sum_{t \in T} Overload_{(i,j),t}

                where:
                - \(Overload_{(i,j),t}\) = line loading slack variable for branch (i,j) at time t
                - \(\alpha_{line}\) = weighting factor for line limit violations (``Line_soft_penalty_factor``)
                - \(p_{pen}\) = global penalty factor

                This term is only active if `pEnable_Soft_Line_Limits = True`.
                It penalizes deviation from the soft line thermal limits encouraging the model to stay within tighter physical line limits if possible.

                6. **Soft voltage limit violation penalty (optional):**

                    .. math::

                        Cost_{voltage} =
                        \alpha_{volt} \cdot p_{pen}
                        \cdot \sum_{i \in Buses} \sum_{t \in T}
                        (V^{under}_{i,t} + V^{over}_{i,t})

                    where:
                    - \(V^{under}_{i,t}\) = voltage undershoot slack at bus i, time t
                    - \(V^{over}_{i,t}\) = voltage overshoot slack at bus i, time t
                    - \(\alpha_{volt}\) = weighting factor for voltage violations (``V_soft_penalty_factor``)
                    - \(p_{pen}\) = global penalty factor

                    This term is only active if `pEnable_Soft_Voltage_Limits = True`.
                    It penalizes deviations from the soft voltage bounds encouraging the model to stay in tighter voltage bounds if possible.

                7. **Penalty-CCP slack (only while tightening SOCP exactness):**

                    .. math::

                        Cost_{CCP} = \tau \cdot \sum_{(i,j) \in L^{canon}} \sum_{t \in T} \sigma_{ij,t}

                    This term is an algorithmic device of the exactness tightening and is not a
                    system cost; reported costs exclude it.

                Overall, the objective function minimizes total system cost while allowing
                prioritized relaxation of network constraints via weighted soft penalties.
                            """
        )

        self.model = model

        print('Pyomo model created in', time.time() - start_time, 'seconds')

    def _report_socp_gap_stats(self, label):
        """
        Print summary statistics of the SOCP relaxation gap
        (``vCii_i*vCii_j - vCij**2 - vSij**2``) across all canonical branches and
        timesteps, using the model's currently solved variable values.

        Parameters
        ----------
        label : str
            Short tag included in the printed output (e.g. ``'before tightening'``).

        Returns
        -------
        int
            Number of branch-timesteps whose gap exceeds ``pSOCP_Exactness_Epsilon``.
        """
        model = self.model
        gaps = []
        worst = None
        for (i, j) in model.CanonicalBranches:
            for t in model.Time:
                gap = (value(model.vCii[i, t]) * value(model.vCii[j, t])
                       - value(model.vCij[i, j, t]) ** 2 - value(model.vSij[i, j, t]) ** 2)
                gaps.append(gap)
                if worst is None or gap > worst[0]:
                    worst = (gap, i, j, t)

        n_over = sum(1 for g in gaps if g > value(model.pSOCP_Exactness_Epsilon))
        print(f"SOCP relaxation gap ({label}): "
              f"min={min(gaps):.6g}, mean={sum(gaps) / len(gaps):.6g}, max={worst[0]:.6g} "
              f"at branch ({worst[1]}, {worst[2]}), t={worst[3]}; "
              f"{n_over}/{len(gaps)} branch-timesteps exceed epsilon="
              f"{value(model.pSOCP_Exactness_Epsilon):.6g}")
        return n_over

    def _solve_ok(self):
        """Return True if the last solve terminated optimal."""
        return (self.results.solver.status == SolverStatus.ok
                and self.results.solver.termination_condition == TerminationCondition.optimal)

    def _run_solver(self):
        """
        Run a single solve of ``self.model`` and report solver status.

        A fresh ``gurobi_persistent`` instance is built for every solve: a persistent
        solver does not pick up later changes to the Pyomo model (activated constraints,
        changed mutable parameters, relaxed integrality) on its own. Solver parameters
        are taken from ``Gurobi_options`` in ``config.yml``.

        An optimal solution is always loaded into the model. A non-optimal solve
        (e.g. numerical trouble) still loads Gurobi's solution if one exists
        (``SolCount > 0``), so it can be exported for debugging or used as a CCP
        starting point.

        Returns
        -------
        bool
            True if a solution (optimal or not) was loaded into the model.
        """
        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(self.model, symbolic_solver_labels=True)
        gurobi_model = solver._solver_model
        for name, val in self.gurobi_options.items():
            gurobi_model.setParam(name, val)

        if self.enable_rMIP:
            print('Solving model as a relaxed MIP (rMIP)')
        else:
            print('Solving model as a MIP')
            gurobi_model.Params.MIPGap = self.MIP_Gap  # Set optimal MIP gap (e.g.: 0.0001 = 0.01%)
        self.results = solver.solve(self.model, tee=True, load_solutions=False)

        # Check solver status
        if self._solve_ok():
            self.model.solutions.load_from(self.results)
            print("Solver found an optimal solution.")
            print("Optimal cost (excl. SOCP CCP penalty): ",
                  value(self.model.obj) - value(self.model.socp_ccp_penalty))
            return True

        print("Solver did not find an optimal solution. Status:", self.results.solver.status)
        print("Solver termination condition:", self.results.solver.termination_condition)
        if gurobi_model.SolCount > 0:
            solver.load_vars()
            print("Loaded the non-optimal solution Gurobi still holds (SolCount > 0). Cost (excl. SOCP CCP penalty): ",
                  value(self.model.obj) - value(self.model.socp_ccp_penalty))
            return True
        return False

    def _run_socp_ccp(self):
        """
        Enforce SOCP relaxation exactness with the penalty convex-concave procedure
        (Lipp & Boyd 2016), starting from the already-solved first solution.

        In MIP mode the charge/discharge binaries are fixed to the first solution so
        that every iteration is a convex SOCP (and unfixed again at the end). Each
        iteration re-linearizes ``socp_exactness_constraint`` at the current solution,
        re-solves, and raises ``pCCP_Tau`` until the true gap is within epsilon
        everywhere, a solve fails, or ``CCP_max_iter`` is reached. If an iteration
        fails, the last good iterate is restored so it can still be exported.

        Returns
        -------
        tuple
            ``(n_over, failed_iter)`` - number of branch-timesteps still exceeding epsilon
            at the returned solution, and the iteration that failed (``None`` if none did).
        """
        model = self.model

        fixed_binaries = []
        if not self.enable_rMIP:
            for var in (model.vbCh_EV, model.vbCh_Batt):
                for index in var:
                    if not var[index].fixed and var[index].value is not None:
                        var[index].fix(round(var[index].value))
                        fixed_binaries.append(var[index])
            print(f'Fixed {len(fixed_binaries)} charge/discharge binaries for the SOCP CCP iterations')

        model.pCCP_Tau = self.pCCP_Tau_Init
        model.socp_exactness_constraint.activate()
        all_vars = list(model.component_data_objects(Var))
        n_over = None
        failed_iter = None

        for k in range(1, self.pCCP_Max_Iter + 1):
            # Snapshot of the last good iterate, restored if this iteration fails
            snapshot = [v.value for v in all_vars]

            for i in model.Buses:
                for t in model.Time:
                    model.pCii_lin[i, t] = value(model.vCii[i, t])
            for (i, j) in model.CanonicalBranches:
                for t in model.Time:
                    model.pCij_lin[i, j, t] = value(model.vCij[i, j, t])
                    model.pSij_lin[i, j, t] = value(model.vSij[i, j, t])

            print(f'SOCP CCP iteration {k}/{self.pCCP_Max_Iter}, tau={value(model.pCCP_Tau):.3g}')
            self._run_solver()
            if not self._solve_ok():
                for v, val in zip(all_vars, snapshot):
                    v.set_value(val, skip_validation=True)
                failed_iter = k
                print(f'SOCP CCP iteration {k} did not reach optimal - stopping CCP and restoring '
                      f'the last good iterate.')
                break

            slack_sum = sum(value(model.vSOCP_Slack[i, j, t])
                            for (i, j) in model.CanonicalBranches for t in model.Time)
            n_over = self._report_socp_gap_stats(f'CCP iteration {k}')
            print(f'SOCP CCP iteration {k}: slack sum={slack_sum:.6g}, cost (excl. CCP penalty)='
                  f'{value(model.obj) - value(model.socp_ccp_penalty):.6g}')

            if n_over == 0:
                print(f'SOCP CCP converged after {k} iteration(s): true gap within epsilon everywhere.')
                break

            model.pCCP_Tau = min(value(model.pCCP_Tau) * self.pCCP_Tau_Mult, self.pCCP_Tau_Max)

        if failed_iter is not None and n_over is None:
            # Failed in the first iteration: the restored solution is the initial one
            n_over = self._report_socp_gap_stats('restored initial solution')
        elif failed_iter is None and n_over > 0:
            print('WARNING: SOCP CCP did not reach exactness - the last solution still violates the '
                  'epsilon gap tolerance (see the gap statistics above).')

        for var in fixed_binaries:
            var.unfix()
        return n_over, failed_iter

    def solve_model(self):
        """
        Solve the Pyomo optimization model using the Gurobi solver.

        This method performs the following steps:

        1. Initializes the Gurobi solver via Pyomo.
        2. Solves the optimization model.
        3. Checks the solver's status and termination condition.
           - If the solution is optimal, prints the optimal cost.
           - Otherwise, prints solver status and termination information.
        4. If ``Enable_SOCP_Exactness_Penalty`` is set (SOCP formulation only) and the
           true relaxation gap exceeds ``SOCP_exactness_epsilon`` anywhere, runs a penalty
           convex-concave procedure (Lipp & Boyd 2016): in MIP mode the charge/discharge
           binaries are fixed to the first solution, ``socp_exactness_constraint`` is
           activated, and the model is re-linearized at each new solution and re-solved
           with ``pCCP_Tau`` multiplied by ``CCP_tau_mult`` (capped at ``CCP_tau_max``)
           until the true gap is within epsilon everywhere or ``CCP_max_iter`` is reached.
           The binaries are unfixed afterwards. See the constraint's docstring in
           :meth:`create_model` for the derivation. A first solve that is not optimal but
           returns a solution is used as the starting point as well.
        5. Sets ``run_status`` to one of ``exact``, ``inexact_max_iter``,
           ``initial_suboptimal``, ``ccp_failed_iter_<k>``, ``no_solution`` or
           ``exactness_not_checked``.
        6. Iterates through the model's variables and optionally prints their values.

        Notes
        -----
        - Requires that the model has been created using :meth:`create_model`.
        - Solver options such as MIP gap are taken from instance attributes.
        - Results are stored in the instance attribute ``results``.

        Returns
        -------
        None
        """

        start_time = time.time()

        # Relax integrality before any solver instance is built (see _run_solver)
        if self.enable_rMIP:
            TransformationFactory('core.relax_integer_vars').apply_to(self.model)

        self.solution_loaded = self._run_solver()
        first_solve_ok = self._solve_ok()
        n_over = None
        failed_iter = None

        if not self.solution_loaded:
            print('First solve returned no solution - skipping SOCP exactness tightening '
                  '(nothing to linearize around).')
        elif not self.dc_opf and self.pEnable_SOCP_Exactness_Penalty:
            if not first_solve_ok:
                print('First solve did not reach optimal but returned a solution - continuing the '
                      'SOCP exactness tightening from this (suboptimal) starting point.')
            n_over = self._report_socp_gap_stats('before tightening')

            if n_over == 0:
                print('SOCP relaxation is already within epsilon everywhere - skipping tightening re-solve.')
            else:
                n_over, failed_iter = self._run_socp_ccp()

        # Overall solve outcome
        if not self.solution_loaded:
            self.run_status = 'no_solution'
        elif failed_iter is not None:
            self.run_status = f'ccp_failed_iter_{failed_iter}'
        elif not first_solve_ok:
            self.run_status = 'initial_suboptimal'
        elif n_over is None:
            self.run_status = 'exactness_not_checked'
        elif n_over > 0:
            self.run_status = 'inexact_max_iter'
        else:
            self.run_status = 'exact'

        # Check variable values
        for var in self.model.component_objects(Var, active=False):
            for index in var:
                print(f"Variable {var.name}:")
                print(f"  {index} = {var[index].value}")
        print(f'Pyomo model solved in {time.time() - start_time} seconds - run status: {self.run_status}')

    def export_results(self):
        """
        Export the results of the solved Pyomo model to a SQLite database.

        This method performs the following steps:

        1. Saves the model's results to a SQLite database file named ``result.sqlite``
           in the ``output`` directory.
        2. If the file already exists, it removes all existing tables before saving
           the new results.
        3. Maps model variables and solver outputs into structured tables.
        4. If ``ev_type == "UC"``, additionally writes the precomputed per-EV
           charging detail, per-node charging totals, and charging cost (known ahead
           of the model run, since the optimizer only ever sees a single inert
           placeholder vehicle under UC) into ``uc_ev_charging``,
           ``uc_node_charging_total``, and ``uc_charging_cost`` tables in the same
           database - see :meth:`SystemData._apply_uc_dumb_charging_demand`.

        Raises
        ------
        AttributeError
            If ``results`` are not available (i.e., the model has not been solved yet),
            or if no solution was loaded. Non-optimal solutions (numerical trouble,
            failed CCP iteration) are still exported, since they help finding errors;
            ``run_status`` tells them apart.

        Returns
        -------
        None
        """

        if self.results is None:
            raise AttributeError("No results available to export - solve_model() has not run yet.")
        if not self.solution_loaded:
            raise AttributeError(
                f"Cannot export results - no solution was loaded "
                f"(status={self.results.solver.status}, "
                f"termination_condition={self.results.solver.termination_condition})."
            )
        if self.run_status not in ('exact', 'exactness_not_checked'):
            print(f"WARNING: exporting a solution with run status '{self.run_status}'.")

        filename = os.path.join("output", "result.sqlite")

        # Clean old SQL data
        if os.path.exists(filename):
            with sqlite3.connect(filename) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                for (table_name,) in cursor.fetchall():
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()

        dh.model_to_sqlite(self.model, filename)

        # Precomputed UC EV-charging results (known ahead of the model run, since the
        # optimizer only ever sees a single inert placeholder vehicle under UC - see
        # SystemData._apply_uc_dumb_charging_demand). Written under distinct table
        # names, after model_to_sqlite so the cleanup above doesn't wipe them and they
        # don't collide with the model's own (trivial, dummy-only) vCh_EV/vDch_EV/
        # vSoC_EV tables.
        if self.ev_type == "UC" and self.uc_ev_charging_df is not None:
            with sqlite3.connect(filename) as conn:
                self.uc_ev_charging_df.to_sql('uc_ev_charging', conn, if_exists='replace', index=False)
                self.uc_node_charging_total_df.to_sql('uc_node_charging_total', conn, if_exists='replace', index=False)
                self.uc_charging_cost_df.to_sql('uc_charging_cost', conn, if_exists='replace', index=False)
            print('UC EV results written to result.sqlite '
                  '(uc_ev_charging, uc_node_charging_total, uc_charging_cost)')

    def plot_network(self, input_only=False):
        """
        Plot the network based on the provided data and configuration.

        With ``input_only=True`` the grid topology (buses and branches) is shown on a map
        with plotly. Otherwise the solved results are opened in the interactive
        visualization tool LEGO-GPT (V2G-QUESTS-PSOM version, branch ``V2G-PSOM`` of
        https://github.com/IEE-TUGraz/LEGO-GPT), which reads ``output/result.sqlite``.

        LEGO-GPT runs in its own conda environment: ``LEGO_GPT_path`` in ``config.yml``
        points to ``LEGO_GPT.py`` and ``LEGO_GPT_env_path`` to the Python interpreter of
        the environment created from LEGO-GPT's ``environment.yml``. LEGO-GPT is started
        with its own folder as working directory and reads ``plot_config.yml``; a relative
        ``input_dir`` there (default ``output``) is resolved against the V2G-QUESTS-PSOM
        folder. The call blocks until the LEGO-GPT server is stopped (Ctrl+C).

        Parameters
        ----------
        input_only : bool, optional
            If True, only the input data (buses, branches, and nodes) are plotted.
            If False, the method requires the model to be solved and opens the results
            in LEGO-GPT. Defaults to False.

        Raises
        ------
        ValueError
            If ``input_only`` is False and the model has not been solved yet
            (i.e., ``self.results`` is None).

        Returns
        -------
        None
        """

        if input_only:
            dh.plot_input_only(self.buses_df, self.branches_df)
            return

        # check if model is available
        if self.results is None:
            raise ValueError('Model not solved yet! Please solve the model before plotting.')

        lego_gpt_path = os.path.abspath(str(self.lego_gpt_path))
        lego_gpt_env_path = str(self.lego_gpt_env_path)
        plot_config_path = os.path.abspath("plot_config.yml")
        missing = [f"{name} ({path})" for name, path in (('LEGO_GPT_path', lego_gpt_path),
                                                          ('LEGO_GPT_env_path', lego_gpt_env_path),
                                                          ('plot_config.yml', plot_config_path))
                   if not os.path.isfile(path)]
        if missing:
            print('Skipping LEGO-GPT result plot - file(s) not found: ' + ', '.join(missing)
                  + '. Set LEGO_GPT_path and LEGO_GPT_env_path in config.yml (see README).')
            return

        # Run LEGO-GPT in its own environment and folder, so its Dash assets are found
        print('Starting LEGO-GPT - stop the server with Ctrl+C to finish.')
        completed = subprocess.run([lego_gpt_env_path, lego_gpt_path, plot_config_path],
                                   cwd=os.path.dirname(lego_gpt_path))
        if completed.returncode not in (0, 1, 3221225786):  # 1 / 0xC000013A: stopped with Ctrl+C
            print(f'LEGO-GPT exited with code {completed.returncode}. Check that LEGO_GPT_env_path '
                  f"points to the environment created from LEGO-GPT's environment.yml.")


if __name__ == '__main__':

    # Check if there exists a config file in the current directory, if not copy it from the default config file
    if not os.path.exists("config.yml"):
        h.copy_default_config(config_path="config.yml",
                              default_config_path="config_template.yml")  # copy the default config file if it does not exist
    if not os.path.exists("plot_config.yml"):
        h.copy_default_config(config_path="plot_config.yml", default_config_path="plot_config_template.yml")
    else:
        v2g = V2G()  # Initialize the V2G class
        v2g.get_data()  # Load data from the database
        v2g.extract_generator_params()  # Extract generator data and generate parameters
        v2g.preprocess_data()  # Preprocess the data
        v2g.plot_network(input_only=True)  # Plot the grid topology of the input data
        v2g.create_model()  # Create the Pyomo model
        v2g.solve_model()  # Solve the Pyomo model
        if v2g.solution_loaded:
            v2g.export_results()  # Export the results to a SQLite database
            v2g.plot_network()  # Open the results in LEGO-GPT (see README)