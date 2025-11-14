import math
import os
import sqlite3
import subprocess
import time

import pandas as pd
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, minimize, Constraint, Param, value, \
    SolverFactory, TransformationFactory, Binary, Reals, Set, Any, Expression
from pyomo.opt import SolverStatus, TerminationCondition

from submodules import SystemData as sd
from submodules  import data_helper as dh
from submodules  import helper_functions as h


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
        self.pQfactor = config['pQfactor']
        self.pV_slack = config['pV_slack']
        self.pCh_Inf_Cost = config['pCh_Inf_Cost']
        self.pPenality = config['pPenality']

        # Initialize placeholders for the Pyomo model and solver results
        self.model = None  # Pyomo model object
        self.results = None  # Pyomo results object

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
            - ``num_t``, ``start_t``, ``case_study``, ``enable_pv``, ``enable_wind``,
              ``enable_hydro``, ``year``, ``pBatt_cap``, ``pCharge_Invest``

        Returns
        -------
        None
        """

        # measure start time for loading
        start_time = time.time()
        print('Loading system data')

        # instantiate SystemData with configuration values and load the data
        case_study = sd.SystemData(self.num_t, self.start_t, self.case_study, self.enable_pv, self.enable_wind,
                                   self.enable_hydro, self.year, self.pBatt_cap, self.pCharge_Invest, self.slack_bus)

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

        # Define grid sets
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

        print('Sets sucessfully defined')
        # ============================================
        # Define Parameters
        # ============================================

        # --------------------------------------------
        # General Parameters
        # --------------------------------------------
        model.add_component('pBigM',
                            Param(initialize=1e5, domain=NonNegativeReals, doc='Big M constant for linearization'))

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

        # AC-OPF specific parameters
        if not self.dc_opf:
            model.add_component('pQfactor', Param(initialize=self.pQfactor, domain=NonNegativeReals,
                                      doc='Reactive power cost factor for SOCP formulation'))

            model.add_component('pV_slack', Param(initialize=self.pV_slack, domain=NonNegativeReals,
                                      doc='Slack bus voltage magnitude for SOCP formulation'))
        # --------------------------------------------
        # Generator Parameters
        # --------------------------------------------
        # Define the types of generators and their parameters
        generator_types = ['PV', 'Wind', 'Hydro']  # List of generator types (Photovoltaic, Wind, Hydro)
        params = ['Pmax', 'Pmin', 'Qmax', 'Qmin']  # List of generator parameters (maximum/minimum active and reactive power)
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
        model.add_component('pEV_SOC_max', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'EV_SOC_max [MWh]'].astype(float).to_dict(), domain=NonNegativeReals,
                                                doc='Maximum SoC of EVs'))

        model.add_component('pEta_ch', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'eta_ch_EV'].astype(float).to_dict(), domain=NonNegativeReals,
                                                doc='Charging efficiency of EVs'))

        model.add_component('pEta_dch', Param(model.EVs,initialize=self.evs_df.set_index('vehicle_name')[
            'eta_dch_EV'].astype(float).to_dict(), domain=NonNegativeReals,
                                                doc='Discharge efficiency of EVs'))

        model.add_component('pEVCh_max', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'Ev_ch_max [MW]'].astype(float).to_dict(), domain=NonNegativeReals,
                                                doc='Maximum charging power of EVs'))

        model.add_component('pEVDch_max', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'Ev_dch_max [MW]'].astype(float).to_dict(), domain=NonNegativeReals,
                                                doc='Maximum discharging power of EVs'))

        model.add_component('pEV_SOC_init', Param(model.EVs, initialize=self.evs_df.set_index('vehicle_name')[
            'SoC_init_EV'].astype(float).to_dict(),
                                                doc='Initial state of charge of EVs'))

        model.add_component('pCh_Inf_Cost', Param(initialize=self.pCh_Inf_Cost, domain=Reals,
                                                doc = 'Charging infrastructure cost per MW'))

        # Create a dictionary mapping each EV's name and departure time to its energy demand.
        # The `evs_df` DataFrame is indexed by 'vehicle_name' and 'departure_time [h]',
        # and the 'EV_demand [MWh]' column is converted to float and then to a dictionary.
        soc_dict = self.evs_df.set_index(['vehicle_name', 'departure_time [h]'])['EV_demand [MWh]'].astype(float).to_dict()

        model.add_component('pSOC_final', Param(model.EVs, model.Time, initialize=lambda m, ev, t: soc_dict.get((ev, t), 0), domain=NonNegativeReals,
                                                doc='Required SoC at departure of EVs'))

        # Add availability matrix as a parameter that indicates if a EV is available at a certain node at a certain time

        model.pAvailable = Param(model.EVs, model.Buses, model.Time, initialize=lambda m, ev, bus, t:
        self.availability_matrix.set_index(['vehicle', 'node', 'time'])['available'].astype(int).to_dict().get((ev, bus, t), 0), domain=Binary,
                                                doc="Binary availability of EVs at nodes over time")

        model.add_component('pCharge_Invest', Param(initialize=float(self.pCharge_Invest), domain=Reals,
                                                doc='Charging infrastructure investment cost per MW per EV'))
        # --------------------------------------------
        # Decentralized Battery Parameters
        # --------------------------------------------
        model.add_component('pEta_ch_batt', Param(initialize=float(self.distributed_batt_data_df['eta_ch_batt'].iat[0]),
                                                doc='Charging efficiency of decentralized batteries'))

        model.add_component('pEta_dch_batt' ,Param(initialize=float(self.distributed_batt_data_df['eta_dch_batt'].iat[0]),
                                                doc='Discharging efficiency of decentralized batteries'))

        model.add_component('pBatt_SOC_init', Param(initialize=float(self.distributed_batt_data_df['SoC_init_batt'].iat[0]),
                                                doc='Initial state of charge of decentralized batteries'))

        model.add_component('pBatt_SOC_min', Param(initialize=float(self.distributed_batt_data_df['DoD_batt'].iat[0]),
                                                doc='Depth of discharge of decentralized batteries'))

        model.add_component('pC_rate_batt', Param(initialize=float(self.distributed_batt_data_df['C_rate_batt'].iat[0]),
                                                doc='C-rate of decentralized batteries'))

        model.add_component('pBus_batt_cap', Param(model.Buses, initialize=self.bus_batt_cap_df['capacity'].astype(float).to_dict(), domain=Reals,
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
        # --------------------------------------------
        # EV Variables
        # --------------------------------------------
        model.vSoc_EV = Var(model.EVs, model.Time, domain=NonNegativeReals,
                                                doc='State of Charge of EVs')
        model.vCh_EV = Var(model.EVs, model.Time, domain=NonNegativeReals,
                                                doc='Charging Amount of EVs')
        model.vDch_EV = Var(model.EVs, model.Time, domain=NonNegativeReals,
                                                doc='Discharging Amount of EVs')
        model.vSoc_n = Var(model.EVs, model.Time, domain=NonNegativeReals,
                                                doc='Slack variable for undercharging EVs')
        model.vSoc_p = Var(model.EVs, model.Time, domain=NonNegativeReals,
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
        model.vSOC_Batt = Var(model.Buses, model.Time, domain=NonNegativeReals,
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
            model.vCii = Var(model.Buses, model.Time, domain=NonNegativeReals,
                                                doc='Squared voltage magnitude at bus'
                                                    ' :math:''`i`, i.e. :math:`v_{C,ii} = V_{i,re}^2 + V_{i,im}^2`')

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
        @model.Constraint(model.Branches, model.Time)
        def thermal_constraint(model, i, j, t):
            r"""
            Thermal limit constraint for active power flow on branch :math:`(i, j)`.

            Ensures that the active power flow does not exceed the thermal capacity of the line:

            .. math::

                vLineP_{ij,t} \leq P^{\max}_{ij}

            """
            return model.vLineP[(i, j), t] <= model.pPmax_line[i, j]

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

                SOC_{i,0} = pBus\_batt\_cap_i \cdot pBatt\_SoC\_init

                SOC_{i,t} = SOC_{i,t-1}
                           + \eta_{ch} \cdot Ch_{i,t-1}
                           - \frac{Dch_{i,t-1}}{\eta_{dch}}, \quad t > 0
            """
            if t == model.Time.first():
                return model.vSOC_Batt[i, t] == model.pBus_batt_cap[i] * model.pBatt_SOC_init
            else:
                return (
                        model.vSOC_Batt[i, t] ==
                        model.vSOC_Batt[i, t - 1]
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

                SOC_{i,t} \leq pBus\_batt\_cap_i
            """
            return model.vSOC_Batt[i, t] <= model.pBus_batt_cap[i]

        @model.Constraint(model.Buses, model.Time)
        def soc_batt_min_constraint(model, i, t):
            r"""
            Minimum SoC constraint (cannot fall below allowed depth of discharge).

            .. math::

                SOC_{i,t} \geq pBatt\_SoC\_min \cdot pBus\_batt\_cap_i
            """
            return model.vSOC_Batt[i, t] >= model.pBatt_SOC_min * model.pBus_batt_cap[i]

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
        # --------------------------------------------
        # EV equations
        # --------------------------------------------
        @model.Constraint(model.EVs, model.Time)
        def soc_departure_constraint(model, ev, t):
            r"""
            Ensure EV SoC meets required departure SoC.

            .. math::

                SOC_{ev,t} \geq SoC^{final}_{ev,t}
            """
            return model.vSoc_EV[ev, t] >= model.pSOC_final[ev, t]

        @model.Constraint(model.EVs, model.Time)
        def soc_charge_constraint(model, ev, t):
            r"""
            EV SoC evolution over time.

            - At the first time step, SoC is initialized based on initial SoC fraction:

            .. math::

                SOC_{ev,0} = pEV\_SoC\_init_{ev} \cdot pEV\_SoC\_max_{ev}

            - For subsequent time steps, SoC evolves considering charging, discharging,
              and slack variables to guarantee solvability:

            .. math::

                SOC_{ev,t} = SOC_{ev,t-1}
                            + \eta_{ch,ev} \cdot Ch_{ev,t-1}
                            - \frac{Dch_{ev,t-1}}{\eta_{dch,ev}}
                            + SoC^n_{ev,t-1} - SoC^p_{ev,t-1} - SoC^{final}_{ev,t-1}, \quad t > 0
            """
            if t == model.Time.first():
                return model.vSoc_EV[ev, t] == model.pEV_SOC_init[ev] * model.pEV_SOC_max[ev]
            else:
                return (
                        model.vSoc_EV[ev, t] ==
                        model.vSoc_EV[ev, t - 1]
                        + model.vCh_EV[ev, t - 1] * model.pEta_ch[ev]
                        - model.vDch_EV[ev, t - 1] / model.pEta_dch[ev]
                        + model.vSoc_n[ev, t - 1]
                        - model.vSoc_p[ev, t - 1]
                        - model.pSOC_final[ev, t - 1]
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
        def soc_max_constraint(model, ev, t):
            r"""
            Limit maximum SoC of EVs.

            .. math::

                SOC_{ev,t} \leq SoC^{max}_{ev}
            """
            return model.vSoc_EV[ev, t] <= model.pEV_SOC_max[ev]

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
            return sum((model.vCh_EV[ev, t] + model.vDch_EV[ev, t]) * model.pAvailable[ev, i, t] for ev in model.EVs) <= model.vCh_cap_EV[i]

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
                return (
                        - model.pDemand[t, i]
                        - sum(model.pAvailable[ev, i, t] * model.vCh_EV[ev, t] for ev in model.EVs)
                        + sum(model.pAvailable[ev, i, t] * model.vDch_EV[ev, t] for ev in model.EVs)
                        + (model.vProd_PV[i, t] if i in model.PV_Buses else 0)
                        + (model.vProd_Wind[i, t] if i in model.Wind_Buses else 0)
                        + (model.vProd_Hydro[i, t] if i in model.Hydro_Buses else 0)
                        - model.vCh_Batt[i, t]
                        + model.vDch_Batt[i, t]
                        + model.vPNS[i, t]
                        - model.vEPS[i, t]
                        == sum(model.vLineP[(i, j), t] for j in h.delta(i,model))) # vLineP is Positive for outgoing flow from i to j

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
                return (
                        - model.pDemand[t, i]
                        - sum(model.pAvailable[ev, i, t] * model.vCh_EV[ev, t] for ev in model.EVs)
                        + sum(model.pAvailable[ev, i, t] * model.vDch_EV[ev, t] for ev in model.EVs)
                        + (model.vProd_PV[i, t] if i in model.PV_Buses else 0)
                        + (model.vProd_Wind[i, t] if i in model.Wind_Buses else 0)
                        + (model.vProd_Hydro[i, t] if i in model.Hydro_Buses else 0)
                        + model.Gs[i] * model.vCii[i, t]
                        - model.vCh_Batt[i, t]
                        + model.vDch_Batt[i, t]
                        + model.vPNS[i, t]
                        - model.vEPS[i, t]
                        == sum([model.vLineP[(i, j), t]
                                for j in h.delta(i, model)]))

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

            @model.Constraint(model.Buses, model.Time)
            def voltage_limit_min(model, i, t):
                r"""
                Minimum squared voltage magnitude at a bus.

                .. math::

                    V_{min,i}^2 \leq V_{ii,t}
                """
                if i != value(model.slack_bus):
                    return (model.Vmin[i] ** 2 <= model.vCii[i, t])
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
                    return (model.vCii[i, t] <= model.Vmax[i] ** 2)  #
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
        # --------------------------------------------
        # Cost expressions and Objective-function
        # --------------------------------------------
        model.cost_function = Expression(
            expr=sum((model.vImpP[t]) * model.pElPrice[t] for t in model.Time))

        model.ev_energy_cost = Expression(
            expr=sum(model.pElPrice[t] * (model.vCh_EV[ev, t] - model.vDch_EV[ev, t])
                     for ev in model.EVs for t in model.Time))

        model.ens_cost = Expression(
            expr=(sum(model.vSoc_n[ev, t] + model.vSoc_p[ev, t] for ev in model.EVs for t in model.Time)
                + sum(model.vPNS[i, t] + model.vEPS[i, t] for i in model.Buses for t in model.Time)
                        ) * model.pPenality
        )

        model.charging_investment_cost = Expression(
            expr=sum(model.vCh_cap_EV[i] for i in model.Buses) * model.pCh_Inf_Cost)


        model.obj = Objective(
            expr=model.cost_function
                 + model.ens_cost
                 + model.charging_investment_cost
                 + model.ev_energy_cost
            , sense=minimize,
            doc=r"""
                Total system cost objective, including electricity import, EV energy costs,
                Energy Not Served (ENS), and charging infrastructure investment.

                The objective function is the sum of four components:

                1. **Electricity import cost:**

                    .. math::

                        Cost_{import} = \sum_{t \in T} P^{imp}_t \cdot Price_t

                    where:
                    - \(P^{imp}_t\) = imported electricity at time t
                    - \(Price_t\) = electricity price at time t

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
                """
        )

        self.model = model

        print('Pyomo model created in', time.time() - start_time, 'seconds')

    def solve_model(self):
        """
        Solve the Pyomo optimization model using the Gurobi solver.

        This method performs the following steps:

        1. Initializes the Gurobi solver via Pyomo.
        2. Solves the optimization model.
        3. Checks the solver's status and termination condition.
           - If the solution is optimal, prints the optimal cost.
           - Otherwise, prints solver status and termination information.
        4. Iterates through the model's variables and optionally prints their values.

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
        solver = SolverFactory('gurobi')
        # Solve the model as MIP or rMIP based on the enable_rMIP flag
        if self.enable_rMIP:
            print('Solving model as a relaxed MIP (rMIP)')
            TransformationFactory('core.relax_integer_vars').apply_to(self.model)
            self.results = solver.solve(self.model, tee=True)
        else:
            print('Solving model as a MIP')
            solver.options['mipgap'] = self.MIP_Gap  # Set optimal MIP gap (e.g.: 0.0001 = 0.10%)
            self.results = solver.solve(self.model, tee=True)

        # Check solver status
        if self.results.solver.status == SolverStatus.ok and self.results.solver.termination_condition == TerminationCondition.optimal:
            print("Solver found an optimal solution.")
            print("Optimal cost: ", value(self.model.obj))
        else:
            print("Solver did not find an optimal solution. Status:", self.results.solver.status)
            print("Solver termination condition:", self.results.solver.termination_condition)

        # Check variable values
        for var in self.model.component_objects(Var, active=False):
            for index in var:
                print(f"Variable {var.name}:")
                print(f"  {index} = {var[index].value}")
        print('Pyomo model solved in', time.time() - start_time, 'seconds')

    def export_results(self):
        """
        Export the results of the solved Pyomo model to a SQLite database.

        This method performs the following steps:

        1. Saves the model's results to a SQLite database file named ``result.sqlite``
           in the ``output`` directory.
        2. If the file already exists, it removes all existing tables before saving
           the new results.
        3. Maps model variables and solver outputs into structured tables.

        Raises
        ------
        AttributeError
            If ``results`` are not available (i.e., the model has not been solved yet).

        Returns
        -------
        None
        """

        if self.results is None:
            raise AttributeError("No results available to export.")

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

    def plot_network(self, input_only=False):

        # Note: The plot_network function uses an external tool (LEGO-GPT) to plot the network

        """
        Plot the network based on the provided data and configuration.

        This method visualizes the network topology, including buses, branches,
        and optionally the solved model results. It uses the external tool LEGO-GPT
        to generate plots in a separate environment. The LEGO-GPT toll is currently not available for the public.

        Parameters
        ----------
        input_only : bool, optional
            If True, only the input data (buses, branches, and nodes) are plotted.
            If False, the method requires the model to be solved and plots results
            as well. Defaults to False.

        Raises
        ------
        ValueError
            If ``input_only`` is False and the model has not been solved yet
            (i.e., ``self.results`` is None).

        Notes
        -----
        - When ``input_only`` is True, the method plots only the input data:
          buses, branches, and node positions.
        - When ``input_only`` is False, the method requires the solved model
          and overlays results such as voltages, flows, and EV operations.
        - Plots are generated using the external LEGO-GPT tool via a configuration
          file (``plot_config.yml``) in a separate environment.
        """

        if input_only:
            dh.plot_input_only(self.buses_df, self.branches_df)
        else:
            # check if model is available
            if self.results is None:
                raise ValueError('Model not solved yet! Please solve the model before plotting.')

            # run LEGO-GPT in seperate environment
            subprocess.run([self.lego_gpt_env_path, self.lego_gpt_path, "plot_config.yml"])


if __name__ == '__main__':

    # Check if there exists a config file in the current directory, if not copy it from the default config file
    if not os.path.exists("config.yml"):
        h.copy_default_config(config_path="config.yml",
                              default_config_path="config_template.yml")  # copy the default config file if it does not exist
    if not os.path.exists("plot_config.yml"):
        h.copy_default_config(config_path="plot_config.yml", default_config_path="plot_config_template.yml")

    else:
        v2g = V2G() # Initialize the V2G class
        v2g.get_data() # Load data from the database
        v2g.extract_generator_params() # Extract generator data and generate parameters
        v2g.preprocess_data() # Preprocess the data
        # v2g.plot_network(input_only=True) # Plot the network with the input data. Not available in the public version
        v2g.create_model() # Create the Pyomo model
        v2g.solve_model() # Solve the Pyomo model
        v2g.export_results() # Export the results to a SQLite database
        # v2g.plot_network() # Plot the network with the result data. Not available in the public version