**V2G-QUESTS-PSOM**

V2G-QUESTS-PSOM (Vehicle to Grid for Equitable Zero-Emission Transitions in Positive Energy Districts – Power System Optimization Model) is an optimization tool for medium-voltage grid systems. It models three test regions: Aradas (Portugal), Annelinn (Estonia), and Kanaleneiland (Netherlands).

The model optimizes a stylized grid, including renewable energy generation, decentralized storage, and the management of electric vehicle charging and discharging with the overall goal of minimizing costs for imported energy, vehicle charging and unfulfilled demand.<br/>
It offers DC-OPF and Second-Order-Cone-Programming (SOCP) variants, as well as the option to solve the model as a relaxed Mixed-Integer Program (rMIP). For the SOCP variant, the exactness of the relaxation can be enforced with an iterative penalty convex-concave procedure.

The public version includes the input data of all three test regions. Vehicles can be modelled as smart-charging EVs, bidirectional V2G vehicles, or with uncontrolled charging (UC).

**Getting Started**

- Install the V2G-QUESTS_PSOM environment using the activate_environment_windows.bat script. This will automatically install all required dependencies and packages from the environment.yml file.

- The model is solved with Gurobi, which requires a valid Gurobi license (free academic licenses are available from Gurobi).

- On the first run, the program will copy config_template.yml to create config.yml and plot_config_template.yml to create plot_config.yml, and then stop. Adjust both files and run the program again. config.yml contains the overall model, solver and scenario parameters.

- Additionally, case_studies/standard_parameter.yml provides default values for modeling parameters, such as voltage limits, which can be adjusted as required.

**Inputs**

The model inputs are provided either in Excel sheets or in the standard_parameter.yml file for easy accessibility and adaptability.
These inputs include:

-Energy prices

-Demand profiles, including standard profiles

-Standard parameters for lines, EVs, and battery systems

-Renewable capacity factors

-Decentralized renewable generation

-Decentralized storage systems

-EV/V2G movement and energy demand

**All data is expressed in MW or MWh.**



**Outputs**

After the model is solved, all model components, including sets, parameters, and variable values are exported into the SQLite file output/result.sqlite.

**Visualization with LEGO-GPT**

The results can be explored interactively with LEGO-GPT, a Dash-based visualization tool for network maps, power flows, dispatch, demand and EV charging. The V2G-QUESTS-PSOM version of LEGO-GPT is available on the V2G-PSOM branch of https://github.com/IEE-TUGraz/LEGO-GPT.

1. Clone LEGO-GPT, check out the V2G-PSOM branch, and create its conda environment from LEGO-GPT's environment.yml (environment name LEGO_GPT_env).

2. In config.yml, set LEGO_GPT_path to the LEGO_GPT.py file and LEGO_GPT_env_path to the python.exe of the LEGO_GPT_env environment.

3. plot_config.yml configures the visualization. Its default input_dir: output points to the result file of the last model run.

After a successful run, V2G.py opens the results in LEGO-GPT automatically. The address of the web interface is printed in the console; stop the server with Ctrl+C. If the LEGO-GPT paths are not set, this step is skipped with a message.
