**V2G-QUESTS-PSOM**

V2G-QUESTS-PSOM (Vehicle to Grid for Equitable Zero-Emission Transitions in Positive Energy Districts – Power System Optimization Model) is an optimization tool for medium-voltage grid systems. It models three test regions: Aradas (Portugal), Annelinn (Estonia), and Kanaleneiland (Netherlands).

The model optimizes a stylized grid, including renewable energy generation, decentralized storage, and the management of electric vehicle charging and discharging with the overall goal of minimizing costs for imported energy vehicle charging and unfulfilled demand.<br/>
It offers DC-OPF and Second-Order-Cone-Programming (SOCP) variants, as well as the option to solve the model as a relaxed Mixed-Integer Program (rMIP).

The public version includes a test case for Kanaleneiland, featuring a simplified grid topology and a reduced representation of electric vehicle usage.

**Getting Started**

- Install the V2G-QUEST_PSOM environment using the activate_environment_windows.bat script. This will automatically install all required dependencies and packages from the environment.yml file.

- On the first run, the program will copy config_template.yml to create config.yml. This file allows users to modify overall model parameters.

- Additionally, standard_parameters.yml provides default values for modeling parameters, which can be adjusted as required.

**Inputs**

The model inputs are provided either in Excel sheets or in the standard-parameters.yml file for easy accessibility and adaptability.
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

After the model is solved, all model components, including sets, parameters, and variable values are exported into an SQLite file. This file can also be visualized using the LEGO-GPT tool, which is currently not included in the public version.
