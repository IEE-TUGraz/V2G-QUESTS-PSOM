"""
This module provides utility functions for operations on Pyomo models and
data structures used in power systems and electric vehicle simulations.

Key Features
------------
- Generate availability dictionaries for electric vehicles across scenarios, times, and buses.
- Identify upstream bus connections in network branch data.
- Safely copy and load YAML configuration files for flexible simulation setup.

Usage
-----
Import this module as a utility toolkit in the V2G-QUESTS project.
"""

import os

import yaml

def delta(i: str, model) -> list:
    """
    Identify and return the upstream buses connected to a given bus.

    Parameters
    ----------
    i : str
        Identifier of the "to bus" (tbus) to search for.
    model : object
        An object containing a `Branches` attribute, which is an iterable of
        tuples representing branch connections (fbus, tbus).

    Returns
    -------
    list
        List of "from buses" (fbus) connected to the specified "to bus" (tbus).
    """

    tbuses = []

    # iterate through all branches
    for fbus, tbus in model.Branches:
        if tbus == i:
            tbuses.append(fbus)
    return tbuses


def copy_default_config(config_path="config.yml", default_config_path="config_template.yml"):
    """
    Copy a default configuration file to a specified path if it does not exist.

    Parameters
    ----------
    config_path : str, optional
        Target path for the configuration file. Defaults to "config.yml".
    default_config_path : str, optional
        Path to the default configuration template. Defaults to "config_template.yml".

    Behavior
    --------
    - If `config_path` does not exist, reads `default_config_path` and writes it to `config_path`.
    - If `config_path` exists, no action is taken.

    Prints
    ------
    - A message indicating that the default configuration has been copied.
    - Or a message indicating that the configuration file already exists.
    """

    if not os.path.exists(config_path):
        with open(default_config_path, 'r') as f:
            config = f.read()
        with open(config_path, 'w') as f:
            f.write(config)
        print(f"Copied default configuration to {config_path}. Enter path settings in the file, then rerun.")

    else:
        print(f"Configuration file already exists at {config_path}.")


def load_config(config_path="config.yml") -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. Defaults to "config.yml".

    Returns
    -------
    dict
        Contents of the YAML configuration file.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
