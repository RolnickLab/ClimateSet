import os
import subprocess
import numpy as np
import xarray as xr

from pathlib import Path


def read_gridfile(grid_file) -> dict:
    """ Read out a gridfile.txt from cdo or manually created one.
    Params:
        grid_file (Path): Path to grid file
    Returns:
        dict: Python dictionary containing the relevant grid information
    """
    grid_attrs = {}
    with open(grid_file, 'r') as f:
        lines = f.read().splitlines()
        lines = [line.replace(' ', '') for line in lines]
        for line in lines:
            dict_pair = line.split('=')
            grid_attrs[dict_pair[0]] = dict_pair[1]
    return grid_attrs

def get_single_example(dir):
    """ Gets an example file of a directory
    Parameters:
        dir (Path): Path to the larger directory that contains the right shapes
    Return:
        Path: Path to the nc file that can be used as example
    """
    for path, subdirs, files in os.walk(dir):
        if len(files) > 0:
            first_file = Path(path, files[0])
            return first_file

def get_keys_from_value(d, val):
    keys = [k for k, v in d.items() if val in v]
    if keys:
        return keys[0]
    print(f"WARNING: source not found vor var {val}")
    return None


def runcmd(cmd, verbose=False, *args, **kwargs):
    """
    Run a bash command.
    """

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_MIP(experiment: str):
    """
    Return name of MIP group given the specific experiment name.
    """
    if experiment == "ssp245-covid":
        return "DAMIP"
    elif experiment == "ssp370-lowNTCF":
        return "AerChemMIP"
    elif experiment.startswith("ssp"):
        return "ScenarioMIP"
    elif experiment.startswith("hist-"):
        return "DAMIP"
    else:
        return "CMIP"


def get_lowest_entry(availabe: [str], hierachy: [str]):
    """
    From a given list of available items return the lowest item according to a predefined hierachy.
    Used for e.g. finding the lowest available temporal resolution.
    """
    index_list = np.where([i in availabe for i in hierachy])[0]
    if len(index_list) == 0:
        return None
    else:
        return hierachy[index_list[0]]
