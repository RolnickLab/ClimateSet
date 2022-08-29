import subprocess
import numpy as np


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
