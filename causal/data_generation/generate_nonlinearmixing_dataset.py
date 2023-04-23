import os
import json
import copy
import numpy as np
from pathlib import Path


def generate_all_dataset(root_path: str, varying_params: dict, default_params: dict, n_dataset: int = 10):
    """
    This code generate all the datasets used in the main text.
    Args:
        root_path: path where all the datasets will be saved
        varying_params
        default_params
        n_dataset: number of datasets with the same parameters
    """
    i = 0
    func_type_list = ["linear"]
    fixed_diagonal_list = [True]

    for key_param, val_param in varying_params.items():
        for p in val_param:
            for func_type in func_type_list:
                for fixed_diag in fixed_diagonal_list:
                    seed = np.random.randint(1, 1000)
                    parent_directory = f"exp_nonlinearmixing"

                    for i_exp in range(n_dataset):
                        params = copy.deepcopy(default_params)
                        params[key_param] = p

                        params["func_type"] = func_type
                        if func_type == "linear":
                            params["radius_correct"] = 1e-2
                        elif func_type == "add_nonlinear":
                            params["radius_correct"] = 1e0
                        params["fixed_diagonal"] = fixed_diag
                        params["nonlinear_mixing"] = True
                        params["exp_id"] = i_exp
                        params["random_seed"] = seed + i_exp
                        if fixed_diag:
                            diag = 1
                        else:
                            diag = 0
                        directory = f"data_tau{params['tau']}_density{params['prob']}_dz{params['d_z']}"
                        directory += f"_dx{params['d_x']}_d{params['d']}_{func_type}_diagonal{diag}"

                        params["exp_path"] = os.path.join(root_path, parent_directory, directory, f"data_{i_exp}")
                        print(i)
                        print(params["exp_path"])
                        i += 1

                        # create directory structure
                        Path(params["exp_path"]).mkdir(parents=True, exist_ok=True)

                        # save the parameters in a json file
                        config_path = os.path.join(params["exp_path"], "params.json")
                        with open(config_path, "w") as file:
                            json.dump(params, file, indent=4)

                        # launch in shell
                        os.system(f"python main.py --use-config --config-path {config_path}")


if __name__ == "__main__":
    varying_params = {
        "d_z": [10]
    }

    default_params = {
        "latent": True,
        "tau": 1,
        "prob": 0.15,
        "d_z": 10,
        "d_x": 100,
        "d": 1,
        "t": 5000,
        "n": 1,
        "noise_type": "gaussian",
        "instantaneous": False,
        "noise_x_std": 0.1,
        "noise_z_std": 1,
        "radius_correct": 1e-1
    }

    generate_all_dataset("data", varying_params, default_params)
