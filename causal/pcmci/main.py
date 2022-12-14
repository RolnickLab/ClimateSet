import argparse
import warnings
import os
import json
import torch
import numpy as np
from data_loader import DataLoader
from pcmci import varimax_pcmci


class Bunch:
    """
    A class that has one variable for each entry of a dictionnary.
    """

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def to_dict(self):
        return self.__dict__

    def fancy_print(self, prefix=''):
        str_list = []
        for key, val in self.__dict__.items():
            str_list.append(prefix + f"{key} = {val}")
        return '\n'.join(str_list)


def main(hp):
    """
    :param hp: object containing hyperparameter values
    """
    # Control as much randomness as possible
    torch.manual_seed(hp.random_seed)
    np.random.seed(hp.random_seed)

    # Create folder
    args.exp_path = os.path.join(args.exp_path, f"exp{args.exp_id}")
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    # generate data and split train/test
    data_loader = DataLoader(ratio_train=hp.ratio_train,
                             ratio_valid=hp.ratio_valid,
                             data_path=hp.data_path,
                             data_format=hp.data_format,
                             latent=hp.latent,
                             no_gt=hp.no_gt,
                             debug_gt_w=False,
                             instantaneous=hp.instantaneous,
                             tau=hp.tau)
    data = data_loader.x
    data = data.squeeze(0)
    data = data.reshape(data.shape[0], -1)

    learned_dag, learned_W, metrics = varimax_pcmci(data, data_loader.idx_train,
                                                    data_loader.idx_valid, hp,
                                                    data_loader.z,
                                                    data_loader.gt_w,
                                                    data_loader.gt_graph)

    # create path to exp and save hyperparameters
    save_path = os.path.join(hp.exp_path, "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(hp.exp_path, "params.json"), "w") as file:
        json.dump(vars(hp), file, indent=4)

    # save final results if have GT (shd, f1 score, etc)
    with open(os.path.join(hp.exp_path, "results.json"), "w") as file:
        json.dump(metrics, file, indent=4)


def assert_args(args):
    """
    Raise errors or warnings if some args should not take some combination of
    values.
    """
    # raise errors if some args should not take some combination of values
    # if args.no_gt and (args.debug_gt_graph or args.debug_gt_z or args.debug_gt_w):
    #     raise ValueError("Since no_gt==True, all other args should not use ground-truth values")

    if args.latent and (args.d_z is None or args.d_x is None or args.d_z <= 0 or args.d_x <= 0):
        raise ValueError("When using latent model, you need to define d_z and d_x with integer values greater than 0")

    if args.ratio_valid == 0:
        args.ratio_valid = 1 - args.ratio_train
    if args.ratio_train + args.ratio_valid > 1:
        raise ValueError("The sum of the ratio for training and validation set is higher than 1")

    # string input with limited possible values
    supported_dataformat = ["numpy", "hdf5"]
    if args.data_format not in supported_dataformat:
        raise ValueError(f"This file format ({args.data_format}) is not \
                         supported. Supported types are: {supported_dataformat}")

    # warnings, strange choice of args combination
    if args.latent and (args.d_z > args.d_x):
        warnings.warn("Are you sure you want to have a higher dimension for d_z than d_x")

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal models for climate data")
    # for the default values, check default_params.json

    parser.add_argument("--exp-path", type=str, default="causal_climate_exp",
                        help="Path to experiments")
    parser.add_argument("--config-path", type=str, default="default_params.json",
                        help="Path to a json file with values for all hyperparamters")
    parser.add_argument("--use-data-config", action="store_true",
                        help="If true, overwrite some parameters to fit \
                        parameters that have been used to generate data")
    parser.add_argument("--exp-id", type=int,
                        help="ID specific to the experiment")
    parser.add_argument("--random-seed", type=int,
                        help="Seed to use for random number generators")

    # For synthetic datasets, can use the ground-truth values to do ablation studies
    parser.add_argument("--debug-gt-z", action="store_true",
                        help="If true, use the ground truth value of Z (use only to debug)")
    parser.add_argument("--debug-gt-w", action="store_true",
                        help="If true, use the ground truth value of W (use only to debug)")
    parser.add_argument("--debug-gt-graph", action="store_true",
                        help="If true, use the ground truth graph (use only to debug)")

    # Dataset properties
    parser.add_argument("--data-path", type=str, help="Path to the dataset")
    parser.add_argument("--data-format", type=str, help="numpy|hdf5")
    parser.add_argument("--no-gt", action="store_true",
                        help="If True, does not use any ground-truth for plotting and metrics")

    # specific to model with latent variables
    parser.add_argument("--latent", action="store_true", help="Use the model that assumes latent variables")
    parser.add_argument("--d-z", type=int, help="if latent, d_z is the number of cluster z")
    parser.add_argument("--d-x", type=int, help="if latent, d_x is the number of gridcells")
    parser.add_argument("--instantaneous", action="store_true", help="Use instantaneous connections")
    parser.add_argument("--tau-min", type=int, help="Number of past timesteps to consider")
    parser.add_argument("--tau", type=int, help="Number of past timesteps to consider")
    parser.add_argument("--ratio-train", type=float,
                        help="Proportion of the data used for the training set")
    parser.add_argument("--ratio-valid", type=float,
                        help="Proportion of the data used for the validation set")

    # Specific to PCMCI
    parser.add_argument("--pc-alpha", type=float,
                        help="Significance level (alpha) used by PCMCI")
    parser.add_argument("--ci-test", type=str,
                        help="Type of conditional independence test used by PCMCI")
    parser.add_argument("--fct-type", type=str,
                        help="Function used when ftiting the data [linear, gaussian_process]")

    args = parser.parse_args()

    # if a json file with params is given,
    # update params accordingly
    if args.config_path != "":
        default_params = vars(args)
        with open(args.config_path, 'r') as f:
            params = json.load(f)

        for key, val in params.items():
            if default_params[key] is None or not default_params[key]:
                default_params[key] = val
        args = Bunch(**default_params)

    # use some parameters from the data generating process
    if args.use_data_config != "":
        with open(os.path.join(args.data_path, "data_params.json"), 'r') as f:
            params = json.load(f)
        args.d_x = params['d_x']
        if 'latent' in params:
            args.latent = params['latent']
            if args.latent:
                args.d_z = params['d_z']
        if 'tau' in params:
            args.tau = params['tau']
        if 'neighborhood' in params:
            args.tau_neigh = params['neighborhood']
        if 'num_features' in params:
            args.d = params['num_features']


    args = assert_args(args)

    main(args)
