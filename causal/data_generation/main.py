import argparse
import os
import torch
import json

import numpy as np
from pathlib import Path
from data_generation import DataGeneratorWithLatent, DataGeneratorWithoutLatent
from plot import plot_adjacency_graphs, plot_adjacency_w, plot_x, plot_z


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


def main(hp, func_types: list, noise_types: list):
    """
    Args:
        hp: object containing hyperparameter values
        func_types: type of functions that are possible
        noise_types: type of noise that are possible

    Returns:
        The observable data that has been generated
    """
    # Control as much randomness as possible
    if hp.random_seed != 999:
        torch.manual_seed(hp.random_seed)
        np.random.seed(hp.random_seed)

    if hp.latent:
        generator = DataGeneratorWithLatent(args)
    else:
        generator = DataGeneratorWithoutLatent(args)

    if hp.func_type not in func_types:
        raise NotImplementedError(f"Types of function that are presently implemented: {func_types}")

    if hp.noise_type not in noise_types:
        raise NotImplementedError(f"Types of noise that are presently implemented: {noise_types}")

    # Generate, save and plot data
    data = generator.generate()
    print(f"Saving data at {hp.exp_path}...")
    generator.save_data(hp.exp_path)
    print("Plotting data...")
    plot_adjacency_graphs(generator.G, hp.exp_path)

    plot_x(generator.X.detach().numpy(), hp.exp_path)
    # if hp.latent:
    #     plot_z(generator.Z.detach().numpy(), hp.exp_path)
    #     plot_adjacency_w(generator.w, hp.exp_path)

    return data


if __name__ == "__main__":
    func_types = ["linear", "mlp", "add_nonlinear"]
    noise_types = ["gaussian", "laplacian", "uniform"]

    parser = argparse.ArgumentParser(description="Code use to generate synthetic data to \
                                     test the end-to-end clustering idea.")

    parser.add_argument("--use-config", action="store_true",
                        help="If True, use a config file")
    parser.add_argument("--config-path", type=str,
                        help="Path to config file used if use-config is True")
    parser.add_argument("--exp-path", type=str,
                        help="Path to experiments")
    parser.add_argument("--exp-id", type=int, default=0,
                        help="ID unique to the dataset")
    parser.add_argument("--random-seed", type=int, default=999,
                        help="Random seed used for torch and numpy")

    # Dataset properties
    parser.add_argument("--latent", action="store_true",
                        help="Use generative model with latents")
    parser.add_argument("--func-type", type=str, default="linear",
                        help=f"Type of function for the generating process {func_types}")
    parser.add_argument("--noise-type", type=str, default="gaussian",
                        help=f"Type of noise for the generating process {noise_types}")
    parser.add_argument("--instantaneous", action="store_true",
                        help="Use instantaneous connection when generating data")
    parser.add_argument("-n", type=int,
                        help="Number of time series")
    parser.add_argument("-t", type=int,
                        help="Number of timesteps in total")
    parser.add_argument("-d", type=int,
                        help="Number of features")
    parser.add_argument("--d-x", type=int,
                        help="Number of gridcells d_x")
    parser.add_argument("--d-z", type=int,
                        help="Number of clusters")
    parser.add_argument("-p", "--prob", type=float,
                        help="Probability of an edge in the causal graphs")
    parser.add_argument("--tau", type=int,
                        help="Number of previous timestep that interacts with a timestep t")
    parser.add_argument("--noise-coeff", type=float,
                        help="Coefficient for the additive noise")
    parser.add_argument("--noise-x-std", type=float,
                        help="Std of the noise applied on X")
    parser.add_argument("--noise-z-std", type=float,
                        help="Std of the noise applied on Z")

    # params specific to non-latent
    parser.add_argument("--world-dim", type=float, default=1,
                        help="Number of dimension for the gridcell (1D or 2D)")  # J: Can we also do 3d?
    parser.add_argument("--circular-padding", action="store_true",
                        help="If true, pad gridcells in a circular fashion")
    parser.add_argument("--neighborhood", type=int, default=0,
                        help="'Radius' of neighboring gridcells that have an influence")
    parser.add_argument("--eta", type=int, default=1.0,
                        help="Weight decay applied to linear coefficients")

    # Neural network (NN) architecture
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of layers in NN")
    parser.add_argument("--num-hidden", type=int, default=6,
                        help="Number of hidden units in NN")
    parser.add_argument("--non-linearity", type=str, default="leaky_relu",
                        help="Type of non-linearity used in the NN")

    args = parser.parse_args()

    # if a config file is given, use it to update the values of arg
    if args.use_config:
        default_params = vars(args)
        with open(args.config_path, 'r') as file:
            params = json.load(file)

        for key, val in params.items():
            if default_params[key] is None or not default_params[key]:
                default_params[key] = val
        args = Bunch(**default_params)

    # Create folder
    if not args.use_config:
        args.exp_path = os.path.join(args.exp_path, f"data_{args.exp_id}")
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)

    main(args, func_types, noise_types)
