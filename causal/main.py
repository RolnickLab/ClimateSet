import argparse
import os
import json
import torch
import numpy as np
from model import CausalModel
from data_loader import DataLoader
from train import Training


def main(hp):
    """
    :param hp: object containing hyperparameter values
    """
    # Control as much randomness as possible
    torch.manual_seed(hp.random_seed)
    np.random.seed(hp.random_seed)

    # Use GPU
    if hp.gpu:
        if hp.float:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
    else:
        if hp.float:
            torch.set_default_tensor_type("torch.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.DoubleTensor")

    if hp.latent is None:
        hp.latent = False

    # generate data and split train/test
    # TODO: add args
    data_loader = DataLoader(hp.ratio_train,
                             hp.ratio_valid,
                             hp.data_path,
                             hp.latent,
                             hp.tau)

    # initialize model
    d = data_loader.x.shape[1]

    # TODO: change args
    model = CausalModel("fixed",
                        hp.num_layers,
                        hp.num_hidden,
                        d,
                        2,
                        hp.tau,
                        hp.tau_neigh
                        )

    # create path to exp and save hyperparameters
    save_path = os.path.join(hp.exp_path, "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(hp.exp_path, "params.json"), "w") as file:
        json.dump(vars(hp), file, indent=4)

    # train
    trainer = Training(model, data_loader, hp)
    trainer.train_with_QPM()

    # save final results?


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal models for climate data")

    parser.add_argument("--exp-path", type=str, default="causal_climate_exp",
                        help="Path to experiments")
    parser.add_argument("--exp-id", type=int, default=0,
                        help="ID specific to the experiment")
    parser.add_argument("--data-path", type=str, default="dataset/data0",
                        help="Path to the dataset")

    # Dataset properties
    # parser.add_argument("--n", type=int, default=1000,
    #                    help="Number of samples")
    # parser.add_argument("--d", type=int, default=20,
    #                    help="Number of features")
    # parser.add_argument("--prob-connection", type=float, default=0.2,
    #                    help="Probability of connection in the DAG")
    parser.add_argument("--ratio-train", type=int, default=0.8,
                        help="Proportion of the data used for the training set")
    parser.add_argument("--ratio-valid", type=int, default=0.2,
                        help="Proportion of the data used for the validation set")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of samples per minibatch")

    parser.add_argument("--latent", action="store_true",
                        help="Use the model that assumes latent variables")
    parser.add_argument("--tau", type=int, default=2,
                        help="Number of past timesteps to consider")
    parser.add_argument("--tau-neigh", type=int, default=1,
                        help="Radius of neighbor cells to consider")

    # Model hyperparameters: architecture
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="Number of hidden units")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-output", type=int, default=2,
                        help="number of output units")

    # Model hyperparameters: optimization
    parser.add_argument("--optimizer", type=str, default="rmsprop",
                        help="sgd|rmsprop")
    parser.add_argument("--reg-coeff", type=float, default=0,
                        help="Coefficient for the regularisation term")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for optim")
    parser.add_argument("--random-seed", type=int, default=1,
                        help="Random seed for torch and numpy")

    # QPM options
    parser.add_argument("--omega-gamma", type=float, default=1e-4,
                        help="Precision to declare convergence of subproblems")
    parser.add_argument("--omega-mu", type=float, default=0.9,
                        help="After subproblem solved, h should have reduced by this ratio")
    parser.add_argument("--mu-init", type=float, default=1e-1,
                        help="initial value of mu")
    parser.add_argument("--mu-mult-factor", type=float, default=1.5,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument("--h-threshold", type=float, default=1e-8,
                        help="Can stop if h smaller than h-threshold")
    parser.add_argument("--min-iter-convergence", type=int, default=1000,
                        help="Minimal number of iteration before checking if has converged")
    parser.add_argument("--max-iteration", type=int, default=100000,
                        help="Maximal number of iteration before stopping")
    parser.add_argument("--patience", type=int, default=100,
                        help="Patience used after the acyclicity constraint is respected")
    parser.add_argument("--patience-post-thresh", type=int, default=100,
                        help="Patience used after the thresholding of the adjacency matrix")

    # logging
    parser.add_argument("--plot-freq", type=int, default=2000,
                        help="Plotting frequency")
    parser.add_argument("--valid-freq", type=int, default=100,
                        help="Plotting frequency")
    parser.add_argument("--print-freq", type=int, default=100,
                        help="Printing frequency")

    # device and numerical precision
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--float", action="store_true", help="Use Float precision")

    args = parser.parse_args()

    # Create folder
    args.exp_path = os.path.join(args.exp_path, f"exp{args.exp_id}")
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    main(args)
