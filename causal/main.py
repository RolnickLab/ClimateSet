import argparse
import os
import json
import torch
import numpy as np
import metrics
from model import TSDCD, LatentTSDCD
from data_loader import DataLoader
from train import Training
from train_latent import TrainingLatent


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

    # Create folder
    args.exp_path = os.path.join(args.exp_path, f"exp{args.exp_id}")
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    # generate data and split train/test
    data_loader = DataLoader(ratio_train=hp.ratio_train,
                             ratio_valid=hp.ratio_valid,
                             data_path=hp.data_path,
                             latent=hp.latent,
                             debug_gt_w=hp.debug_gt_w,
                             instantaneous=hp.instantaneous,
                             tau=hp.tau)

    # initialize model
    d = data_loader.x.shape[2]

    if hp.instantaneous:
        num_input = d * (hp.tau + 1) * (hp.tau_neigh * 2 + 1)
    else:
        num_input = d * hp.tau * (hp.tau_neigh * 2 + 1)

    if not hp.latent:
        model = TSDCD(model_type="fixed",
                      num_layers=hp.num_layers,
                      num_hidden=hp.num_hidden,
                      num_input=num_input,
                      num_output=2,
                      d=d,
                      tau=hp.tau,
                      tau_neigh=hp.tau_neigh,
                      instantaneous=hp.instantaneous,
                      hard_gumbel=hp.hard_gumbel)
    else:
        model = LatentTSDCD(num_layers=hp.num_layers,
                            num_hidden=hp.num_hidden,
                            num_input=num_input,
                            num_output=2,
                            d=d,
                            distr_z0="gaussian",
                            distr_encoder="gaussian",
                            distr_transition="gaussian",
                            distr_decoder="gaussian",
                            d_x=hp.d_x,
                            k=hp.k,
                            tau=hp.tau,
                            instantaneous=hp.instantaneous,
                            hard_gumbel=hp.hard_gumbel,
                            debug_gt_graph=hp.debug_gt_graph,
                            debug_gt_z=hp.debug_gt_z,
                            debug_gt_w=hp.debug_gt_w,
                            gt_w=data_loader.gt_w,
                            gt_graph=data_loader.gt_graph)

    # create path to exp and save hyperparameters
    save_path = os.path.join(hp.exp_path, "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(hp.exp_path, "params.json"), "w") as file:
        json.dump(vars(hp), file, indent=4)

    # train
    if not hp.latent:
        trainer = Training(model, data_loader, hp)
    else:
        trainer = TrainingLatent(model, data_loader, hp)
    trainer.train_with_QPM()

    # save final results (shd, f1 score, etc)
    gt_dag = trainer.gt_dag
    learned_dag = trainer.model.get_adj().detach().numpy().reshape(gt_dag.shape[0], gt_dag.shape[1], -1)
    errors = metrics.edge_errors(learned_dag, gt_dag)
    shd = metrics.shd(learned_dag, gt_dag)
    __import__('ipdb').set_trace()
    f1 = metrics.f1_score(learned_dag, gt_dag)
    errors["shd"] = shd
    errors["f1"] = f1
    print(errors)
    with open(os.path.join(hp.exp_path, "results.json"), "w") as file:
        json.dump(errors, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal models for climate data")

    parser.add_argument("--exp-path", type=str, default="causal_climate_exp",
                        help="Path to experiments")
    parser.add_argument("--config-path", type=str, default="default_params.json",
                        help="Path to a json file with values for all hyperparamters")
    parser.add_argument("--use-data-config", action="store_true",
                        help="If true, overwrite some parameters to fit \
                        parameters that have been used to generate data")
    parser.add_argument("--exp-id", type=int, default=0,
                        help="ID specific to the experiment")
    parser.add_argument("--data-path", type=str, default="dataset/data0",
                        help="Path to the dataset")

    parser.add_argument("--debug-gt-z", action="store_true",
                        help="If true, use the ground truth value of Z (use only to debug)")
    parser.add_argument("--debug-gt-w", action="store_true",
                        help="If true, use the ground truth value of W (use only to debug)")
    parser.add_argument("--debug-gt-graph", action="store_true",
                        help="If true, use the ground truth graph (use only to debug)")

    # Dataset properties

    # specific to model with latent variables
    parser.add_argument("--latent", action="store_true",
                        help="Use the model that assumes latent variables")
    parser.add_argument("--k", type=int,
                        help="if latent, k is the number of cluster z")
    parser.add_argument("--d-x", type=int,
                        help="if latent, d_x is the number of gridcells")

    parser.add_argument("--instantaneous", action="store_true",
                        help="Use instantaneous connections")
    parser.add_argument("--tau", type=int, default=3,
                        help="Number of past timesteps to consider")
    parser.add_argument("--tau-neigh", type=int, default=0,
                        help="Radius of neighbor cells to consider")
    parser.add_argument("--ratio-train", type=int, default=0.8,
                        help="Proportion of the data used for the training set")
    parser.add_argument("--ratio-valid", type=int, default=0.2,
                        help="Proportion of the data used for the validation set")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of samples per minibatch")

    # Model hyperparameters: architecture
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="Number of hidden units")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-output", type=int, default=2,
                        help="number of output units")

    # Model hyperparameters: optimization
    parser.add_argument("--optimizer", type=str, default="rmsprop",
                        help="sgd|rmsprop")
    parser.add_argument("--reg-coeff", type=float, default=1e-2,
                        help="Coefficient for the regularisation term")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for optim")
    parser.add_argument("--random-seed", type=int, default=2,
                        help="Random seed for torch and numpy")
    parser.add_argument("--hard-gumbel", action="store_true",
                        help="If true, use the hard version when sampling the masks")

    # QPM options
    parser.add_argument("--omega-gamma", type=float, default=1e-4,
                        help="Precision to declare convergence of subproblems")
    parser.add_argument("--omega-mu", type=float, default=0.9,
                        help="After subproblem solved, h should have reduced by this ratio")
    parser.add_argument("--mu-init", type=float, default=1e-1,
                        help="initial value of mu")
    parser.add_argument("--mu-mult-factor", type=float, default=2,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument("--h-threshold", type=float, default=1e-8,
                        help="Can stop if h smaller than h-threshold")
    parser.add_argument("--min-iter-convergence", type=int, default=1000,
                        help="Minimal number of iteration before checking if has converged")
    parser.add_argument("--max-iteration", type=int, default=100000,
                        help="Maximal number of iteration before stopping")
    parser.add_argument("--patience", type=int, default=1000,
                        help="Patience used after the acyclicity constraint is respected")
    parser.add_argument("--patience-post-thresh", type=int, default=100,
                        help="Patience used after the thresholding of the adjacency matrix")

    # logging
    parser.add_argument("--plot-freq", type=int, default=1000,
                        help="Plotting frequency")
    parser.add_argument("--valid-freq", type=int, default=100,
                        help="Plotting frequency")
    parser.add_argument("--print-freq", type=int, default=100,
                        help="Printing frequency")

    # device and numerical precision
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--float", action="store_true", help="Use Float precision")

    args = parser.parse_args()

    # if a json file with params is given,
    # update params accordingly
    if args.config_path != "":
        default_params = vars(args)
        with open(args.config_path, 'r') as f:
            params = json.load(f)
        default_params.update(params)
        args = Bunch(**default_params)

    # use some parameters from the data generating process
    if args.use_data_config != "":
        with open(os.path.join(args.data_path, "data_params.json"), 'r') as f:
            params = json.load(f)
        args.tau = params['tau']
        args.tau_neigh = params['neighborhood']
        args.latent = params['latent']
        if args.latent:
            args.k = params['k']
            args.d_x = params['d_x']

    main(args)
