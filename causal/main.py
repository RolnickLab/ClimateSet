import argparse
import warnings
import os
import json
import torch
import numpy as np
from metrics import mcc_latent, shd, precision_recall, edge_errors, w_mae
from model.tsdcd import TSDCD
from model.tsdcd_latent import LatentTSDCD
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
                             data_format=hp.data_format,
                             latent=hp.latent,
                             no_gt=hp.no_gt,
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
                            coeff_kl=hp.coeff_kl,
                            d=d,
                            distr_z0="gaussian",
                            distr_encoder="gaussian",
                            distr_transition="gaussian",
                            distr_decoder="gaussian",
                            d_x=hp.d_x,
                            d_z=hp.d_z,
                            tau=hp.tau,
                            instantaneous=hp.instantaneous,
                            nonlinear_mixing=hp.nonlinear_mixing,
                            hard_gumbel=hp.hard_gumbel,
                            no_gt=hp.no_gt,
                            debug_gt_graph=hp.debug_gt_graph,
                            debug_gt_z=hp.debug_gt_z,
                            debug_gt_w=hp.debug_gt_w,
                            gt_w=data_loader.gt_w,
                            gt_graph=data_loader.gt_graph,
                            tied_w=hp.tied_w)

    # create path to exp and save hyperparameters
    save_path = os.path.join(hp.exp_path, "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(hp.exp_path, "params.json"), "w") as file:
        json.dump(vars(hp), file, indent=4)

    # load the best metrics
    with open(os.path.join(hp.data_path, "best_metrics.json"), 'r') as f:
        best_metrics = json.load(f)

    # train
    if not hp.latent:
        trainer = Training(model, data_loader, hp)
    else:
        trainer = TrainingLatent(model, data_loader, hp, best_metrics)
    valid_loss = trainer.train_with_QPM()

    # save final results, (MSE)
    metrics = {"shd": 0., "precision": 0., "recall": 0., "train_mse": 0., "val_mse": 0., "mcc": 0.}
    # if we have the GT, also compute (SHD, Pr, Re, MCC)
    if not hp.no_gt:
        if hp.instantaneous:
            gt_graph = trainer.gt_dag
        else:
            gt_graph = trainer.gt_dag[:-1]  # remove the graph G_t

        # TODO: doublecheck that
        learned_graph = trainer.model.get_adj().detach().numpy().reshape(gt_graph.shape[0], gt_graph.shape[1], -1)

        score, cc_program_perm, assignments, z, z_hat = mcc_latent(trainer.model, trainer.data)
        permutation = np.zeros((gt_graph.shape[1], gt_graph.shape[1]))
        permutation[np.arange(gt_graph.shape[1]), assignments[1]] = 1
        gt_graph = permutation.T @ gt_graph @ permutation

        metrics['mcc'] = score
        metrics['w_mse'] = w_mae(trainer.model.encoder.w.detach().numpy()[:, :, assignments[1]], data_loader.gt_w)
        metrics['shd'] = shd(learned_graph, gt_graph)
        metrics['precision'], metrics['recall'] = precision_recall(learned_graph, gt_graph)
        errors = edge_errors(learned_graph, gt_graph)
        metrics['tp'] = errors['tp']
        metrics['fp'] = errors['fp']
        metrics['tn'] = errors['tn']
        metrics['fn'] = errors['fn']
        metrics['n_edge_gt_graph'] = np.sum(gt_graph)
        metrics['n_edge_learned_graph'] = np.sum(learned_graph)

        for key, val in valid_loss.items():
            metrics[key] = val

    metrics['train_mse'] = prediction(trainer, False)
    metrics['val_mse'] = prediction(trainer, True)

    with open(os.path.join(hp.exp_path, "results.json"), "w") as file:
        json.dump(metrics, file, indent=4)


def prediction(trainer, valid):
    """
    Calculate the MSE between X_{t+1} and X_hat_{t+1}
    """
    if not valid:
        x, y, y_pred = trainer.train_step()
    else:
        x, y, y_pred = trainer.valid_step()

    mse = torch.mean(0.5 * (y - y_pred) ** 2)
    return mse.item()


def assert_args(args):
    """
    Raise errors or warnings if some args should not take some combination of
    values.
    """
    # raise errors if some args should not take some combination of values
    if args.no_gt and (args.debug_gt_graph or args.debug_gt_z or args.debug_gt_w):
        raise ValueError("Since no_gt==True, all other args should not use ground-truth values")

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
    supported_optimizer = ["sgd", "rmsprop"]
    if args.optimizer not in supported_optimizer:
        raise ValueError(f"This optimizer type ({args.optimizer}) is not \
                         supported. Supported types are: {supported_optimizer}")

    # warnings, strange choice of args combination
    if not args.latent and args.debug_gt_z:
        warnings.warn("Are you sure you want to use gt_z even if you don't have latents")
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
    parser.add_argument("--tied-w", action="store_true", help="Use the same matrix W, as the decoder, for the encoder")
    parser.add_argument("--nonlinear-mixing", action="store_true", help="The encoder/decoder use NN")
    parser.add_argument("--coeff-kl", type=float, help="coefficient that is multiplied to the KL term ")
    parser.add_argument("--d-z", type=int, help="if latent, d_z is the number of cluster z")
    parser.add_argument("--d-x", type=int, help="if latent, d_x is the number of gridcells")

    parser.add_argument("--instantaneous", action="store_true", help="Use instantaneous connections")
    parser.add_argument("--tau", type=int, help="Number of past timesteps to consider")
    parser.add_argument("--tau-neigh", type=int, help="Radius of neighbor cells to consider")
    parser.add_argument("--ratio-train", type=int, help="Proportion of the data used for the training set")
    parser.add_argument("--ratio-valid", type=int, help="Proportion of the data used for the validation set")
    parser.add_argument("--batch-size", type=int, help="Number of samples per minibatch")

    # Model hyperparameters: architecture
    parser.add_argument("--num-hidden", type=int, help="Number of hidden units")
    parser.add_argument("--num-layers", type=int, help="Number of hidden layers")
    parser.add_argument("--num-output", type=int, help="Number of output units")

    # Model hyperparameters: optimization
    parser.add_argument("--optimizer", type=str, help="sgd|rmsprop")
    parser.add_argument("--reg-coeff", type=float, help="Coefficient for the sparsity regularisation term")
    parser.add_argument("--reg-coeff-connect", type=float, help="Coefficient for the connectivity regularisation term")
    parser.add_argument("--lr", type=float, help="learning rate for optim")
    parser.add_argument("--random-seed", type=int, help="Random seed for torch and numpy")
    parser.add_argument("--schedule-reg", type=int,
                        help="Start applying the sparsity regularization only after X number of steps")
    parser.add_argument("--schedule-ortho", type=int,
                        help="Start applying the orthogonality constraint only after X number of steps")
    parser.add_argument("--hard-gumbel", action="store_true",
                        help="If true, use the hard version when sampling the masks")

    # ALM/QPM options
    # orthogonality constraint
    parser.add_argument("--ortho-mu-init", type=float,
                        help="initial value of mu for the constraint")
    parser.add_argument("--ortho-mu-mult-factor", type=float,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument("--ortho-omega-gamma", type=float,
                        help="Precision to declare convergence of subproblems")
    parser.add_argument("--ortho-omega-mu", type=float,
                        help="After subproblem solved, h should have reduced by this ratio")
    parser.add_argument("--ortho-h-threshold", type=float,
                        help="Can stop if h smaller than h-threshold")
    parser.add_argument("--ortho-min-iter-convergence", type=int,
                        help="Minimal number of iteration before checking if has converged")

    # acyclicity constraint
    parser.add_argument("--acyclic-mu-init", type=float,
                        help="initial value of mu for the constraint")
    parser.add_argument("--acyclic-mu-mult-factor", type=float,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument("--acyclic-omega-gamma", type=float,
                        help="Precision to declare convergence of subproblems")
    parser.add_argument("--acyclic-omega-mu", type=float,
                        help="After subproblem solved, h should have reduced by this ratio")
    parser.add_argument("--acyclic-h-threshold", type=float,
                        help="Can stop if h smaller than h-threshold")
    parser.add_argument("--acyclic-min-iter-convergence", type=int,
                        help="Minimal number of iteration before checking if has converged")

    parser.add_argument("--mu-acyclic-init", type=float,
                        help="initial value of mu for the acyclicity constraint")
    parser.add_argument("--h-acyclic-threshold", type=float,
                        help="Can stop if h smaller than h-threshold")

    parser.add_argument("--max-iteration", type=int,
                        help="Maximal number of iteration before stopping")
    parser.add_argument("--patience", type=int,
                        help="Patience used after the acyclicity constraint is respected")
    parser.add_argument("--patience-post-thresh", type=int,
                        help="Patience used after the thresholding of the adjacency matrix")

    # logging
    parser.add_argument("--valid-freq", type=int, help="Frequency of evaluating the loss on the validation set")
    parser.add_argument("--plot-freq", type=int, help="Plotting frequency")
    parser.add_argument("--plot-through-time", action="store_true", help="If true, save each plot in a \
                        different file with a name depending on the iteration")
    parser.add_argument("--print-freq", type=int, help="Printing frequency")

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

        for key, val in params.items():
            if default_params[key] is None:  # or not default_params[key]:
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

    args = assert_args(args)

    main(args)
