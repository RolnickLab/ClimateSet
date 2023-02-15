import numpy as np
import torch
import sklearn
from metrics import shd, mean_corr_coef, precision_recall, edge_errors

from typing import Tuple
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn
from tigramite.models import Prediction
from savar.dim_methods import get_varimax_loadings_standard as varimax
from linear_model import train


def dim_reduc(data, d_z, method='varimax'):
    if method == "varimax":
        modes = varimax(data, max_comps=d_z)

        # Get matrix W, apply it to grid-level
        W = modes['weights']
        # W = modes['unrotated_weights']
        z_hat = data @ W
    else:
        raise ValueError("only varimax is implemented")

    return z_hat, W


def pcmci(z_hat, ind_test, tau_min, tau_max, pc_alpha, alpha=0.05):
    pcmci = PCMCI(dataframe=z_hat, cond_ind_test=ind_test)

    if pc_alpha == 0.:
        pc_alpha = None

    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
    graph = np.zeros_like(results['p_matrix'])
    graph[results['p_matrix'] < alpha] = 1

    # no instantaneous connections
    # if tau_min == 1:
    #     graph = graph[:, :, 1]

    return graph


def varimax_pcmci(data: np.ndarray, idx_train, idx_valid, hp, gt_z, gt_w,
                  gt_graph) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply Varimax+ to find modes and then PCMCI to recover the causal graph
    relating the different modes.
    args:
        data: expected shape (t, d)
        hp: hyperparameters
        gt_z:
        gt_w:
    Returns:
        graph, W, model: 1) the graph between the latents,
        2) the matrix linking the obs to the latents,
        3) the linear model fitted to the data
    """
    if not hp.latent:
        raise NotImplementedError("The case without is not implemented.")
    if hp.instantaneous:
        raise NotImplementedError("The case with contemporaneous relations is not implemented.")

    # Options: ind_test, tau_max, pc_alpha
    # train_data = data[idx_train]
    train_data = data[idx_train]
    valid_data = data[idx_valid]

    tau_min = hp.tau_min
    tau_max = hp.tau
    # d_x = hp.d_x
    d_z = hp.d_z * hp.d
    pc_alpha = hp.pc_alpha
    alpha = hp.alpha

    if hp.ci_test == "linear":
        ind_test = ParCorr()
    elif hp.ci_test == "nonlinear":
        ind_test = CMIknn()
    else:
        raise ValueError(f"{hp.ci_test} is not valid as a CI test. It should be either 'linear' or 'nonlinear'")

    if hp.fct_type == "linear":
        prediction_model = sklearn.linear_model.LinearRegression()
    elif hp.fct_type == "gaussian_process":
        prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
    else:
        raise ValueError(f"{hp.fct_type} is not valid as a type of function. \
                         It should be either 'linear' or 'gaussian_process'")

    # 1 - Apply varimax+ to the data in order to find W
    # (the relations from the grid locations to the modes)
    if not hp.debug_gt_z:
        z_hat, W = dim_reduc(train_data, d_z)
        z_hat_valid = valid_data @ W
    else:
        W = gt_w
        z_hat = gt_z.squeeze(0)
        z_hat = z_hat.squeeze(1)

    # TODO: plot found modes...

    # 2 - Apply PCMCI to the latent variables (modes)
    df_z_hat = pp.DataFrame(z_hat)
    graph = pcmci(df_z_hat, ind_test, tau_min, tau_max, pc_alpha, alpha)

    # 3 - Fit a linear model on training set
    method = "other"

    # Metrics: SHD, Pr/Re, MSE of pred, MCC
    metrics = {"shd": 0.,
               "precision": 0.,
               "recall": 0.,
               "train_mse": 0.,
               "val_mse": 0.,
               "mcc": 0.}

    # TODO: check original method
    if method == "original":
        z = data @ W
        df_z = pp.DataFrame(z)
        pred = Prediction(dataframe=df_z,
                          cond_ind_test=ind_test,
                          prediction_model=prediction_model,
                          data_transform=sklearn.preprocessing.StandardScaler(),
                          train_indices=idx_train,
                          test_indices=idx_valid)
        all_predictors = pred.get_predictors(selected_targets=range(d_z),
                                             steps_ahead=1,
                                             tau_max=tau_max,
                                             pc_alpha=None)

        pred.fit(target_predictors=all_predictors, tau_max=tau_max)
        predicted = pred.predict(list(range(d_z)))

    elif method == "torch_linear":
        train_mse, val_mse, flag_max_iter = train(graph,
                                                  z_hat,
                                                  idx_train,
                                                  idx_valid,
                                                  max_iter=10000,
                                                  batch_size=30)
        metrics['train_mse'] = train_mse
        metrics['val_mse'] = val_mse

    if not hp.no_gt:
        with torch.no_grad():
            graph = np.transpose(graph, (2, 1, 0))
            graph = graph[1:]
            gt_graph = gt_graph[:-1]
            gt_graph = gt_graph[::-1]
            assert graph.shape == gt_graph.shape, f"{graph.shape} != {gt_graph.shape}"

            gt_z = gt_z.reshape(gt_z.shape[0] * gt_z.shape[1], gt_z.shape[2] * gt_z.shape[3])
            gt_z = gt_z[idx_valid]
            # z_hat = z_hat.reshape(z_hat.shape[0] * z_hat.shape[1], z_hat.shape[2] * z_hat.shape[3])

            # find the permutation of Z
            score, cc_program_perm, assignments = mean_corr_coef(gt_z, z_hat_valid, 'pearson')

            permutation = np.zeros((gt_graph.shape[1], gt_graph.shape[1]))
            permutation[np.arange(gt_graph.shape[1]), assignments[1]] = 1
            gt_graph = permutation.T @ gt_graph @ permutation
            # gt_graph = np.swapaxes(gt_graph, 1, 2)

            metrics['mcc'] = score
            metrics['shd'] = shd(graph, gt_graph, True)
            metrics['precision'], metrics['recall'] = precision_recall(graph, gt_graph)
            errors = edge_errors(graph, gt_graph)
            metrics['tp'] = errors['tp']
            metrics['fp'] = errors['fp']
            metrics['tn'] = errors['tn']
            metrics['fn'] = errors['fn']
            metrics['n_edge_gt_graph'] = np.sum(gt_graph)
            metrics['n_edge_learned_graph'] = np.sum(graph)
            print(metrics)
            __import__('ipdb').set_trace()
            print(metrics)

    return graph, W, metrics
