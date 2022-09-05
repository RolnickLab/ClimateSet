import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import mean_corr_coef


def moving_average(a: np.ndarray, n: int = 10):
    """
    Returns: the moving average of the array 'a' with a timewindow of 'n'
    """
    # from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot(learner):
    """
    Main plotting function.
    Plot the learning curves and
    if the ground-truth is known the adjacency and adjacency through time.
    """
    # plot learning curves
    # (for latent models, there is a finer decomposition of the loss)
    if learner.latent:
        plot_learning_curves(train_loss=learner.train_loss_list,
                             train_recons=learner.train_recons_list,
                             train_kl=learner.train_kl_list,
                             valid_loss=learner.valid_loss_list,
                             valid_recons=learner.valid_recons_list,
                             valid_kl=learner.valid_kl_list,
                             path=learner.hp.exp_path)
    else:
        plot_learning_curves(train_loss=learner.train_loss_list,
                             valid_loss=learner.valid_loss_list,
                             path=learner.hp.exp_path)

    # plot the adjacency matrix (learned vs ground-truth)
    if not learner.no_gt:
        adj = learner.model.get_adj().detach().numpy()
        if learner.latent:
            # for latent models, find the right permutation of the latent
            # variables using MCC
            score, cc_program_perm, assignments, z, z_hat = mean_corr_coef(learner.model, learner.data)
            print(score)
            print(assignments)
            permutation = np.zeros((learner.gt_dag.shape[1], learner.gt_dag.shape[1]))
            permutation[np.arange(learner.gt_dag.shape[1]), assignments[1]] = 1
            gt_dag = permutation.T @ learner.gt_dag @ permutation
        else:
            gt_dag = learner.gt_dag

        plot_adjacency_matrix(adj,
                              gt_dag,
                              learner.hp.exp_path,
                              'transition',
                              learner.no_gt)
        plot_adjacency_through_time(learner.adj_tt,
                                    gt_dag,
                                    learner.iteration,
                                    learner.hp.exp_path,
                                    'transition')

    # plot the weights W for latent models (between the latent Z and the X)
    if learner.latent:
        adj_w = learner.model.encoder_decoder.get_w().detach().numpy()
        plot_adjacency_matrix_w(adj_w,
                                learner.gt_w,
                                learner.hp.exp_path,
                                'w',
                                learner.no_gt)
        if not learner.no_gt:
            plot_adjacency_through_time_w(learner.adj_w_tt,
                                          learner.gt_w,
                                          learner.iteration,
                                          learner.hp.exp_path,
                                          'w')


def plot_learning_curves(train_loss: list, train_recons: list = None, train_kl: list = None,
                         valid_loss: list = None, valid_recons: list = None, valid_kl: list = None, path: str = ""):
    """ Plot the training and validation loss through time
    Args:
      train_loss: training loss
      train_recons: for latent models, the reconstruction part of the loss
      train_kl: for latent models, the Kullback-Leibler part of the loss
      valid_loss: validation loss (on held-out dataset)
      valid_recons: see train_recons
      valid_kl: see train_kl
      path: path where to save the plot
    """
    # remove first steps to avoid really high values
    t_loss = moving_average(train_loss[10:])
    v_loss = moving_average(valid_loss[10:])
    if train_recons is not None:
        t_recons = moving_average(train_recons[10:])
        t_kl = moving_average(train_kl[10:])
        # v_recons = moving_average(valid_recons[10:])
        # v_kl = moving_average(valid_kl[10:])

    ax = plt.gca()
    # ax.set_ylim([0, 5])
    ax.set_yscale("log")
    plt.plot(v_loss, label="valid")
    if train_recons is not None:
        plt.plot(t_recons, label="tr recons")
        plt.plot(t_kl, label="tr kl")
        # plt.plot(v_recons, label="val recons")
        # plt.plot(v_kl, label="val kl")
    else:
        plt.plot(t_loss, label="train")
    plt.title("Learning curves")
    plt.legend()
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()


# TODO: add no_gt
def plot_adjacency_matrix(mat1: np.ndarray, mat2: np.ndarray, path: str,
                          name_suffix: str, no_gt: bool = False):
    """ Plot the adjacency matrices learned and compare it to the ground truth,
    the first dimension of the matrix should be the time (tau)
    Args:
      mat1: learned adjacency matrices
      mat2: ground-truth adjacency matrices
      path: path where to save the plot
      name_suffix: suffix for the name of the plot
      no_gt: if True, does not use the ground-truth graph
    """
    tau = mat1.shape[0]

    subfig_names = ["Learned", "Ground Truth", "Difference: Learned - GT"]

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Adjacency matrices: learned vs ground-truth")

    if no_gt:
        nrows = 1
    else:
        n_rows = 3

    if tau == 1:
        axes = fig.subplots(nrows=no_gt, ncols=1)
        for row in range(no_gt):
            # axes.set_title(f"t - {i+1}")
            if row == 0:
                sns.heatmap(mat1[0], ax=axes[row], cbar=False, vmin=-1, vmax=1,
                            cmap="Blues", xticklabels=False, yticklabels=False)
            elif row == 1:
                sns.heatmap(mat2[0], ax=axes[row], cbar=False, vmin=-1, vmax=1,
                            cmap="Blues", xticklabels=False, yticklabels=False)
            elif row == 2:
                sns.heatmap(mat1[0] - mat2[0], ax=axes[row], cbar=False, vmin=-1, vmax=1,
                            cmap="Blues", xticklabels=False, yticklabels=False)

    else:
        subfigs = fig.subfigures(nrows=no_gt, ncols=1)
        for row, subfig in enumerate(subfigs):
            subfig.suptitle(f'{subfig_names[row]}')

            axes = subfig.subplots(nrows=1, ncols=tau)
            for i in range(tau):
                axes[i].set_title(f"t - {i+1}")
                if row == 0:
                    sns.heatmap(mat1[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 1:
                    sns.heatmap(mat2[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 2:
                    sns.heatmap(mat1[tau - i - 1] - mat2[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)

    plt.savefig(os.path.join(path, f'adjacency_{name_suffix}.png'))
    plt.close()


def plot_adjacency_matrix_w(mat1: np.ndarray, mat2: np.ndarray, path: str,
                            name_suffix: str, no_gt: bool = False):
    """ Plot the adjacency matrices learned and compare it to the ground truth,
    the first dimension of the matrix should be the features (d)
    Args:
      mat1: learned adjacency matrices
      mat2: ground-truth adjacency matrices
      path: path where to save the plot
      name_suffix: suffix for the name of the plot
      no_gt: if True, does not use ground-truth W
    """
    d = mat1.shape[0]
    subfig_names = ["Learned", "Ground Truth", "Difference: Learned - GT"]

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Adjacency matrices: learned vs ground-truth")

    if no_gt:
        nrows = 1
    else:
        n_rows = 3

    if d == 1:
        axes = fig.subplots(nrows=nrows, ncols=1)
        for row in range(nrows):
            # axes.set_title(f"t - {i+1}")
            if row == 0:
                sns.heatmap(mat1[0], ax=axes[row], cbar=False, vmin=-1, vmax=1,
                            cmap="Blues", xticklabels=False, yticklabels=False)
            elif row == 1:
                sns.heatmap(mat2[0], ax=axes[row], cbar=False, vmin=-1, vmax=1,
                            cmap="Blues", xticklabels=False, yticklabels=False)
            elif row == 2:
                sns.heatmap(mat1[0] - mat2[0], ax=axes[row], cbar=False, vmin=-1, vmax=1,
                            cmap="Blues", xticklabels=False, yticklabels=False)

    else:
        subfigs = fig.subfigures(nrows=nrows, ncols=1)
        for row, subfig in enumerate(subfigs):
            subfig.suptitle(f'{subfig_names[row]}')

            axes = subfig.subplots(nrows=1, ncols=d)
            for i in range(d):
                axes[i].set_title(f"d = {i}")
                if row == 0:
                    sns.heatmap(mat1[d - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 1:
                    sns.heatmap(mat2[d - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 2:
                    sns.heatmap(mat1[d - i - 1] - mat2[d - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)

    plt.savefig(os.path.join(path, f'adjacency_{name_suffix}.png'))
    plt.close()


def plot_adjacency_through_time(w_adj: np.ndarray, gt_dag: np.ndarray, t: int,
                                path: str, name_suffix: str):
    """ Plot the probability of each edges through time up to timestep t
    Args:
      w_adj: weight of edges
      gt_dag: ground-truth DAG
      t: timestep where to stop plotting
      path: path where to save the plot
      name_suffix: suffix for the name of the plot
    """
    taus = w_adj.shape[1]
    d = w_adj.shape[2]  # * w_adj.shape[3]
    w_adj = w_adj.reshape(w_adj.shape[0], taus, d, d)
    fig, ax1 = plt.subplots()

    for tau in range(taus):
        for i in range(d):
            for j in range(d):
                # plot in green edges that are in the gt_dag
                # otherwise in red
                if gt_dag[tau, i, j]:
                    color = 'g'
                    zorder = 2
                else:
                    color = 'r'
                    zorder = 1
                ax1.plot(range(1, t), w_adj[1:t, tau, i, j], color, linewidth=1, zorder=zorder)
    fig.suptitle("Learned adjacencies through time")
    fig.savefig(os.path.join(path, f'adjacency_time_{name_suffix}.png'))
    fig.clf()


def plot_adjacency_through_time_w(w_adj: np.ndarray, gt_dag: np.ndarray, t: int,
                                  path: str, name_suffix: str):
    """ Plot the probability of each edges through time up to timestep t
    Args:
      w_adj: weight of edges
      gt_dag: ground-truth DAG
      t: timestep where to stop plotting
      path: path where to save the plot
      name_suffix: suffix for the name of the plot
    """
    tau = w_adj.shape[1]
    dk = w_adj.shape[2]
    dk = w_adj.shape[3]
    # w_adj = w_adj.reshape(w_adj.shape[0], taus, d, d)
    fig, ax1 = plt.subplots()

    for i in range(tau):
        for j in range(dk):
            for k in range(dk):
                ax1.plot(range(1, t), np.abs(w_adj[1:t, i, j, k] - gt_dag[i, j, k]), linewidth=1)
    fig.suptitle("Learned adjacencies through time")
    fig.savefig(os.path.join(path, f'adjacency_time_{name_suffix}.png'))
    fig.clf()
