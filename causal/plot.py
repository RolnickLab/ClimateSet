import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def moving_average(a: np.ndarray, n: int = 10):
    # from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot(learner):
    plot_learning_curves(learner.train_loss_list,
                         learner.valid_loss_list,
                         learner.hp.exp_path)
    adj = learner.model.get_adj().detach().numpy().reshape(learner.gt_dag.shape[0], learner.gt_dag.shape[1], -1)
    plot_adjacency_matrix(adj,
                          learner.gt_dag,
                          learner.hp.exp_path)
    plot_adjacency_through_time(learner.adj_tt,
                                learner.gt_dag,
                                learner.iteration,
                                learner.hp.exp_path)


def plot_learning_curves(train_loss: list, valid_loss: list, path: str):
    """ Plot the training and validation loss through time
    :param train_loss: training loss
    :param valid_loss: ground-truth adjacency matrices
    :param path: path where to save the plot
    """
    # remove first steps to avoid really high values
    t_loss = moving_average(train_loss[10:])
    v_loss = moving_average(valid_loss[10:])

    ax = plt.gca()
    # ax.set_ylim([0, 5])
    ax.set_yscale("log")
    plt.plot(t_loss, label="train")
    plt.plot(v_loss, label="valid")
    plt.title("Learning curves")
    plt.legend()
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()


def plot_adjacency_matrix(mat1: np.ndarray, mat2: np.ndarray, path: str):
    """ Plot the adjacency matrices learned and compare it to the ground truth
    :param mat1: learned adjacency matrices
    :param mat2: ground-truth adjacency matrices
    :param path: path where to save the plot
    """
    tau = mat1.shape[0]
    subfig_names = ["Learned", "Ground Truth", "Difference: Learned - GT"]

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Adjacency matrices: learned vs ground-truth")

    if tau == 1:
        axes = fig.subplots(nrows=3, ncols=1)
        for row in range(3):
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
        subfigs = fig.subfigures(nrows=3, ncols=1)
        for row, subfig in enumerate(subfigs):
            subfig.suptitle(f'{subfig_names[row]}')

            axes = subfig.subplots(nrows=1, ncols=tau)
            for i in range(tau):
                axes[i].set_title(f"t - {i+1}")
                if row == 0:
                    sns.heatmap(mat1[i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 1:
                    sns.heatmap(mat2[i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 2:
                    sns.heatmap(mat1[i] - mat2[i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)

    plt.savefig(os.path.join(path, 'adjacency.png'))
    plt.close()


def plot_adjacency_through_time(w_adj: np.ndarray, gt_dag: np.ndarray, t: int, path: str):
    """ Plot the probability of each edges through time up to timestep t
    :param w_adj: weight of edges
    :param gt_dag: ground-truth DAG
    :param t: timestep where to stop plotting
    :param path: path where to save the plot
    """
    taus = w_adj.shape[1]
    d = w_adj.shape[2]
    w_adj = w_adj.reshape(w_adj.shape[0], taus, d, -1)
    fig, ax1 = plt.subplots()

    for tau in range(taus):
        for i in range(d):
            for j in range(w_adj.shape[-1]):
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
    fig.savefig(os.path.join(path, 'adjacency_time.png'))
    fig.clf()
