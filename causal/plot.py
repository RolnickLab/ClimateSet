import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def moving_average(a, n=10):
    # from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot(learner):
    plot_learning_curves(learner.train_loss_list,
                         learner.valid_loss_list,
                         learner.hp.exp_path)
    plot_adjacency_matrix(learner.model.get_adj().detach().numpy(),
                          learner.gt_dag.numpy(),
                          learner.hp.exp_path)
    plot_adjacency_through_time(learner.adj_tt,
                                learner.gt_dag.numpy(),
                                learner.iteration,
                                learner.hp.exp_path)


def plot_learning_curves(train_loss, valid_loss, path):
    t_loss = moving_average(train_loss[10:])
    v_loss = moving_average(valid_loss[10:])

    ax = plt.gca()
    ax.set_ylim([0, 30])
    plt.plot(t_loss, label="train")
    plt.plot(v_loss, label="valid")
    plt.legend()
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()


def plot_adjacency_matrix(mat1, mat2, path):
    _, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
    sns.heatmap(mat1, ax=ax1, cbar=False, vmin=-1, vmax=1,
                cmap="hot", xticklabels=False, yticklabels=False)
    sns.heatmap(mat2, ax=ax2, cbar=False, vmin=-1, vmax=1,
                cmap="hot", xticklabels=False, yticklabels=False)
    sns.heatmap(mat1 - mat2, ax=ax3, cbar=False, vmin=-1, vmax=1,
                cmap="hot", xticklabels=False, yticklabels=False)

    ax1.set_title("Learned")
    ax2.set_title("Ground truth")
    ax3.set_title("Learned - GT")

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(path, 'adjacency.png'))
    plt.close()


def plot_adjacency_through_time(w_adj, gt_dag, t, path):
    d = w_adj.shape[1]
    fig, ax1 = plt.subplots()

    for i in range(d):
        for j in range(d):
            if i != j:
                if gt_dag[i, j]:
                    color = 'g'
                    zorder = 2
                else:
                    color = 'r'
                    zorder = 1
                ax1.plot(range(1, t), w_adj[1:t, i, j], color, linewidth=1, zorder=zorder)
    fig.savefig(os.path.join(path, 'adjacency_time.png'))
    fig.clf()
