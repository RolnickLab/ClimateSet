import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_adjacency_graphs(g, path):
    n_graph = g.shape[0]
    _, axes = plt.subplots(ncols=n_graph, nrows=1)

    for i in range(n_graph):
        sns.heatmap(g[i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                    cmap="Blues", xticklabels=False, yticklabels=False)
        axes[i].set_title(f"G_T - {i}")
        axes[i].set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(path, 'adjacencies.png'))
    plt.close()


def plot_adjacency_w(w, path):
    n_w = w.shape[1]
    _, axes = plt.subplots(ncols=n_w, nrows=1)

    for i in range(n_w):
        sns.heatmap(w[:, i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                    cmap="Blues", xticklabels=False, yticklabels=False)
        axes[i].set_title(f"w_d={i}")
        axes[i].set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(path, 'ws.png'))
    plt.close()


def plot_x(x, path):
    d = x.shape[1]
    d_x = x.shape[2]

    for i in range(d):
        _, axes = plt.subplots(nrows=d_x, ncols=1)
        for i_x in range(d_x):
            axes[i_x].plot(x[:, i, i_x])
        plt.savefig(os.path.join(path, f'x_{i}.png'))
        plt.close()


def plot_z(z, path):
    d = z.shape[1]
    k = z.shape[2]

    for i in range(d):
        _, axes = plt.subplots(nrows=k, ncols=1)
        for i_k in range(k):
            axes[i_k].plot(z[:, i, i_k])
        plt.savefig(os.path.join(path, f'z_{i}.png'))
        plt.close()
