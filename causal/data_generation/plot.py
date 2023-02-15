import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_adjacency_graphs(g: np.ndarray, path: str):
    """Plot all the graphs G (connections between the Zs) as adjacency matrices

    Args:
        g: an array of graphs G (shape: tau x (d x k) x (d x k))
        path: path where to save the generated figure
    """
    tau = g.shape[0]
    if tau == 1:
        _, axes = plt.subplots(ncols=1, nrows=1)
        sns.heatmap(g[0], ax=axes, cbar=False, vmin=-1, vmax=1,
                    cmap="Blues", xticklabels=False, yticklabels=False)
    else:
        _, axes = plt.subplots(ncols=tau, nrows=1)

        for i in range(tau):
            sns.heatmap(g[i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                        cmap="Blues", xticklabels=False, yticklabels=False)
            axes[i].set_title(f"G_T - {i}")
            axes[i].set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(path, 'adjacencies.png'))
    plt.close()


def plot_adjacency_w(w: np.ndarray, path: str):
    """Plot all the graphs w (connections between the Z and X) as adjacency matrices

    Args:
        w: an array of graphs w (shape: d_x x d x k)
        path: path where to save the generated figure
    """
    n_w = w.shape[1]
    _, axes = plt.subplots(ncols=n_w, nrows=1)

    for i in range(n_w):
        sns.heatmap(w[:, i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                    cmap="Blues", xticklabels=False, yticklabels=False)
        axes[i].set_title(f"w_d={i}")
        axes[i].set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(path, 'ws.png'))
    plt.close()


def plot_x(x: np.ndarray, path: str, t: int = 200, d_max: int = 5, dx_max: int = 8):
    """Plot the timeseries data X from T - t to T.
    (where T is the total number of timesteps)
    Generate a figure for each feature.

    Args:
        x: an array containing the data X (shape: n x t x d x d_x)
        path: path where to save the generated figure
        t: show only the last t timesteps
        d_max: total number of grid-locations to show
        dx_max: total number of grid-locations to show
    """
    d = x.shape[2]
    d_x = x.shape[3]
    i_n = 0

    # if the timeserie is too short, plot it entirely
    # TODO: change code if use multiple timeseries
    if t > x.shape[1]:
        t = x.shape[1]

    if d > d_max:
        d = d_max
    if d_x > dx_max:
        d_x = dx_max

    for i in range(d):
        _, axes = plt.subplots(nrows=d_x, ncols=1)
        if d_x == 1:
            # specific case when there is only one gridcell
            axes.plot(x[i_n, :t, i, 0])
        else:
            for i_x in range(d_x):
                axes[i_x].plot(x[i_n, :t, i, i_x])
        plt.savefig(os.path.join(path, f'x_first_{i}.png'))
        plt.close()

        _, axes = plt.subplots(nrows=d_x, ncols=1)
        if d_x == 1:
            # specific case when there is only one gridcell
            axes.plot(x[i_n, -t:, i, 0])
        else:
            for i_x in range(d_x):
                axes[i_x].plot(x[i_n, -t:, i, i_x])
        plt.savefig(os.path.join(path, f'x_last_{i}.png'))
        plt.close()


def plot_z(z: np.ndarray, path: str, t: int = 200):
    """Plot the timeseries data Z (which are latent) from 0 to t.
    Generate a figure for each feature.

    Args:
        z: an array containing the data Z, shape: (n, t, d, k)
        path: path where to save the generated figure
        t: last timestep to include
    """
    d = z.shape[2]
    k = z.shape[3]
    i_n = 0

    # if the timeserie is too short, plot it entirely
    if t > z.shape[1]:
        t = z.shape[1]

    for i in range(d):
        _, axes = plt.subplots(nrows=k, ncols=1)
        for i_k in range(k):
            axes[i_k].plot(z[i_n, :t, i, i_k])
        plt.savefig(os.path.join(path, f'z_first_{i}.png'))
        plt.close()
        print(f"z0_first_mean: {np.mean(z[0, 100:t, i, 0])}, std: {np.std(z[0, :t, i, 0])}")
        print(f"z1_first_mean: {np.mean(z[0, 100:t, i, 1])}, std: {np.std(z[0, :t, i, 1])}")
        print(f"z2_first_mean: {np.mean(z[0, 100:t, i, 2])}, std: {np.std(z[0, :t, i, 2])}")

    for i in range(d):
        _, axes = plt.subplots(nrows=k, ncols=1)
        for i_k in range(k):
            axes[i_k].plot(z[i_n, -t:, i, i_k])
        plt.savefig(os.path.join(path, f'z_last_{i}.png'))
        plt.close()
        print(f"z0_last_mean: {np.mean(z[0, -t:, i, 0])}, std: {np.std(z[0, -t:, i, 0])}")
        print(f"z1_last_mean: {np.mean(z[0, -t:, i, 1])}, std: {np.std(z[0, -t:, i, 2])}")
        print(f"z2_last_mean: {np.mean(z[0, -t:, i, 1])}, std: {np.std(z[0, -t:, i, 2])}")
