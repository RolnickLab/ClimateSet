import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from metrics import mcc_latent


def moving_average(a: np.ndarray, n: int = 10):
    """
    Returns: the moving average of the array 'a' with a timewindow of 'n'
    """
    # from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Plotter:
    def __init__(self):
        self.mcc = []
        self.assignments = []

    def save(self):
        pass

    def plot(self, learner):
        """
        Main plotting function.
        Plot the learning curves and
        if the ground-truth is known the adjacency and adjacency through time.
        """
        # plot learning curves
        # (for latent models, there is a finer decomposition of the loss)
        if learner.latent:
            # ============================
            # TODO: temporary, remove
            # save matrix W

            w = learner.model.encoder_decoder.get_w().detach().numpy()
            np.save(os.path.join(learner.hp.exp_path, "w_tensor"), w)

            # plot distribution of weights
            if learner.hp.plot_through_time:
                fname = f"w_distr_{learner.iteration}.png"
            else:
                fname = "w_distr.png"
            plt.hist(w.flatten(), bins=50)
            plt.savefig(os.path.join(learner.hp.exp_path, fname))
            plt.close()
            # ============================

            self.plot_learning_curves(train_loss=learner.train_loss_list,
                                      train_recons=learner.train_recons_list,
                                      train_kl=learner.train_kl_list,
                                      valid_loss=learner.valid_loss_list,
                                      valid_recons=learner.valid_recons_list,
                                      valid_kl=learner.valid_kl_list,
                                      best_metrics=learner.best_metrics,
                                      iteration=learner.logging_iter,
                                      plot_through_time=learner.hp.plot_through_time,
                                      path=learner.hp.exp_path)
            losses = [{"name": "sparsity", "data": learner.train_sparsity_reg_list, "s": "-"},
                      {"name": "tr ortho", "data": learner.train_ortho_cons_list, "s": ":"},
                      {"name": "mu ortho", "data": learner.mu_ortho_list, "s": ":"},
                      ]
            # {"name": "tr acyclic", "data": learner.train_acyclic_cons_list, "s": "-"},
            # {"name": "tr connect", "data": learner.train_connect_reg_list, "s": "-"},
            self.plot_learning_curves2(losses=losses,
                                       iteration=learner.logging_iter,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="penalties")
            losses = [{"name": "tr ELBO", "data": learner.train_loss_list, "s": "-."},
                      {"name": "Recons", "data": learner.train_recons_list, "s": "-"},
                      {"name": "KL", "data": learner.train_kl_list, "s": "-"},
                      {"name": "val ELBO", "data": learner.valid_loss_list, "s": "-."},
                      ]
            self.plot_learning_curves2(losses=losses,
                                       iteration=learner.logging_iter,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="losses")
        else:
            self.plot_learning_curves(train_loss=learner.train_loss_list,
                                      valid_loss=learner.valid_loss_list,
                                      iteration=learner.logging_iter,
                                      plot_through_time=learner.hp.plot_through_time,
                                      path=learner.hp.exp_path)

        # TODO: plot the prediction vs gt
        # plot_compare_prediction(x, x_hat)

        # plot the adjacency matrix (learned vs ground-truth)
        adj = learner.model.get_adj().detach().numpy()
        if not learner.no_gt:
            if learner.latent:
                # for latent models, find the right permutation of the latent
                adj_w = learner.model.encoder_decoder.get_w().detach().numpy()
                adj_w2 = learner.model.encoder_decoder.get_w(True).detach().numpy()
                # variables using MCC
                if learner.debug_gt_z:
                    gt_dag = learner.gt_dag
                    gt_w = learner.gt_w
                    self.mcc.append(1.)
                    self.assignments.append(np.arange(learner.gt_dag.shape[1]))
                else:
                    score, cc_program_perm, assignments, z, z_hat = mcc_latent(learner.model, learner.data)
                    permutation = np.zeros((learner.gt_dag.shape[1], learner.gt_dag.shape[1]))
                    permutation[np.arange(learner.gt_dag.shape[1]), assignments[1]] = 1
                    self.mcc.append(score.item())
                    self.assignments.append(assignments[1])

                    gt_dag = permutation.T @ learner.gt_dag @ permutation
                    gt_w = learner.gt_w
                    adj_w = adj_w[:, :, assignments[1]]
                    adj_w2 = adj_w2[:, :, assignments[1]]
                self.save_mcc_and_assignement(learner.hp.exp_path)
            else:
                gt_dag = learner.gt_dag

            self.plot_adjacency_through_time(learner.adj_tt,
                                             gt_dag,
                                             learner.logging_iter,
                                             learner.hp.exp_path,
                                             'transition')
        else:
            gt_dag = None
        self.plot_adjacency_matrix(adj,
                                   gt_dag,
                                   learner.hp.exp_path,
                                   'transition',
                                   learner.no_gt)

        # plot the weights W for latent models (between the latent Z and the X)
        if learner.latent:
            # plot the decoder matrix W
            self.plot_adjacency_matrix_w(adj_w,
                                         gt_w,
                                         learner.hp.exp_path,
                                         'w',
                                         learner.no_gt)
            # plot the encoder matrix W_2
            # gt_w2 = np.swapaxes(gt_w, 1, 2)
            gt_w2 = gt_w
            self.plot_adjacency_matrix_w(adj_w2,
                                         gt_w2,
                                         learner.hp.exp_path,
                                         'encoder_w',
                                         learner.no_gt)
            if not learner.no_gt:
                self.plot_adjacency_through_time_w(learner.adj_w_tt,
                                                   learner.gt_w,
                                                   learner.logging_iter,
                                                   learner.hp.exp_path,
                                                   'w')
            else:
                self.plot_regions_map(adj_w,
                                      learner.data.coordinates,
                                      learner.logging_iter,
                                      learner.hp.plot_through_time,
                                      path=learner.hp.exp_path)


    def plot_compare_prediction(self, x, x_past, x_hat, coordinates: np.ndarray, path: str):
        """
        Plot the predicted x_hat compared to the ground-truth x
        Args:
            x: ground-truth x (for a specific physical variable)
            x_past: ground-truth x at (t-1)
            x_hat: x predicted by the model
            coordinates: xxx
            path: path where to save the plot
        """

        fig = plt.figure()
        fig.suptitle("Ground-truth vs prediction")

        lat = np.unique(coordinates[:, 0])
        lon = np.unique(coordinates[:, 1])
        X, Y = np.meshgrid(lon, lat)

        for i in range(3):
            if i == 0:
                z = x_past
                axes = fig.add_subplot(311)
                axes.set_title("Previous GT")
            if i == 1:
                z = x
                axes = fig.add_subplot(312)
                axes.set_title("Ground-truth")
            if i == 2:
                z = x_hat
                axes = fig.add_subplot(313)
                axes.set_title("Prediction")

            map = Basemap(projection='robin', lon_0=0)
            map.drawcoastlines()
            map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
            # map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])

            Z = z.reshape(X.shape[0], X.shape[1])

            map.contourf(X, Y, Z, latlon=True)

        # plt.colorbar()
        plt.savefig(os.path.join(path, "prediction.png"), format="png")
        plt.close()


    def plot_compare_regions():
        pass


    def plot_regions_map(self, w_adj, coordinates: np.ndarray, iteration: int,
                         plot_through_time: bool, path: str):
        """
        Plot the regions
        Args:
            w_adj: weight of edges between X and latents Z
            coordinates: lat, lon of every grid location
            iteration: number of training iteration
            plot_through_time: if False, overwrite the plot
            path: path where to save the plot
        """

        # plot the map
        map = Basemap(projection='robin', lon_0=0)
        map.drawcoastlines()
        map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
        map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])

        # d = w_adj.shape[0]
        # d_x = w_adj.shape[1]
        d_z = w_adj.shape[2]

        # TODO: make sure it works for multiple features

        # find the argmax per row
        idx = np.argmax(w_adj, axis=2)[0]
        # norms = np.linalg.norm(w_adj, axis=2)[0]
        norms = np.max(w_adj, axis=2)[0]

        # plot the regions
        colors = plt.cm.rainbow(np.linspace(0, 1, d_z))

        for k, color in zip(range(d_z), colors):
            alpha = norms[idx == k] / np.max(norms)
            threshold = np.percentile(alpha, 30)
            alpha[alpha < threshold] = 0
            region = coordinates[idx == k]
            c = np.repeat(np.array([color]), region.shape[0], axis=0)
            map.scatter(x=region[:, 1], y=region[:, 0], c=c, alpha=alpha, s=3, latlon=True)

        if plot_through_time:
            fname = f"spatial_aggregation_{iteration}.png"
        else:
            fname = "spatial_aggregation.png"

        plt.title("Learned regions")
        plt.savefig(os.path.join(path, fname))
        plt.close()


    def plot_learning_curves(self, train_loss: list, train_recons: list = None, train_kl: list = None,
                             valid_loss: list = None, valid_recons: list = None,
                             valid_kl: list = None, best_metrics: dict = None, iteration: int = 0,
                             plot_through_time: bool = False, path: str = ""):
        """ Plot the training and validation loss through time
        Args:
          train_loss: training loss
          train_recons: for latent models, the reconstruction part of the loss
          train_kl: for latent models, the Kullback-Leibler part of the loss
          valid_loss: validation loss (on held-out dataset)
          valid_recons: see train_recons
          valid_kl: see train_kl
          iteration: number of iterations
          plot_through_time: if False, overwrite the plot
          path: path where to save the plot
        """
        # remove first steps to avoid really high values
        start = 1
        t_loss = moving_average(train_loss[start:])
        v_loss = moving_average(valid_loss[start:])

        if train_recons is not None:
            t_recons = moving_average(train_recons[start:])
            t_kl = moving_average(train_kl[start:])
            # v_recons = moving_average(valid_recons[10:])
            # v_kl = moving_average(valid_kl[10:])


        ax = plt.gca()
        # ax.set_ylim([0, 5])
        # ax.set_yscale("log")
        plt.plot(v_loss, label="valid ELBO", color="green")
        if train_recons is not None:
            plt.plot(t_recons, label="tr recons", color="blue")
            plt.axhline(y=best_metrics["recons"], color='blue', linestyle='dotted')
            plt.plot(t_kl, label="tr kl", color="red")
            plt.axhline(y=best_metrics["kl"], color='red', linestyle='dotted')
            plt.plot(t_loss, label="tr ELBO", color="purple")
            plt.axhline(y=best_metrics["elbo"], color='purple', linestyle='dotted')
            # plt.plot(v_recons, label="val recons")
            # plt.plot(v_kl, label="val kl")
        else:
            plt.plot(t_loss, label="tr ELBO")

        if plot_through_time:
            fname = f"loss_{iteration}.png"
        else:
            fname = "loss.png"


        plt.title("Learning curves")
        plt.legend()
        plt.savefig(os.path.join(path, fname))
        plt.close()


    def plot_learning_curves2(self, losses: list, iteration: int = 0, plot_through_time:
                              bool = False, path: str = "", fname = "loss_detailed"):
        """
        Plot all list present in 'losses'.
        Args:
            losses: contains losses, their name, and style
            iteration: number of iterations
            plot_through_time: if False, overwrite the plot
            path: path where to save the plot
        """
        ax = plt.gca()
        if fname != "losses":
            ax.set_yscale("log")

        # compute moving_averages and
        # remove first steps to avoid really high values
        for loss in losses:
            smoothed_loss = moving_average(loss["data"][1:])
            plt.plot(smoothed_loss, label=loss["name"], linestyle=loss["s"])

        if plot_through_time:
            fname = f"{fname}_{iteration}.png"
        else:
            fname = f"{fname}.png"

        plt.title("Learning curves")
        plt.legend()
        plt.savefig(os.path.join(path, fname))
        plt.close()


    # TODO: add no_gt
    def plot_adjacency_matrix(self, mat1: np.ndarray, mat2: np.ndarray, path: str,
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
            nrows = 3

        if tau == 1:
            axes = fig.subplots(nrows=nrows, ncols=1)
            for row in range(nrows):
                if no_gt:
                    ax = axes
                else:
                    ax = axes[row]
                # axes.set_title(f"t - {i+1}")
                if row == 0:
                    sns.heatmap(mat1[0], ax=ax, cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 1:
                    sns.heatmap(mat2[0], ax=ax, cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 2:
                    sns.heatmap(mat1[0] - mat2[0], ax=ax, cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)

        else:
            subfigs = fig.subfigures(nrows=nrows, ncols=1)
            for row in range(nrows):
                if nrows == 1:
                    subfig = subfigs
                else:
                    subfig = subfigs[row]
                subfig.suptitle(f'{subfig_names[row]}')

                axes = subfig.subplots(nrows=1, ncols=tau)
                for i in range(tau):
                    axes[i].set_title(f"t - {i+1}")
                    if row == 0:
                        sns.heatmap(mat1[i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)
                    elif row == 1:
                        sns.heatmap(mat2[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)
                    elif row == 2:
                        sns.heatmap(mat1[i] - mat2[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)

        plt.savefig(os.path.join(path, f'adjacency_{name_suffix}.png'))
        plt.close()


    def plot_adjacency_matrix_w(self, mat1: np.ndarray, mat2: np.ndarray, path: str,
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
        fig.suptitle("Matrices W")

        if no_gt:
            nrows = 1
        else:
            nrows = 3

        if d == 1:
            axes = fig.subplots(nrows=nrows, ncols=1)
            for row in range(nrows):
                if no_gt:
                    ax = axes
                else:
                    ax = axes[row]

                if row == 0:
                    mat = mat1[0]
                elif row == 1:
                    mat = mat2[0]
                else:
                    mat = mat1[0] - mat2[0]

                sns.heatmap(mat, ax=ax, cbar=False, vmin=-1, vmax=1,
                            annot=True, fmt=".5f", cmap="Blues",
                            xticklabels=False, yticklabels=False)

                # if the matrix is small enough, print also the value of each
                # element of W in the heatmap
                # if mat1.size < 500:
                #     for i in range(mat.shape[0]):
                #         for j in range(mat.shape[1]):
                #             text = ax.text(j, i, f"{mat[i, j]:.1f}",
                #                            ha="center", va="center", color="w")


        else:
            subfigs = fig.subfigures(nrows=nrows, ncols=1)

            for row in range(nrows):
                if nrows == 1:
                    subfig = subfigs
                else:
                    subfig = subfigs[row]
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


    def plot_adjacency_through_time(self, w_adj: np.ndarray, gt_dag: np.ndarray, t: int,
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


    def plot_adjacency_through_time_w(self, w_adj: np.ndarray, gt_dag: np.ndarray, t: int,
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


    def save_mcc_and_assignement(self, exp_path):
        np.save(os.path.join(exp_path, "mcc"), np.array(self.mcc))
        np.save(os.path.join(exp_path, "assignments"), np.array(self.assignments))
        if len(self.mcc) > 1:
            fig = plt.figure()
            plt.plot(self.mcc)
            plt.title("MCC score through time")
            fig.savefig(os.path.join(exp_path, f'mcc.png'))
            fig.clf()
