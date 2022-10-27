import torch
import numpy as np
import os

from geopy import distance
from dag_optim import compute_dag_constraint
from plot import plot, plot_compare_prediction
# from prox import monkey_patch_RMSprop


class TrainingLatent:
    def __init__(self, model, data, hp):
        self.model = model
        self.data = data
        self.hp = hp
        self.latent = hp.latent
        self.debug_gt_z = hp.debug_gt_z
        self.k = hp.k
        self.no_gt = hp.no_gt
        self.gt_dag = data.gt_graph
        self.gt_w = data.gt_w
        self.converged = False
        self.thresholded = False
        self.ended = False

        self.d = data.x.shape[2]
        self.patience = hp.patience
        self.best_valid_loss = np.inf
        self.batch_size = hp.batch_size
        self.tau = hp.tau
        self.d_x = hp.d_x
        self.instantaneous = hp.instantaneous
        self.patience_freq = 50

        self.iteration = 1
        self.logging_iter = 0

        # TODO: put as arguments
        self.train_loss_list = []
        self.train_elbo_list = []
        self.train_recons_list = []
        self.train_kl_list = []
        self.train_sparsity_reg_list = []
        self.train_connect_reg_list = []
        self.train_ortho_cons_list = []
        self.train_acyclic_cons_list = []

        self.valid_loss_list = []
        self.valid_elbo_list = []
        self.valid_recons_list = []
        self.valid_kl_list = []
        self.valid_sparsity_reg_list = []
        self.valid_connect_reg_list = []
        self.valid_ortho_cons_list = []
        self.valid_acyclic_cons_list = []

        # TODO just equal size of G
        if self.instantaneous:
            raise NotImplementedError("Soon")
            self.adj_tt = np.zeros((self.hp.max_iteration, self.tau + 1,
                                    self.d * self.k, self.d * self.k))
        else:
            self.adj_tt = np.zeros((self.hp.max_iteration, self.tau, self.d *
                                    self.k, self.d * self.k))
        if not self.no_gt:
            self.adj_w_tt = np.zeros((self.hp.max_iteration, self.d, self.d_x, self.k))

        # self.model.mask.fix(self.gt_dag)

        # optimizer
        if hp.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr)
        elif hp.optimizer == "rmsprop":
            # TODO: put back
            # monkey_patch_RMSprop(torch.optim.RMSprop)
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=hp.lr)
        else:
            raise NotImplementedError("optimizer {} is not implemented".format(hp.optimizer))

        # compute constraint normalization
        with torch.no_grad():
            full_adjacency = torch.ones((model.d, model.d)) - torch.eye(model.d)
            self.acyclic_constraint_normalization = compute_dag_constraint(full_adjacency).item()

            # TODO: to change, use Frobenius norm!
            # orthogonal up to a small variation epsilon
            if self.latent:
                # eps = 1e-8
                # almost_orthogonal = torch.zeros((model.d_x, model.k))
                # partition = np.array([model.d_x // model.k + (1 if x < model.d_x % model.k else 0)
                #                       for x in range(model.k)])
                # idx = np.repeat(np.arange(model.k), partition)
                # almost_orthogonal[np.arange(model.d_x), idx] = 1
                # almost_orthogonal = almost_orthogonal / np.linalg.norm(almost_orthogonal, axis=0)
                # almost_orthogonal = almost_orthogonal + eps

                # self.ortho_normalization = self.d * torch.norm(almost_orthogonal.T @ almost_orthogonal
                #                                                - torch.eye(model.k), p=2)

                # expected frobenius norm of A^TA where A_ij \sim U([0, 1])
                self.ortho_normalization = self.d_x * self.k
                # 1./16 * self.d_x ** 2 * self.k ** 2
                # + 7./144 * self.d_x * self.k

    def log_losses(self):
        # train
        self.train_loss_list.append(self.train_loss)
        self.train_recons_list.append(-self.train_recons)
        self.train_kl_list.append(self.train_kl)

        self.train_sparsity_reg_list.append(self.train_sparsity_reg)
        self.train_connect_reg_list.append(self.train_connect_reg)
        self.train_ortho_cons_list.append(self.train_ortho_cons)
        self.train_acyclic_cons_list.append(self.train_acyclic_cons)

        # valid
        self.valid_loss_list.append(self.valid_loss)
        self.valid_recons_list.append(-self.valid_recons)
        self.valid_kl_list.append(self.valid_kl)

        self.valid_sparsity_reg_list.append(self.valid_sparsity_reg)
        self.valid_connect_reg_list.append(self.valid_connect_reg)
        self.valid_ortho_cons_list.append(self.valid_ortho_cons)
        self.valid_acyclic_cons_list.append(self.valid_acyclic_cons)

        # self.mu_ortho_list.append(self.mu_ortho)
        self.adj_tt[self.iteration] = self.model.get_adj().detach().numpy()
        w = self.model.encoder_decoder.get_w().detach().numpy()
        if not self.no_gt:
            self.adj_w_tt[self.iteration] = w

    def print_results(self):
        print("============================================================")
        print(f"Iteration #{self.iteration}")
        print(f"Converged: {self.converged}")

        print(f"train_nll: {self.train_nll:.4f}")
        print(f"train_recons: {self.train_recons:.4f}")
        print(f"train_kl: {self.train_kl:.4f}")

        print(f"train_sparsity_reg: {self.train_sparsity_reg:.1e}")
        print(f"train_connect_reg: {self.train_connect_reg:.1e}")

        print(f"ortho cons: {self.train_ortho_cons:.1e}")
        print(f"ortho delta_gamma: {self.ALM_ortho.delta_gamma}")
        print(f"ortho gamma: {self.ALM_ortho.gamma}")
        print(f"ortho mu: {self.ALM_ortho.mu}")

        if self.instantaneous:
            print(f"acyclic cons: {self.train_acyclic_cons:.4f}")
            print(f"acyclic gamma: {self.QPM_ortho.mu}")
        print("-------------------------------")

        print(f"valid_nll: {self.valid_nll:.4f}")
        # print(f"valid_recons: {self.valid_recons:.4f}")
        # print(f"valid_kl: {self.valid_kl:.4f}")
        print(f"patience: {self.patience}")

    def train_with_QPM(self):
        # TODO: add comment

        # initialize ALM/QPM for orthogonality and acyclicity constraints
        self.ALM_ortho = ALM(self.hp.ortho_mu_init,
                             self.hp.ortho_mu_mult_factor,
                             self.hp.ortho_omega_gamma,
                             self.hp.ortho_omega_mu,
                             self.hp.ortho_h_threshold,
                             self.hp.ortho_min_iter_convergence)
        if self.instantaneous:
            self.QPM_acyclic = ALM(self.hp.acyclic_mu_init,
                                   self.hp.acyclic_mu_mult_factor,
                                   self.hp.acyclic_omega_gamma,
                                   self.hp.acyclic_omega_mu,
                                   self.hp.acyclic_h_threshold,
                                   self.hp.acyclic_min_iter_convergence)

        while self.iteration < self.hp.max_iteration and not self.ended:

            # train and valid step
            self.train_step()
            if self.iteration % self.hp.valid_freq == 0:
                self.logging_iter += 1
                x, y, y_pred = self.valid_step()
                self.log_losses()

                # print and plot losses
                if self.iteration % (self.hp.valid_freq * self.hp.print_freq) == 0:
                    self.print_results()
                if self.logging_iter > 10 and self.iteration % (self.hp.valid_freq * self.hp.plot_freq) == 0:
                    plot(self)

                    if self.no_gt:
                        plot_compare_prediction(x[0, -1].detach().numpy(),
                                                y[0].detach().numpy(),
                                                y_pred[0].detach().numpy(),
                                                self.data.coordinates,
                                                self.hp.exp_path)

            # train in 3 phases: first with QPM, then until the likelihood
            # remain stable, then continue after thresholding the adjacency
            # matrix
            if not self.converged:
                # train with penalty method
                if self.iteration % self.hp.valid_freq == 0:
                    self.ALM_ortho.update(self.iteration,
                                          self.valid_ortho_cons_list,
                                          self.valid_loss_list)
                    self.converged = self.ALM_ortho.has_converged

                    if self.ALM_ortho.has_increased_mu:
                        if self.hp.optimizer == "sgd":
                            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hp.lr)
                        elif self.hp.optimizer == "rmsprop":
                            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.hp.lr)

                    if self.instantaneous:
                        self.QPM_acyclic.update(self.iteration,
                                                self.valid_acyclic_cons_list,
                                                self.valid_loss_list)
                        self.converged = self.converged & self.QPM_acyclic.has_converged
                        # TODO: add optimizer reinit
            else:
                # continue training without penalty method
                if not self.thresholded and self.iteration % self.patience_freq == 0:
                    if not self.has_patience(self.hp.patience, self.valid_loss):
                        self.threshold()
                        self.patience = self.hp.patience_post_thresh
                        self.best_valid_loss = np.inf
                # continue training after thresholding
                else:
                    if self.iteration % self.patience_freq == 0:
                        if not self.has_patience(self.hp.patience_post_thresh, self.valid_loss):
                            self.ended = True

            self.iteration += 1

        # final plotting and printing
        plot(self)
        self.print_results()

        # save tensor W
        w = self.model.encoder_decoder.get_w().detach().numpy()
        np.save("w_tensor", w)

    def has_patience(self, patience_init, valid_loss):
        if self.patience > 0:
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience = patience_init
                print(f"Best valid loss: {self.best_valid_loss}")
            else:
                self.patience -= 1
            return True
        else:
            return False

    def train_step(self):
        self.model.train()

        # sample data
        x, y, z = self.data.sample(self.batch_size, valid=False)
        nll, recons, kl, pred = self.get_nll(x, y, z)

        # compute regularisations (sparsity and connectivity)
        sparsity_reg = self.get_regularisation()
        connect_reg = torch.tensor([0.])
        if self.hp.latent and self.hp.reg_coeff_connect > 0:
            connect_reg = self.connectivity_reg()

        # compute constraints (acyclicity and orthogonality)
        h_acyclic = torch.tensor([0.])
        h_ortho = torch.tensor([0.])
        if self.instantaneous and not self.converged:
            h_acyclic = self.get_acyclicity_violation()
        # if self.hp.reg_coeff_connect:
        h_ortho = self.get_ortho_constraint(self.model.encoder_decoder.get_w())

        # compute total loss
        loss = nll + sparsity_reg + connect_reg
        loss = loss + self.ALM_ortho.gamma * h_ortho + \
            0.5 * self.ALM_ortho.mu * h_ortho ** 2
        if self.instantaneous:
            loss = loss + 0.5 * self.QPM_acyclic.mu * h_acyclic ** 2

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        _, _ = self.optimizer.step() if self.hp.optimizer == "rmsprop" else self.optimizer.step(), self.hp.lr

        # projection of the gradient for w
        self.model.encoder_decoder.project_gradient()

        self.train_loss = loss.item()
        self.train_nll = nll.item()
        self.train_recons = recons.item()
        self.train_kl = kl.item()
        self.train_sparsity_reg = sparsity_reg.item()
        self.train_connect_reg = connect_reg.item()
        self.train_ortho_cons = h_ortho.item()
        self.train_acyclic_cons = h_acyclic.item()

        return pred

    def valid_step(self):
        # TODO: merge valid and train step? almost the same
        self.model.eval()

        # sample data
        # data = self.test_data
        # idx = np.random.choice(data.shape[0], size=100, replace=False)
        # x = data[idx]
        x, y, z = self.data.sample(self.data.n_valid - self.data.tau, valid=True)
        nll, recons, kl, y_pred = self.get_nll(x, y, z)

        # compute regularisations (sparsity and connectivity)
        sparsity_reg = self.get_regularisation()
        connect_reg = torch.tensor([0.])
        if self.hp.latent and self.hp.reg_coeff_connect > 0:
            connect_reg = self.connectivity_reg()

        # compute constraints (acyclicity and orthogonality)
        h_acyclic = torch.tensor([0.])
        h_ortho = torch.tensor([0.])
        if self.instantaneous and not self.converged:
            h_acyclic = self.get_acyclicity_violation()
        h_ortho = self.get_ortho_constraint(self.model.encoder_decoder.get_w())

        # compute total loss
        loss = nll + sparsity_reg + connect_reg
        loss = loss + self.ALM_ortho.gamma * h_ortho + \
            0.5 * self.ALM_ortho.mu * h_ortho ** 2
        if self.instantaneous:
            loss = loss + 0.5 * self.QPM_acyclic.mu * h_acyclic ** 2

        self.valid_loss = loss.item()
        self.valid_nll = nll.item()
        self.valid_recons = recons.item()
        self.valid_kl = kl.item()
        self.valid_sparsity_reg = sparsity_reg.item()
        self.valid_connect_reg = connect_reg.item()
        self.valid_ortho_cons = h_ortho.item()
        self.valid_acyclic_cons = h_acyclic.item()

        return x, y, y_pred

    def get_acyclicity_violation(self) -> torch.Tensor:
        adj = self.model.get_adj()[-1].view(self.d, self.d)
        # __import__('ipdb').set_trace()
        h = compute_dag_constraint(adj) / self.acyclic_constraint_normalization

        return h

    def get_nll(self, x, y, z=None) -> torch.Tensor:
        elbo, recons, kl, pred = self.model(x, y, z, self.iteration)
        return -elbo, recons, kl, pred

    # def get_nll(self, x, y) -> torch.Tensor:
    #     density_param = self.model(x)
    #     mu = density_param[:, :, :, 0].view(-1, 1)
    #     std = density_param[:, :, :, 1].view(-1, 1)

    #     nll = -1/(y.shape[0] * y.shape[1] * y.shape[2]) * self.model.get_likelihood(y, mu, std, self.iteration)
    #     return nll

    def get_regularisation(self) -> float:
        reg = self.sparsity_reg()
        return reg

    def sparsity_reg(self):
        adj = self.model.get_adj()
        reg = self.hp.reg_coeff * torch.norm(adj, p=1)
        reg /= adj.shape[0] ** 2

        return reg

    def connectivity_reg_complete(self):
        """
        Calculate the connectivity constraint, ie the sum of all the distances
        inside each clusters.
        """
        c = torch.tensor([0.])
        w = self.model.encoder_decoder.get_w()
        d = self.data.distances
        for i in self.d:
            for k in self.k:
                c = c + torch.sum(torch.outer(w[i, :, k], w[i, :, k]) * d)
        return self.hp.reg_coeff_connect * c

    def connectivity_reg(self, ratio: float = 0.0005):
        """
        Calculate a connectivity regularisation only on a subsample of the
        complete data.
        """
        c = torch.tensor([0.])
        w = self.model.encoder_decoder.get_w()
        n = int(self.d_x * ratio)
        points = np.random.choice(np.arange(self.d_x), n)

        if n <= 1:
            raise ValueError("You should use a higher value for the ratio of considered points for the connectivity constraint")

        for d in range(self.d):
            for k in range(self.k):
                for i, c1 in enumerate(self.data.coordinates[points]):
                    for j, c2 in enumerate(self.data.coordinates[points]):
                        if i > j:
                            dist = distance.geodesic(c1, c2).km
                            c = c + w[d, i, k] * w[d, j, k] * dist
        return self.hp.reg_coeff_connect * c

    def threshold(self):
        with torch.no_grad():
            thresholded_adj = (self.model.get_adj() > 0.5).type(torch.Tensor)
            self.model.mask.fix(thresholded_adj)
        self.thresholded = True
        print("Thresholding ================")

    def save_results(self):
        # TODO
        pass

    # if not self.latent:
    #     raise ValueError("The orthogonality constraint only makes sense \
    #                      when there is latent variables (spatial agg.)")
    def get_ortho_constraint(self, w: torch.Tensor) -> float:
        constraint = torch.tensor([0.])
        k = w.size(2)
        for i in range(w.size(0)):
            constraint = constraint + torch.norm(w[i].T @ w[i] - torch.eye(k), p=2)

        return constraint / self.ortho_normalization


class ALM:
    """
    Augmented Lagrangian Method
    To use the quadratic penalty method (e.g. for the acyclicity constraint),
    just ignore 'self.mu'
    """
    def __init__(self,
                 mu_init: float,
                 mu_mult_factor: float,
                 omega_gamma: float,
                 omega_mu: float,
                 h_threshold: float,
                 min_iter_convergence: int):
        self.gamma = 0
        self.delta_gamma = -np.inf
        self.mu = mu_init
        self.min_iter_convergence = min_iter_convergence
        self.h_threshold = h_threshold
        self.omega_mu = omega_mu
        self.omega_gamma = omega_gamma
        self.mu_mult_factor = mu_mult_factor
        self.stop_crit_window = 100
        self.constraint_violation = []
        self.has_converged = False

    def _compute_delta_gamma(self, iteration: int, val_loss: list):
        # compute delta for gamma
        if iteration >= 2 * self.stop_crit_window and \
           iteration % (2 * self.stop_crit_window) == 0:
            t0, t_half, t1 = val_loss[-3], val_loss[-2], val_loss[-1]

            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if not (min(t0, t1) < t_half < max(t0, t1)):
                self.delta_gamma = -np.inf
            else:
                self.delta_gamma = (t1 - t0) / self.stop_crit_window
        else:
            self.delta_gamma = -np.inf  # do not update gamma nor mu

    def update(self, iteration: int, h_list: list, val_loss: list):
        """
        Update the value of mu and gamma. Return True if it has converged.
        Args:
            iteration: number of training iterations completed
            h_list: list of the values of the constraint
            val_loss: list of validation loss
        """
        self.has_increased_mu = False

        if len(val_loss) >= 3:
            h = h_list[-1]
            past_h = h_list[-2]

            # check if QPM has converged
            if iteration > self.min_iter_convergence and h <= self.h_threshold:
                self.has_converged = True
            else:
                # update delta_gamma
                self._compute_delta_gamma(iteration, val_loss)

                # if we have found a stationary point of the augmented loss
                if abs(self.delta_gamma) < self.omega_gamma or self.delta_gamma > 0:
                    self.gamma = self.mu * h

                    self.constraint_violation.append(h)

                    # increase mu if the constraint has sufficiently decreased
                    # since the last subproblem
                    if len(self.constraint_violation) >= 2:
                        if h > self.omega_mu * self.constraint_violation[-2]:
                            self.mu *= self.mu_mult_factor
                            self.has_increased_mu = True
