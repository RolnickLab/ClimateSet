import torch
import numpy as np

from geopy import distance
from dag_optim import compute_dag_constraint
from plot import Plotter
from utils import ALM
from prox import monkey_patch_RMSprop


class TrainingLatent:
    def __init__(self, model, data, hp, best_metrics):
        self.model = model
        self.data = data
        self.hp = hp
        self.best_metrics = best_metrics

        self.latent = hp.latent
        self.no_gt = hp.no_gt
        self.debug_gt_z = hp.debug_gt_z
        self.gt_dag = data.gt_graph
        self.gt_w = data.gt_w
        self.d_z = hp.d_z

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
        self.converged = False
        self.thresholded = False
        self.ended = False

        self.train_loss_list = []
        self.train_elbo_list = []
        self.train_recons_list = []
        self.train_kl_list = []
        self.train_sparsity_reg_list = []
        self.train_connect_reg_list = []
        self.train_ortho_cons_list = []
        self.train_acyclic_cons_list = []
        self.mu_ortho_list = []
        self.h_ortho_list = []

        self.valid_loss_list = []
        self.valid_elbo_list = []
        self.valid_recons_list = []
        self.valid_kl_list = []
        self.valid_sparsity_reg_list = []
        self.valid_connect_reg_list = []
        self.valid_ortho_cons_list = []
        self.valid_acyclic_cons_list = []

        self.plotter = Plotter()

        if self.instantaneous:
            raise NotImplementedError("Soon")
            self.adj_tt = np.zeros((int(self.hp.max_iteration / self.hp.valid_freq), self.tau + 1,
                                    self.d * self.d_z, self.d * self.d_z))
        else:
            self.adj_tt = np.zeros((int(self.hp.max_iteration / self.hp.valid_freq), self.tau, self.d *
                                    self.d_z, self.d * self.d_z))
        if not self.no_gt:
            self.adj_w_tt = np.zeros((int(self.hp.max_iteration / self.hp.valid_freq), self.d, self.d_x, self.d_z))

        # self.model.mask.fix(self.gt_dag)

        # optimizer
        if hp.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr)
        elif hp.optimizer == "rmsprop":
            # TODO: put back
            monkey_patch_RMSprop(torch.optim.RMSprop)
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=hp.lr)
        else:
            raise NotImplementedError("optimizer {} is not implemented".format(hp.optimizer))

        # compute constraint normalization
        with torch.no_grad():
            full_adjacency = torch.ones((model.d, model.d)) - torch.eye(model.d)
            self.acyclic_constraint_normalization = compute_dag_constraint(full_adjacency).item()

            if self.latent:
                # expected frobenius norm of A^TA where A_ij \sim U([0, 1])
                self.ortho_normalization = (self.d_x * self.d_z) ** 2 / 10000
                # 1./16 * self.d_x ** 2 * self.d_z ** 2
                # + 7./144 * self.d_x * self.d_z

    def train_with_QPM(self):
        """
        Optimize a problem under constraint using the Augmented Lagragian
        method (or QPM). We train in 3 phases: first with ALM, then until
        the likelihood remain stable, then continue after thresholding
        the adjacency matrix
        """

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
                    self.plotter.plot(self)

                    # if self.no_gt:
                    #     plot_compare_prediction(x[0, -1].detach().numpy(),
                    #                             y[0].detach().numpy(),
                    #                             y_pred[0].detach().numpy(),
                    #                             self.data.coordinates,
                    #                             self.hp.exp_path)

            if not self.converged:
                # train with penalty method
                if self.iteration % self.hp.valid_freq == 0:
                    self.ALM_ortho.update(self.iteration,
                                          self.valid_ortho_cons_list,
                                          self.valid_loss_list)
                    if self.iteration > 20000:
                        self.converged = self.ALM_ortho.has_converged
                        # self.converged = False
                    else:
                        self.converged = False

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
        self.plotter.plot(self)
        self.print_results()

        # save tensor W
        w = self.model.encoder_decoder.get_w().detach().numpy()
        np.save("w_tensor", w)

    def train_step(self):
        self.model.train()

        # sample data
        x, y, z = self.data.sample(self.batch_size, valid=False)
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
        # if self.hp.reg_coeff_connect:
        h_ortho = self.get_ortho_violation(self.model.encoder_decoder.get_w())

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

        return x, y, y_pred

    def valid_step(self):
        self.model.eval()

        # sample data
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
        h_ortho = self.get_ortho_violation(self.model.encoder_decoder.get_w())

        # compute total loss
        loss = nll + sparsity_reg + connect_reg
        # loss = loss + self.ALM_ortho.gamma * h_ortho + \
        #     0.5 * self.ALM_ortho.mu * h_ortho ** 2
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

    def has_patience(self, patience_init, valid_loss):
        """
        Check if the validation loss has not improved for
        'patience' steps
        """
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

    def threshold(self):
        """Consider that the graph has been found. Convert it to
        a binary graph and fix it."""
        with torch.no_grad():
            thresholded_adj = (self.model.get_adj() > 0.5).type(torch.Tensor)
            self.model.mask.fix(thresholded_adj)
        self.thresholded = True
        print("Thresholding ================")

    def log_losses(self):
        """Append in lists values of the losses and more"""
        # train
        self.train_loss_list.append(-self.train_loss)
        self.train_recons_list.append(self.train_recons)
        self.train_kl_list.append(self.train_kl)

        self.train_sparsity_reg_list.append(self.train_sparsity_reg)
        self.train_connect_reg_list.append(self.train_connect_reg)
        self.train_ortho_cons_list.append(self.train_ortho_cons)
        self.train_acyclic_cons_list.append(self.train_acyclic_cons)

        # valid
        self.valid_loss_list.append(-self.valid_loss)
        self.valid_recons_list.append(self.valid_recons)
        self.valid_kl_list.append(self.valid_kl)

        self.valid_sparsity_reg_list.append(self.valid_sparsity_reg)
        self.valid_connect_reg_list.append(self.valid_connect_reg)
        self.valid_ortho_cons_list.append(self.valid_ortho_cons)
        self.valid_acyclic_cons_list.append(self.valid_acyclic_cons)

        self.mu_ortho_list.append(self.ALM_ortho.mu)

        self.adj_tt[int(self.iteration / self.hp.valid_freq)] = self.model.get_adj().detach().numpy()
        w = self.model.encoder_decoder.get_w().detach().numpy()
        if not self.no_gt:
            self.adj_w_tt[int(self.iteration / self.hp.valid_freq)] = w

    def print_results(self):
        """Print values of many variable: losses, constraint violation, etc.
        at the frequency self.hp.print_freq"""
        print("============================================================")
        print(f"Iteration #{self.iteration}")
        print(f"Converged: {self.converged}")

        print(f"ELBO: {self.train_nll:.4f}")
        print(f"Recons: {self.train_recons:.4f}")
        print(f"KL: {self.train_kl:.4f}")

        print(f"Sparsity_reg: {self.train_sparsity_reg:.1e}")
        # print(f"Connect_reg: {self.train_connect_reg:.1e}")

        print(f"ortho cons: {self.train_ortho_cons:.1e}")
        # print(f"ortho delta_gamma: {self.ALM_ortho.delta_gamma}")
        # print(f"ortho gamma: {self.ALM_ortho.gamma}")
        print(f"ortho mu: {self.ALM_ortho.mu}")

        if self.instantaneous:
            print(f"acyclic cons: {self.train_acyclic_cons:.4f}")
            print(f"acyclic gamma: {self.QPM_ortho.mu}")
        print("-------------------------------")

        print(f"valid_ELBO: {self.valid_nll:.4f}")
        # print(f"valid_recons: {self.valid_recons:.4f}")
        # print(f"valid_kl: {self.valid_kl:.4f}")
        print(f"patience: {self.patience}")

    def get_nll(self, x, y, z=None) -> torch.Tensor:
        elbo, recons, kl, pred = self.model(x, y, z, self.iteration)
        return -elbo, recons, kl, pred

    def get_regularisation(self) -> float:
        # TODO: change configurable schedule!
        if self.iteration > 40000:
            adj = self.model.get_adj()
            reg = self.hp.reg_coeff * torch.norm(adj, p=1)
            reg /= adj.shape[0] ** 2
        else:
            reg = torch.tensor([0.])

        return reg

    def get_acyclicity_violation(self) -> torch.Tensor:
        if self.iteration > 10000:
            adj = self.model.get_adj()[-1].view(self.d, self.d)
            # __import__('ipdb').set_trace()
            h = compute_dag_constraint(adj) / self.acyclic_constraint_normalization
        else:
            h = torch.tensor([0.])

        return h

    def get_ortho_violation(self, w: torch.Tensor) -> float:
        constraint = torch.tensor([0.])
        k = w.size(2)
        for i in range(w.size(0)):
            constraint = constraint + torch.norm(w[i].T @ w[i] - torch.eye(k), p=2)
        h = constraint / self.ortho_normalization
        return h

    def connectivity_reg_complete(self):
        """
        Calculate the connectivity constraint, ie the sum of all the distances
        inside each clusters.
        """
        c = torch.tensor([0.])
        w = self.model.encoder_decoder.get_w()
        d = self.data.distances
        for i in self.d:
            for k in self.d_z:
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
            raise ValueError("You should use a higher value for the ratio of \
                             considered points for the connectivity constraint")

        for d in range(self.d):
            for k in range(self.d_z):
                for i, c1 in enumerate(self.data.coordinates[points]):
                    for j, c2 in enumerate(self.data.coordinates[points]):
                        if i > j:
                            dist = distance.geodesic(c1, c2).km
                            c = c + w[d, i, k] * w[d, j, k] * dist
        return self.hp.reg_coeff_connect * c
