import torch
import numpy as np

from dag_optim import compute_dag_constraint
from plot import plot


class Training:
    def __init__(self, model, data, hp):
        self.model = model
        self.data = data
        self.gt_dag = data.gt_graph
        self.hp = hp
        self.converged = False
        self.thresholded = False
        self.ended = False
        self.mu = hp.mu_init
        self.d = data.x.shape[2]
        self.patience = hp.patience
        self.best_valid_loss = np.inf
        self.batch_size = hp.batch_size
        self.tau = hp.tau
        self.tau_neigh = hp.tau_neigh
        self.instantaneous = hp.instantaneous
        self.qpm_freq = 1000
        self.patience_freq = 1000

        # TODO: put as arguments
        self.train_h_list = []
        self.train_loss_list = []
        self.train_elbo_list = []
        self.valid_h_list = []
        self.valid_loss_list = []
        self.valid_elbo_list = []
        self.mu_list = []

        # TODO just equal size of G
        if self.instantaneous:
            self.adj_tt = np.zeros((self.hp.max_iteration, self.tau + 1, self.d, self.d, self.model.tau_neigh * 2 + 1))
        else:
            self.adj_tt = np.zeros((self.hp.max_iteration, self.tau, self.d, self.d, self.model.tau_neigh * 2 + 1))
        # TODO: just for tests, remove
        # self.model.mask.fix(gt_dag)

        # optimizer
        if hp.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr)
        elif hp.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=hp.lr)
        else:
            raise NotImplementedError("optimizer {} is not implemented".format(hp.optimizer))

        # compute constraint normalization
        with torch.no_grad():
            full_adjacency = torch.ones((model.d, model.d)) - torch.eye(model.d)
            self.constraint_normalization = compute_dag_constraint(full_adjacency).item()

    def log_losses(self):
        self.train_h_list.append(self.train_h)
        self.train_loss_list.append(self.train_loss)
        self.valid_h_list.append(self.valid_h)
        self.valid_loss_list.append(self.valid_loss)
        self.mu_list.append(self.mu)

        adj = self.model.get_adj().detach().numpy()
        self.adj_tt[self.iteration] = adj

    def print_results(self):
        print("============================================================")
        print(f"Iteration #{self.iteration}")
        # print(f"train_h: {self.train_h:.4f}")
        # print(f"train_loss: {self.train_loss:.4f}")
        # print(f"train_elbo: {self.train_elbo:.4f}")
        print(f"train_nll: {self.train_nll:.4f}")
        # print(f"train_kl: {self.train_kl:.4f}")
        print("-------------------------------")

        print(f"valid_h: {self.valid_h:.4f}")
        # print(f"valid_loss: {self.valid_loss:.4f}")
        # print(f"valid_elbo: {self.valid_elbo:.4f}")
        print(f"valid_nll: {self.valid_nll:.4f}")
        # print(f"valid_kl: {self.valid_kl:.4f}")
        print(f"mu: {self.mu}")
        print(f"patience: {self.patience}")

    def train_with_QPM(self):
        self.iteration = 1

        while self.iteration < self.hp.max_iteration and not self.ended:

            # train and valid step
            self.train_loss, self.train_nll, self.train_h = self.train_step()
            self.valid_loss, self.valid_nll, self.valid_h = self.valid_step()
            self.log_losses()
            if self.iteration % self.hp.print_freq == 0:
                self.print_results()

            # train in 3 phases: first with QPM, then until the likelihood
            # remain stable, then continue after thresholding the adjacency
            # matrix
            if not self.converged:
                # train with penalty method
                if self.iteration % self.qpm_freq == 0:
                    self.QPM(self.iteration, self.valid_loss, self.valid_h)
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

            # Utilities: log, plot and save results
            if self.iteration % self.hp.plot_freq == 0:
                plot(self)
            # self.save()

            self.iteration += 1

            # TODO: remove
            if self.iteration % 30000 == 0:
                pass

        # final plotting and printing
        plot(self)
        self.print_results()

    def QPM(self, iteration: int, valid_loss: float, h: float):
        # check if QPM has converged
        if self.iteration > self.hp.min_iter_convergence and h <= self.hp.h_threshold:
            self.converged = True
        else:
            # check if subproblem has converged
            if self.valid_loss_list[-1] > self.valid_loss_list[-2]:
                if self.valid_h_list[-1] > self.valid_h_list[-2] * self.hp.omega_mu:
                    self.mu *= self.hp.mu_mult_factor
                    # print(self.valid_loss_list[-1])
                    # print(self.valid_loss_list[-2])
                    # print(self.valid_h_list[-1])
                    # print(self.valid_h_list[-2])
                    # __import__('ipdb').set_trace()

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
        x, y = self.data.sample_train(self.batch_size)
        density_param = self.model(x)

        # get acyclicity constraint, regularisation
        reg = self.get_regularisation()
        # TODO: change density_param
        nll = self.get_nll(y, density_param)

        # compute loss
        if self.instantaneous and not self.converged:
            h = self.get_acyclicity_violation()
            loss = nll + reg + 0.5 * self.mu * h ** 2
        else:
            h = torch.tensor([0])
            loss = nll + reg

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        _, _ = self.optimizer.step() if self.hp.optimizer == "rmsprop" else self.optimizer.step(), self.hp.lr

        return loss.item(), nll.item(), h.item()

    def valid_step(self):
        self.model.eval()

        # sample data
        # data = self.test_data
        # idx = np.random.choice(data.shape[0], size=100, replace=False)
        # x = data[idx]
        x, y = self.data.sample_valid(self.data.x_valid.shape[0] - self.data.tau)
        density_param = self.model(x)

        # get acyclicity constraint, regularisation, elbo
        # h = self.get_acyclicity_violation()
        reg = self.get_regularisation()
        nll = self.get_nll(y, density_param)

        # compute loss
        if self.instantaneous and not self.converged:
            h = self.get_acyclicity_violation()
            loss = nll + reg + 0.5 * self.mu * h ** 2
        else:
            h = torch.tensor([0])
            loss = nll + reg

        return loss.item(), nll.item(), h.item()

    def get_acyclicity_violation(self) -> torch.Tensor:
        adj = self.model.get_adj()[-1].view(self.d, self.d)
        # __import__('ipdb').set_trace()
        h = compute_dag_constraint(adj) / self.constraint_normalization

        return h

    def get_nll(self, y, density_param) -> torch.Tensor:
        mu = density_param[:, :, :, 0].view(-1, 1)
        std = density_param[:, :, :, 1].view(-1, 1)
        nll = -1/(y.shape[0] * y.shape[1] * y.shape[2]) * self.model.get_likelihood(y, mu, std, self.iteration)

        return nll

    def get_regularisation(self):
        adj = self.model.get_adj()
        reg = self.hp.reg_coeff * torch.norm(adj, p=1)
        reg /= adj.shape[0] ** 2

        return reg

    def threshold(self):
        with torch.no_grad():
            thresholded_adj = (self.model.get_adj() > 0.5).type(torch.Tensor)
            self.model.mask.fix(thresholded_adj)
        self.thresholded = True
        print("Thresholding ================")

    def save_results(self):
        # TODO
        pass
