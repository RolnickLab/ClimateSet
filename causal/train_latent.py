import torch
import numpy as np

from dag_optim import compute_dag_constraint
from plot import plot


class TrainingLatent:
    def __init__(self, model, data, hp):
        self.model = model
        self.data = data
        self.hp = hp
        self.latent = hp.latent
        self.debug_gt_z = hp.debug_gt_z
        self.k = hp.k
        self.gt_dag = data.gt_graph
        self.gt_w = data.gt_w
        self.converged = False
        self.thresholded = False
        self.ended = False
        self.mu = hp.mu_init
        self.d = data.x.shape[2]
        self.patience = hp.patience
        self.best_valid_loss = np.inf
        self.batch_size = hp.batch_size
        self.tau = hp.tau
        self.d_x = hp.d_x
        self.instantaneous = hp.instantaneous
        self.qpm_freq = 100
        self.patience_freq = 1000

        # TODO: put as arguments
        self.train_h_list = []
        self.train_loss_list = []
        self.train_elbo_list = []
        self.train_recons_list = []
        self.train_kl_list = []
        self.valid_h_list = []
        self.valid_loss_list = []
        self.valid_elbo_list = []
        self.valid_recons_list = []
        self.valid_kl_list = []
        self.mu_list = []

        # TODO just equal size of G
        if self.instantaneous:
            raise NotImplementedError("Soon")
            self.adj_tt = np.zeros((self.hp.max_iteration, self.tau + 1,
                                    self.d * self.k, self.d * self.k))
        else:
            self.adj_tt = np.zeros((self.hp.max_iteration, self.tau, self.d *
                                    self.k, self.d * self.k))
        self.adj_w_tt = np.zeros((self.hp.max_iteration, self.d, self.d_x, self.k))

        # TODO: just for tests, remove
        # self.model.mask.fix(self.gt_dag)

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
        self.train_recons_list.append(-self.train_recons)
        self.train_kl_list.append(self.train_kl)
        self.valid_h_list.append(self.valid_h)
        self.valid_loss_list.append(self.valid_loss)
        self.valid_recons_list.append(-self.valid_recons)
        self.valid_kl_list.append(self.valid_kl)
        self.mu_list.append(self.mu)

        adj = self.model.get_adj().detach().numpy()
        self.adj_tt[self.iteration] = adj
        w = self.model.encoder_decoder.get_w().detach().numpy()
        self.adj_w_tt[self.iteration] = w

    def print_results(self):
        print("============================================================")
        print(f"Iteration #{self.iteration}")
        print(f"train_h: {self.train_h}")
        # print(f"train_loss: {self.train_loss:.4f}")
        # print(f"train_elbo: {self.train_elbo:.4f}")
        print(f"train_nll: {self.train_nll:.4f}")
        print(f"train_recons: {self.train_recons:.4f}")
        print(f"train_kl: {self.train_kl:.4f}")
        print("-------------------------------")

        print(f"valid_h: {self.valid_h}")
        # print(f"valid_loss: {self.valid_loss:.4f}")
        # print(f"valid_elbo: {self.valid_elbo:.4f}")
        print(f"valid_nll: {self.valid_nll:.4f}")
        print(f"valid_recons: {self.valid_recons:.4f}")
        print(f"valid_kl: {self.valid_kl:.4f}")
        print(f"mu: {self.mu}")
        print(f"patience: {self.patience}")

    def train_with_QPM(self):
        self.iteration = 1

        while self.iteration < self.hp.max_iteration and not self.ended:

            # train and valid step
            self.train_loss, self.train_nll, self.train_recons, self.train_kl, self.train_h = self.train_step()
            self.valid_loss, self.valid_nll, self.valid_recons, self.valid_kl, self.valid_h = self.valid_step()
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
        if self.debug_gt_z:
            x, y, z = self.data.sample_train(self.batch_size)
            nll, recons, kl = self.get_nll(x, y, z)
        else:
            x, y = self.data.sample_train(self.batch_size)
            nll, recons, kl = self.get_nll(x, y)

        # get acyclicity constraint, regularisation
        reg = self.get_regularisation()

        # compute loss
        h = get_ortho_constraint(self.model.encoder_decoder.get_w())
        loss = nll + reg + 0.5 * self.mu * h ** 2  # TODO: have a different mu
        # if self.instantaneous and not self.converged:
        #     h = self.get_acyclicity_violation()
        #     loss = loss + 0.5 * self.mu * h ** 2
        # else:
        #     h = torch.tensor([0])

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        _, _ = self.optimizer.step() if self.hp.optimizer == "rmsprop" else self.optimizer.step(), self.hp.lr

        return loss.item(), nll.item(), recons.item(), kl.item(), h.item()

    def valid_step(self):
        self.model.eval()

        # sample data
        # data = self.test_data
        # idx = np.random.choice(data.shape[0], size=100, replace=False)
        # x = data[idx]
        if self.debug_gt_z:
            x, y, z = self.data.sample_valid(self.data.x_valid.shape[0] - self.data.tau)
            nll, recons, kl = self.get_nll(x, y, z)
        else:
            x, y = self.data.sample_valid(self.data.x_valid.shape[0] - self.data.tau)
            nll, recons, kl = self.get_nll(x, y)

        # get acyclicity constraint, regularisation, elbo
        # h = self.get_acyclicity_violation()
        reg = self.get_regularisation()

        # compute loss
        h = get_ortho_constraint(self.model.encoder_decoder.get_w())
        loss = nll + reg + 0.5 * self.mu * h ** 2  # TODO: have a different mu
        # if self.instantaneous and not self.converged:
        #     h = self.get_acyclicity_violation()
        #     loss = loss + 0.5 * self.mu * h ** 2
        # else:
        #     h = torch.tensor([0])

        return loss.item(), nll.item(), recons.item(), kl.item(), h.item()

    def get_acyclicity_violation(self) -> torch.Tensor:
        adj = self.model.get_adj()[-1].view(self.d, self.d)
        # __import__('ipdb').set_trace()
        h = compute_dag_constraint(adj) / self.constraint_normalization

        return h

    def get_nll(self, x, y, z=None) -> torch.Tensor:
        elbo, recons, kl = self.model(x, y, z, self.iteration)
        return -elbo, recons, kl

    # def get_nll(self, x, y) -> torch.Tensor:
    #     density_param = self.model(x)
    #     mu = density_param[:, :, :, 0].view(-1, 1)
    #     std = density_param[:, :, :, 1].view(-1, 1)

    #     nll = -1/(y.shape[0] * y.shape[1] * y.shape[2]) * self.model.get_likelihood(y, mu, std, self.iteration)
    #     return nll

    def get_regularisation(self) -> float:
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


# if not self.latent:
#     raise ValueError("The orthogonality constraint only makes sense \
#                      when there is latent variables (spatial agg.)")
def get_ortho_constraint(w: torch.Tensor) -> float:
    constraint = torch.tensor([0.])
    k = w.size(2)
    for i in range(w.size(0)):
        constraint = constraint + torch.norm(w[i].T @ w[i] - torch.eye(k), p=2)

    return constraint
