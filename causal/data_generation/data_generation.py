import os
import torch
import json
import torch.nn as nn
import torch.distributions as distr
import numpy as np
from collections import OrderedDict
from typing import Tuple


class DataGeneratorWithLatent:
    """
    Code use to generate synthetic data with latent variables
    """
    # TODO: add instantenous relations
    def __init__(self, hp):
        if hp.instantaneous:
            raise NotImplementedError("Instantaneous connections not implemented yet")
        self.hp = hp
        self.n = hp.n
        self.d = hp.num_features
        self.d_x = hp.d_x
        self.tau = hp.tau

        if self.n > 1:
            self.t = self.tau + 1
        else:
            self.t = hp.num_timesteps

        self.k = hp.k
        self.prob = hp.prob
        self.noise_coeff = hp.noise_coeff

        self.num_layers = hp.num_layers
        self.num_hidden = hp.num_hidden
        self.non_linearity = hp.non_linearity
        self.same_cluster_assign = False

        assert self.d_x > self.k, f"dx={self.d_x} should be larger than k={self.k}"

    def save_data(self, path):
        with open(os.path.join(path, "data_params.json"), "w") as file:
            json.dump(vars(self.hp), file, indent=4)
        np.save(os.path.join(path, 'data_x'), self.X.detach().numpy())
        np.save(os.path.join(path, 'data_z'), self.Z.detach().numpy())
        np.save(os.path.join(path, 'graph'), self.G.detach().numpy())
        np.save(os.path.join(path, 'graph_w'), self.w.detach().numpy())

    def sample_graph(self) -> torch.Tensor:
        """
        Sample a random matrix that will be used as an adjacency matrix
        The diagonal is set to 1.
        Returns:
            A Tensor of tau graphs between the Z (shape: tau x (d x k) x (d x k))
        """
        prob_tensor = torch.ones((self.tau, self.d * self.k, self.d * self.k)) * self.prob
        # set diagonal to 1
        # prob_tensor[:, torch.arange(prob_tensor.size(1)), torch.arange(prob_tensor.size(2))] = 1

        G = torch.bernoulli(prob_tensor)

        return G

    def init_mlp_weight(self, model):
        """Function to initialize MLP's weight from a normal.
        In practice, gives more interesting functions than defaul
        initialization
        Args:
            model: a pytorch model (MLP for now)
        """
        if isinstance(model, torch.nn.modules.Linear):
            torch.nn.init.normal_(model.weight.data, mean=0., std=1)

    def sample_mlp(self, n_timesteps):
        """Sample a MLP that outputs the parameters for the distributions of Z
        Args:
            n_timesteps: Number of previous timesteps considered
        """
        mlps = nn.ModuleList()
        num_first_layer = n_timesteps * self.d * self.k
        num_last_layer = 2

        if self.non_linearity == "relu":
            nonlin = nn.ReLU()
        elif self.non_linearity == "leaky_relu":
            nonlin = nn.LeakyReLU()
        else:
            raise NotImplementedError("Nonlinearity is not implemented yet")

        for i_d in range(self.d):
            for k in range(self.k):
                dict_layers = OrderedDict()
                for i in range(self.num_layers + 1):
                    num_input = self.num_hidden
                    num_output = self.num_hidden

                    if i == 0:
                        num_input = num_first_layer
                    if i == self.num_layers:
                        num_output = num_last_layer
                    dict_layers[f"lin{i}"] = nn.Linear(num_input, num_output)

                    if i != self.num_layers:
                        dict_layers[f"nonlin{i}"] = nonlin

                mlp = nn.Sequential(dict_layers)
                mlp.apply(self.init_mlp_weight)
                mlps.append(mlp)

        return mlps

    def sample_lstm(self):
        pass

    def sample_w(self) -> torch.Tensor:
        """Sample matrices that are non-negative and orthogonal.
        They are the links between Z and X.
        Returns:
            A tensor w (shape: d, d_x, k)
        """
        mask = torch.zeros((self.d, self.d_x, self.k))
        for i in range(self.d):
            # assign d_xs uniformly to a cluster k
            cluster_assign = np.random.choice(self.k, size=self.d_x - self.k)
            cluster_assign = np.append(cluster_assign, np.arange(self.k))
            np.random.shuffle(cluster_assign)
            mask[i, np.arange(self.d_x), cluster_assign] = 1

        w = torch.empty((self.d, self.d_x, self.k)).uniform_(0.5, 2)
        w = w * mask

        # normalize to make w orthonormal
        for i in range(self.d):
            w[i] = w[i] / torch.norm(w[i], dim=0)
            assert torch.all(torch.isclose(torch.matmul(w[i].T, w[i]), torch.eye(w.size(-1)), rtol=0.01))
        return w

    def generate(self):
        """Main method to generate data
        Returns:
            X, Z, respectively the observable data and the latent
        """
        # initialize Z for the first timesteps
        self.Z = torch.zeros((self.n, self.t, self.d, self.k))
        self.X = torch.zeros((self.n, self.t, self.d, self.d_x))

        # sample graphs and NNs
        self.G = self.sample_graph()
        self.f = [None]

        # sample observational model
        self.w = self.sample_w()

        for n_timesteps in range(1, self.tau + 1):
            self.f.append(self.sample_mlp(n_timesteps))

        for i_n in range(self.n):
            # sample the latent Z
            for t in range(self.t):
                if t == 0:
                    self.Z[i_n, t].normal_(0, 1)  # test with mu=2
                else:
                    if t < self.tau:
                        nn_idx = t
                        g = self.G[:t]
                        z = self.Z[i_n, :t]
                    else:
                        nn_idx = -1
                        g = self.G
                        z = self.Z[i_n, t - self.tau:t]

                    # nn_input = (g * z).view(-1)
                    for i_d in range(self.d):
                        for k in range(self.k):
                            nn_input = z.view(z.shape[0], -1) * g[:, k + i_d * self.k]
                            params = self.f[nn_idx][k + i_d * self.k](nn_input)

                            # params[:, 1] = 0.5 * torch.exp(params[:, 1]) * 0.01  # TODO: change back, only a test, smaller variance
                            std = torch.ones_like(params[:, 1]) * 0.0001
                            dist = distr.normal.Normal(params[:, 0], std)
                            self.Z[i_n, t, i_d, k] = dist.rsample()

            # sample the data X
            for t in range(self.t):
                # TODO: probably error here:
                for i_d in range(self.d):
                    mu = torch.matmul(self.Z[i_n, t, i_d], self.w[i_d].T)
                # mean = torch.einsum("xdk,dk->dx", self.w, self.Z[i_n, t])
                # TODO: could sample sigma
                    dist = distr.normal.Normal(mu, 0.0001)
                    self.X[i_n, t, i_d] = dist.rsample()

        return self.X, self.Z


class DataGeneratorWithoutLatent:
    """
    Code use to generate synthetic data without latent variables
    """
    def __init__(self, hp):
        self.hp = hp
        self.n = hp.n
        self.d = hp.num_features
        self.d_x = hp.d_x
        self.world_dim = hp.world_dim
        self.tau = hp.tau
        self.tau_neigh = hp.neighborhood
        self.prob = hp.prob
        self.eta = hp.eta
        self.noise_coeff = hp.noise_coeff
        self.instantaneous = hp.instantaneous
        self.func_type = hp.func_type
        self.noise_type = hp.noise_type

        if self.n > 1:
            self.t = self.tau + 1
        else:
            self.t = hp.num_timesteps

        self.num_layers = hp.num_layers
        self.num_hidden = hp.num_hidden
        self.non_linearity = hp.non_linearity

        assert hp.world_dim <= 2 and hp.world_dim >= 1, "world_dim not supported"
        self.num_neigh = (self.tau_neigh * 2 + 1) ** self.world_dim

    def save_data(self, path: str):
        """ Save the data, the parameters, and the graph as .npy files
        Args:
            path: path where to save the files
        """
        with open(os.path.join(path, "data_params.json"), "w") as file:
            json.dump(vars(self.hp), file, indent=4)
        np.save(os.path.join(path, 'data_x'), self.X.detach().numpy())
        np.save(os.path.join(path, 'graph'), self.G.detach().numpy())

    def sample_graph(self, diagonal: bool = False) -> Tuple[torch.Tensor, list]:
        """
        Sample a random matrix that will be used as an adjacency matrix
        The diagonal is set to 1.
        Returns:
            A Tensor of tau graphs, size: (tau, d, num_neighbor x d)
        """
        # TODO: allow data with any number of dimension (1D, 2D, ...)
        prob_tensor = torch.ones((self.tau, self.d, self.num_neigh * self.d)) * self.prob

        if diagonal:
            # set diagonal to 1
            prob_tensor[:, torch.arange(prob_tensor.size(1)), torch.arange(prob_tensor.size(2))] = 1

        G = torch.bernoulli(prob_tensor)

        if self.instantaneous:
            dag, causal_order = self.sample_dag()
            G = torch.cat((G, dag), dim=0)
        else:
            causal_order = torch.arange(self.d)

        return G, causal_order

    def sample_dag(self) -> Tuple[torch.Tensor, list]:
        """
        Sample a random DAG that will be used as an adjacency matrix
        for instantaneous connections
        Returns:
            A Tensor of tau graphs, size: (tau, d, num_neighbor x d)
            and a list containing the causal order of the variables
        """
        prob_tensor = torch.ones((1, self.d, self.num_neigh * self.d)) * self.prob
        # set all elements on and above the diagonal as 0
        prob_tensor = torch.tril(prob_tensor, diagonal=-1)

        G = torch.bernoulli(prob_tensor)

        # G = torch.tensor([[[0., 0., 0., 0., 0., 0.],
        #                    [0., 0., 0., 0., 0., 0.],
        #                    [0., 0., 0., 0., 0., 0.],
        #                    [0., 0., 0., 0., 0., 0.],
        #                    [0., 0., 1., 1., 0., 0.],
        #                    [0., 1., 0., 1., 0., 0.]]])
        # causal_order = torch.tensor([0, 5, 2, 1, 4, 3])
        # print(G)

        # permutation
        causal_order = torch.randperm(self.d)

        # p_G = G[0, :, :]
        # P = torch.zeros(self.d, self.d)
        # P[torch.arange(self.d), causal_order] = 1
        # p_G = torch.matmul(P, torch.matmul(p_G, P.T))

        G = G[:, causal_order]
        causal_order_dag = torch.arange(self.num_neigh * self.d)
        causal_order_dag[self.num_neigh//2 * self.d: (self.num_neigh//2 + 1) * self.d] = causal_order
        G = G[:, :, causal_order_dag]
        assert is_acyclic(G[0, :, self.num_neigh//2 * self.d: (self.num_neigh//2 + 1) * self.d])

        return G, causal_order

    def sample_linear_weights(self, lower: int = 0.3, upper: float = 0.5, eta:
                              float = 1) -> torch.Tensor:
        """Sample the coefficient of the linear relations
        Args:
            lower: lower bound of uniform distr to sample from
            upper: upper bound of uniform distr
            eta: weight decay parameter, reduce the influences of variables
            that are farther back in time, should be >= 1
        Returns:
            tensor of coefficients for linear functions
        """
        # if self.G.shape[1] == 2:
        #     weights = torch.empty_like(self.G).uniform_(lower, upper)
        #     # known periodic linear dynamical system
        #     weights[0] = torch.tensor([[-3.06, 1.68],
        #                                [-4.20, 1.97]])
        #     # weights[0] = torch.tensor([[-1.72, -3.91],
        #     #                            [1.56, 2.97]])
        #     self.G[0] = torch.tensor([[1, 1],
        #                              [1, 1]])
        # else:
        sign = torch.ones_like(self.G) * 0.5
        sign = torch.bernoulli(sign) * 2 - 1
        weights = torch.empty_like(self.G).uniform_(lower, upper)
        weights = sign * weights * self.G

        # apply weight decay (graphs further in time have smaller weights)
        weight_decay = 1 / torch.pow(eta, torch.arange(self.G.size(0)))
        weights = weights * weight_decay.view(-1, 1, 1)

        return weights

    def generate(self) -> torch.Tensor:
        """Main method to generate data
        Returns:
            X, the data, size: (n, t, d, d_x)
        """
        if self.func_type == "linear":
            # X = self.generate_linear()
            X = self.generate_data(self.func_type, self.noise_type)
        else:
            X = self.generate_data(self.func_type, self.noise_type)

        return X

    class LinearFunction:
        def __init__(self, w):
            self.w = w

        def __call__(self, x):
            return torch.matmul(self.w.reshape(-1), x)

    def sample_linears(self) -> list:
        """Sample linear weights in a format usable with generate_data()
        Returns:
            A list of tensors that are linear coefficients
        """
        weights = self.sample_linear_weights()
        ws = []

        for i_d in range(self.d):
            func = self.LinearFunction(weights[:, i_d])
            ws.append(func)

        return ws

    def init_mlp_weight(self, model):
        """Function to initialize MLP's weight from a normal.
        In practice, gives more interesting functions than defaul
        initialization
        Args:
            model: a pytorch model (MLP for now)
        """
        if isinstance(model, torch.nn.modules.Linear):
            torch.nn.init.normal_(model.weight.data, mean=0., std=1)

    def sample_mlps(self) -> list:
        """Sample a MLP that outputs the parameters for the distributions of Z
        Returns:
            A list of MLPs
        """
        mlps = nn.ModuleList()
        dict_layers = OrderedDict()
        if self.instantaneous:
            num_first_layer = (self.tau + 1) * (self.d * self.num_neigh)
        else:
            num_first_layer = self.tau * (self.d * self.num_neigh)
        num_last_layer = 1

        if self.non_linearity == "relu":
            nonlin = nn.ReLU()
        elif self.non_linearity == "leaky_relu":
            nonlin = nn.LeakyReLU()
        else:
            raise NotImplementedError("Nonlinearity is not implemented yet")

        for i_d in range(self.d):
            for i in range(self.num_layers + 1):
                num_input = self.num_hidden
                num_output = self.num_hidden

                if i == 0:
                    num_input = num_first_layer
                if i == self.num_layers:
                    num_output = num_last_layer
                dict_layers[f"lin{i}"] = nn.Linear(num_input, num_output)

                if i != self.num_layers:
                    dict_layers[f"nonlin{i}"] = nonlin

            mlp = nn.Sequential(dict_layers)
            mlp.apply(self.init_mlp_weight)
            mlps.append(mlp)

        return mlps

    def generate_data(self, fct_type: str, noise_type: str, additive: bool = True) -> torch.Tensor:
        """Method to generate data with any function and
        arbitrary noise (either additive or not)
        Args:
            fct_type: type of function: {'nn'}
            noise_type: type of noise: {'gaussian', 'laplacian', 'uniform'}
            additive: if True, the noise is additive
        Returns:
            X, the data, size: (n, t, d, d_x)
        """
        # sample graphs and NN weights
        self.X = torch.zeros((self.n, self.t, self.d, self.d_x))
        self.G, self.causal_order = self.sample_graph()

        if fct_type == "mlp":
            self.fct = self.sample_mlps()
        elif fct_type == "linear":
            self.fct = self.sample_linears()
        else:
            raise NotImplementedError("This type of function is not implemented yet.")

        for i_n in range(self.n):  # example
            # sample noise and initialize X
            if noise_type == "gaussian":
                noise_distr = distr.normal.Normal(0, 1)
            elif noise_type == "laplacian":
                noise_distr = distr.laplace.Laplace(0, 1)
            elif noise_type == "uniform":
                noise_distr = distr.uniform.Uniform(0, 1)
            else:
                raise NotImplementedError("This type of noise is not implemented yet.")
            noise = noise_distr.sample(self.X.size())
            self.X[i_n, :self.tau] = noise[i_n, :self.tau]

            for t in range(self.tau, self.t):  # timstep
                if self.instantaneous:
                    t1 = 1
                else:
                    t1 = 0

                for i in range(self.d_x):  # grid cell
                    lower_x = max(0, i - self.tau_neigh)
                    upper_x = min(self.X.size(-1) - 1, i + self.tau_neigh) + 1

                    # if self.instantaneous:
                    # TODO: sample in causal order (also when applying # permutation)
                    for i_d in self.causal_order:  # feature
                        # __import__('ipdb').set_trace()
                        if self.d_x == 1:
                            # print(self.G.size())
                            # print(self.X.size())
                            # print(self.X[i_n, t - self.tau:t + t1, :, :self.d].size())
                            # __import__('ipdb').set_trace()
                            x = self.X[i_n, t - self.tau:t + t1, :, :self.d]
                            # x = x.reshape(self.G.size(0), -1)
                        else:
                            x = self.X[i_n, t - self.tau:t + t1, :, lower_x:upper_x]
                            # x = x.reshape(self.G.size(0), -1)

                        if additive:
                            # if i_d == 0 and i_n == 0:
                            #     __import__('ipdb').set_trace()
                            #     print(x[:, :, i_d].view(-1))
                            #     print(self.fct[i_d](x[:, :, i_d].view(-1)))

                            # print(self.G[:, i_d].size())
                            # print(x.size())
                            # print(x[:, i_d].size())
                            # print(x[:, i_d])
                            # print(self.fct[i_d].w)
                            # print(self.G[:, i_d])

                            masked_x = x * self.G[:, i_d].unsqueeze(-1)
                            self.X[i_n, t, i_d, i] = self.fct[i_d](masked_x.view(-1))
                            self.X[i_n, t, i_d, i] += self.noise_coeff * noise[i_n, t, i_d, i]
                        else:
                            # TODO: check for x[:, :, i_d]
                            x_ = torch.cat((x, noise[i_n, t, i_d, i]), dim=2)
                            self.X[i_n, t, i_d, i] = self.fct[i_d](x_.view(-1))

        return self.X

    def generate_linear(self) -> torch.Tensor:
        """Method to generate data with linear function
        and Gaussian noise
        Returns:
            X, the data, size: (n, t, d, d_x)
        """

        # sample graphs and weights
        self.X = torch.zeros((self.n, self.t, self.d, self.d_x))
        self.G, self.causal_order = self.sample_graph()
        self.weights = self.sample_linear_weights(eta=self.eta)

        for i_n in range(self.n):
            # initialize X and sample noise
            noise = torch.normal(0, 1, size=self.X.size())
            self.X[i_n, :self.tau] = noise[i_n, :self.tau]

            for t in range(self.tau, self.t):
                # if self.d_x == 1:
                #     # TODO: only test
                #     x = self.X[t-1, :, 0]
                #     w = self.weights
                #     self.X[t, :, 0] = w[0].T @ x  # torch.einsum("tij,tij->i", w, x)
                # else:
                if self.instantaneous:
                    t1 = 1
                else:
                    t1 = 0

                for i in range(self.d_x):
                    # TODO: should add wrap around
                    lower_x = max(0, i - self.tau_neigh)
                    upper_x = min(self.X.size(-1) - 1, i + self.tau_neigh) + 1
                    lower_w = max(0, i - self.tau_neigh) - (i - self.tau_neigh)
                    upper_w = min(self.X.size(-1) - 1, i + self.tau_neigh) - i + self.tau_neigh + 1

                    # w.size: (tau, d, d * (tau_neigh * 2 + 1))
                    # x.size: (tau, d * (tau_neigh * 2 + 1))
                    # print(w.size())
                    # print(x.size())

                    if self.instantaneous:
                        # TODO: sample in causal order (also when applying # permutation)
                        for i_d in range(self.d):
                            if self.d_x == 1:
                                w = self.weights[:, i_d, :self.d]
                                x = self.X[i_n, t - self.tau:t + t1, :, :self.d].reshape(self.G.size(0), -1)
                            else:
                                w = self.weights[:, i_d, lower_w * self.d: upper_w * self.d]
                                x = self.X[i_n, t - self.tau:t + t1, :, lower_x:upper_x].reshape(self.G.size(0), -1)
                            self.X[i_n, t, i_d, i] = torch.einsum("tj,tj->", w, x) + \
                                self.noise_coeff * noise[i_n, t, i_d, i]
                    else:
                        if self.d_x == 1:
                            w = self.weights[:, :, :self.d]
                            x = self.X[i_n, t - self.tau:t + t1, :, :self.d].reshape(self.G.size(0), -1)
                        else:
                            w = self.weights[:, :, lower_w * self.d: upper_w * self.d]
                            x = self.X[i_n, t - self.tau:t + t1, :, lower_x:upper_x].reshape(self.G.size(0), -1)
                        self.X[i_n, t, :, i] = torch.einsum("tij,tj->i", w, x) + self.noise_coeff * noise[i_n, t, :, i]

        return self.X


def is_acyclic(adjacency: np.ndarray) -> bool:
    """
    Return true if adjacency is acyclic
    Args:
        adjacency: adjacency matrix
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True
