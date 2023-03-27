import os
import torch
import json
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import torch.nn as nn
import torch.distributions as distr
import numpy as np
from collections import OrderedDict
from typing import Tuple
from scipy.special import expit


class MixingFunctions:
    def __init__(self, mask, d_x, d_z):
        self.mask = mask
        self.d_x = d_x
        self.d_z = d_z
        self._sample_all_functions()

    class LeakySoftplus:
        def __init__(self):
            self.sign = np.random.randint(0, 2) * 2 - 1
            self.a = np.random.rand() * 0.15 + 0.05

            # self.bias = np.random.rand() * 10 - 5
            # self.b = np.random.rand() * 10 - 5
            # self.slope = np.random.rand() * 5 + 1
            self.bias = np.random.rand() * 0.1 - 0.05
            self.b = np.random.rand() * 0.1 - 0.05
            self.slope = np.random.rand() * 1 + 1
            self.first = True

        def __call__(self, x):
            if self.first:
                self.adj = x.max()
                self.b = x.mean()
                self.first = False
            x = x / self.adj * 5
            out = self.a * (x - self.b)
            out += (1 - self.a) * np.log(1 + np.exp(self.slope * (x - self.b)))
            return self.sign * out + self.bias

    def _sample_all_functions(self):
        self.fct_dict = {}
        for i in range(self.d_x):
            for j in range(self.d_z):
                if self.mask[i, j]:
                    self.fct_dict[(i, j)] = self.LeakySoftplus()

    def __call__(self, z):
        x = torch.zeros((z.shape[0], self.d_x))
        for j in range(self.d_z):
            first = True

            for i in range(self.d_x):
                if self.mask[i, j]:
                    z_ = z[:, i, j] / z[:, i, j].max() * 2

                    if first:
                        x[:, i] = z_
                        first = False
                    # x[:, i] = (torch.sigmoid(20 * z_))
                    # x[:, i] = (z_ + 10 * z_ ** 3)
                    # x[:, i] = self.fct_dict[(i, j)](z[:, i, j])

                    # cubic - good range
                    # i1 = np.random.rand() * 6 - 3
                    # i2 = np.random.rand() * 4 - 2
                    # i3 = np.random.rand() * 2 - 1
                    sign = np.random.rand()
                    if sign < 0.5:
                        sign = -1
                    else:
                        sign = 1

                    if sign == 1:
                        # x[:, i] = sign * ((z_ - i1) ** 3 + 0.8 * (z_ - i2) ** 3 + 0.6 * (z_ - i3) ** 3)
                        i1 = np.random.rand() * 6 + 1

                        x[:, i] = sign * np.sin(i1 * z_)
                    else:
                        x[:, i] = z_
        return x


def sample_logistic_coeff(graph: np.ndarray, tau: int, d: int, d_z: int,
                          low: float, high: float) -> np.ndarray:
    """
    Sample coefficients in the range [low, high]
    Returns:
        coeff: (tau, d*d_z, d*d_z) an array of coefficients
    """
    # sign = np.random.binomial(1, 0.5, size=(tau, d * d_z, d * d_z)) * 2 -1
    coeff = np.random.rand(tau, d * d_z, d * d_z) * (high - low) + low
    coeff = coeff * graph

    return coeff


def sample_stationary_coeff(graph: np.ndarray, tau: int, d: int, d_z: int, eps:float = 1e-4) -> np.ndarray:
    """
    Sample linear coefficients such that the spectrum associated
    to this AR is equal to 1. This is a necessary condition in order
    to have a stationary process.
    Returns:
        coeff: (tau, d*d_z, d*d_z) an array of coefficients that should lead
        to a stationary process
    """
    # sample coefficients from Unif([-1, -0.2]U[0.2, 1])
    coeff = np.random.rand(tau, d * d_z, d * d_z) * 0.8 + 0.2
    sign = np.random.binomial(1, 0.5, size=(tau, d * d_z, d * d_z)) * 2 -1
    coeff = coeff * sign * graph

    # set the spectrum of the coeff to 1
    radius = get_spectrum(coeff)
    print(radius)
    for t in range(tau):
        coeff[t] = coeff[t] / (radius ** (tau - t) + eps)
    radius = get_spectrum(coeff)
    assert radius < 1 + 1e-4
    print(radius)

    min_val = np.min(np.abs(coeff[coeff != 0.]))
    print(f"Min value: {min_val}")
    if min_val < 0.01:
        print("Warning, the magnitude of some coefficients are lower than 0.01")

    return coeff

def get_spectrum(coeff) -> float:
    """
    Return the spectrum of the linear coefficients corresponding to
    an autoregressive process.
    """
    tau = coeff.shape[0]
    d = coeff.shape[1]

    # create a matrix A of dimension d x tau*d 
    A = sparse.hstack([sparse.lil_matrix(coeff[tau - t - 1]) for t in range(tau)])
    # get a square matrix A of dimension tau*d x tau*d
    A = sparse.vstack([A, sparse.eye((tau - 1) * d, tau * d)])

    A = A.todense()
    eigen_val, _ = scipy.linalg.eig(A)
    radius = np.max(np.abs(eigen_val))

    return radius


class NonadditiveNonlinear:
    # uses polynomials of degree 2
    def __init__(self, coeff, d, d_z):
        self.coeff = coeff.T
        self.d = d
        self.d_z = d_z
        self.poly_coeff1 = self.sample_poly_coeff((d_z**2 + d_z))
        self.poly_coeff2 = self.sample_poly_coeff(d_z**2)

    # Create multivariate polynomial of degree 2
    def sample_poly_coeff(self, size):
        # sample sparse
        poly_coeff = np.random.uniform(low=0, high=1, size=size)
        # * np.random.randint(0, 2, size)
        return poly_coeff

    def polynomial(self, coeff, x):
        d_z = self.d_z
        output = x @ coeff[:d_z]
        output += x**2 @ coeff[d_z:2*d_z]
        idx = 0
        for i in range(d_z):
            for j in range(i + 1, d_z):
                output += coeff[2*d_z + idx] * x[:, i] * x[:, j]
                idx += 1

        return output

    def polynomial2(self, coeff, x):
        d_z = self.d_z
        output = x ** 2 @ coeff[:d_z]
        idx = 0
        for i in range(d_z):
            for j in range(i + 1, d_z):
                output += coeff[d_z + idx] * x[:, i] ** 2 * x[:, j] ** 2
                idx += 1

        return output

    def __call__(self, x):
        """
        x: (n, d)
        """
        output = 0
        p1 = self.polynomial(self.poly_coeff1, x)
        p2 = self.polynomial2(self.poly_coeff2, x)
        lin = torch.sum(x @ self.coeff, dim=1)
        output = lin + p1 * np.exp(-p2)

        return output

class NonAdditiveStationaryMechanisms:
    def __init__(self, graph, tau, d, d_z, causal_order, instantaneous, radius_correct, noise_z_std=0.1):
        self.tau = tau
        self.d = d
        self.d_z = d_z
        self.noise_z_std = noise_z_std
        self.G = graph
        self.causal_order = causal_order
        self.instantaneous = instantaneous
        self.mechs = []
        self.radius_correct = radius_correct
        self.coeff = sample_stationary_coeff(self.G, tau + 1, d, d_z, radius_correct)
        self.sample_mechs()

    def sample_mechs(self):
        for i in range(self.d_z):
            self.mechs.append(NonadditiveNonlinear(self.coeff[:-1, i], self.d, self.d_z))

    def apply(self, g, z, t):
        """Apply the mechanisms to z and add noise"""
        if t >= self.tau + 1:
            t = self.tau

        z = z.reshape(z.shape[0], 1, -1)
        noise = np.random.normal(scale=self.noise_z_std, size=(self.d, self.d_z))

        if self.instantaneous:
            raise NotImplementedError("Instantaneous non-additive mech are not implemented yet")
        else:
            output = np.zeros((self.d_z))
            for i in range(self.d_z):
                input_ = (g[:-1, i:i+1] * z[:-1]).view(1, -1)
                output[i] = self.mechs[i](input_)

        return torch.from_numpy(output), noise

class LogisticMechanisms:
    """
    Additive nonlinear mechanisms that lead a stationary process.
    The trick is to use nonlinear functions that are almost linear when x is
    large, and have coefficients with a spectrum <= 1.
    """
    def __init__(self, graph, tau, d, d_z, causal_order, instantaneous, radius_correct, func_type, noise_z_std=0.1):
        self.tau = tau
        self.d = d
        self.d_z = d_z
        self.G = graph
        self.causal_order = causal_order
        self.instantaneous = instantaneous
        self.radius_correct = radius_correct
        self.r = 4
        self.noise_coeff = 0.1

        self.fct = [lambda x: x]
        self.n_mech = len(self.fct)
        self.prob_mech = [1/self.n_mech] * self.n_mech

        self.sample_mech()
        self.apply_vectorized = np.vectorize(self.apply_f)

    def sample_mech(self):
        self.mech = np.random.choice(self.n_mech,
                                     size=(self.tau + 1, self.d * self.d_z, self.d * self.d_z),
                                     p=self.prob_mech)
        # TODO: change sampling process (c should be fixed or in narrow range)
        # self.coeff = sample_stationary_coeff(self.G, self.tau + 1, self.d, self.d_z, self.radius_correct)
        self.coeff = sample_logistic_coeff(self.G, self.tau + 1, self.d, self.d_z, low=2, high=3)

    def apply_f(self, i, x):
        return self.fct[i](x)

    def apply(self, g, z, t):
        """Apply the mechanisms to z and add noise"""
        if t >= self.tau + 1:
            t = self.tau

        z = z.reshape(z.shape[0], 1, -1)
        z = z.repeat(1, z.shape[-1], 1)
        noise = np.random.uniform(size=(self.d, self.d_z))

        if self.instantaneous:
            raise NotImplementedError("not clear yet")
            # output = self.apply_vectorized(self.mech[-t-1:-1], z[:-1])
            # output = output * self.coeff[-t-1:-1]
            # output = np.sum(output, axis=(0, 2))
            # output = output.reshape(self.d, self.d_z)
            # output += noise

            # for i in self.causal_order:
            #     output[:, i] += self.coeff[-1, i] @ self.apply_vectorized(self.mech[-1, i], output[0])
        else:
            output = self.apply_vectorized(self.mech[-t-1:], z)
            output = output * self.coeff[-t-1:]
            output = np.sum(output, axis=(0, 2))
            output = output.reshape(self.d, self.d_z)
            output += self.noise_coeff * (noise - 0.5)

            # r - rX^t-1 - output
            # z[-2, 0] corresponds to z^{t-1}
            output = self.r - self.r * z[-2, 0] - output
            # (X (output)) mod 1
            output = (z[-2, 0] * output) % 1

        return output, noise

class StationaryMechanisms:
    """
    Additive nonlinear mechanisms that lead a stationary process.
    The trick is to use nonlinear functions that are almost linear when x is
    large, and have coefficients with a spectrum <= 1.
    """
    def __init__(self, graph, tau, d, d_z, causal_order, instantaneous, radius_correct, func_type, noise_z_std=0.1):
        self.tau = tau
        self.d = d
        self.d_z = d_z
        self.noise_z_std = noise_z_std
        self.G = graph
        self.causal_order = causal_order
        self.instantaneous = instantaneous
        self.radius_correct = radius_correct

        if func_type == "linear":
            self.fct = [lambda x: x]
        elif func_type == "add_nonlinear":
            # self.fct = [lambda x: x * (1 + 4 * np.exp(-x ** 2 / 2)),
            #             lambda x: x * (1 + 4 * x ** 3 * np.exp(-x ** 2 / 2))]
            self.fct = [lambda x: x * (1 + 4 * np.exp(-x ** 2 / 2)),
                        lambda x: x * (1 + 10 * (np.exp(-x ** 2 / 2 * (np.abs(np.sin(x)) + 1.1)))),
                        lambda x: x * (1 + 5 * (np.exp(-x ** 2 / 4 * (np.abs(np.cos(x)) + 1.1)))),
                        lambda x: x * (1 + 5 * (np.exp(-x ** 2 / 2 * (np.abs(np.cos(x)) + 1.1)) * (np.abs(np.sin(x+ 0.2)) + 1.1))),
                        lambda x: x * (1 + 10 * (expit(-x ** 2 / 5 * (np.abs(np.cos(x)) + 1.1)))),
                        lambda x: x * (1 + 5 * (expit(-x ** 2 / 5)) + 5 * (expit(-(x + 2) ** 2 / 5))),
                        lambda x: x * (1 + 5 * (expit(-x ** 2 / 5)) - 5 * (expit(-(x + 1.5) ** 2 / 5))),
                        lambda x: x * (1 + 2 * (expit(-x ** 2 / 50)) - 15 * (expit(-(x + 10) ** 2 / 50))),
                        lambda x: x * (1 - 5 * (np.exp(-(x * (np.sin(x) + 2)) ** 2 / 100))),
                        lambda x: x * (1 + 4 * x ** 3 * np.exp(-x ** 2 / 10))]
        # lambda x: x * (1 + 10 * (np.exp(-x ** 2 / 20 * np.exp(0.1 * x)))),
        self.n_mech = len(self.fct)
        self.prob_mech = [1/self.n_mech] * self.n_mech

        self.sample_mech()
        self.apply_vectorized = np.vectorize(self.apply_f)

    def sample_mech(self):
        self.mech = np.random.choice(self.n_mech,
                                     size=(self.tau + 1, self.d * self.d_z, self.d * self.d_z),
                                     p=self.prob_mech)
        self.coeff = sample_stationary_coeff(self.G, self.tau + 1, self.d, self.d_z, self.radius_correct)

    def apply_f(self, i, x):
        return self.fct[i](x)

    def apply(self, g, z, t):
        """Apply the mechanisms to z and add noise"""
        if t >= self.tau + 1:
            t = self.tau

        z = z.reshape(z.shape[0], 1, -1)
        z = z.repeat(1, z.shape[-1], 1)
        noise = np.random.normal(scale=self.noise_z_std, size=(self.d, self.d_z))

        if self.instantaneous:
            output = self.apply_vectorized(self.mech[-t-1:-1], z[:-1])
            output = output * self.coeff[-t-1:-1]
            output = np.sum(output, axis=(0, 2))
            output = output.reshape(self.d, self.d_z)
            output += noise

            for i in self.causal_order:
                output[:, i] += self.coeff[-1, i] @ self.apply_vectorized(self.mech[-1, i], output[0])
        else:
            output = self.apply_vectorized(self.mech[-t-1:], z)
            output = output * self.coeff[-t-1:]
            output = np.sum(output, axis=(0, 2))
            output = output.reshape(self.d, self.d_z)

        return torch.from_numpy(output), noise


class DataGeneratorWithLatent:
    """
    Code use to generate synthetic data with latent variables.
    Sample a DAG between latents and generate (X, Z).
    """
    def __init__(self, hp):
        self.hp = hp
        self.n = hp.n
        self.d = hp.d
        self.d_x = hp.d_x
        self.tau = hp.tau
        self.func_type = hp.func_type
        self.instantaneous = hp.instantaneous
        self.radius_correct = hp.radius_correct
        self.nonlinear_mixing = hp.nonlinear_mixing

        self.noise_x_std = hp.noise_x_std
        self.noise_z_std = hp.noise_z_std
        self.fixed_diagonal = hp.fixed_diagonal

        if self.n > 1:
            self.t = self.tau + 1
        else:
            self.t = hp.t

        self.d_z = hp.d_z
        self.prob = hp.prob

        # hp for MLP model
        self.num_layers = hp.num_layers
        self.num_hidden = hp.num_hidden
        self.non_linearity = hp.non_linearity

        assert self.d_x >= self.d_z, f"dx={self.d_x} should be larger or equal than dz={self.d_z}"

    def save_data(self, path: str):
        """ Save all files to 'path' """
        with open(os.path.join(path, "data_params.json"), "w") as file:
            json.dump(vars(self.hp), file, indent=4)
        with open(os.path.join(path, "best_metrics.json"), "w") as file:
            json.dump(self.best_metrics, file, indent=4)

        np.save(os.path.join(path, 'data_x'), self.X.detach().numpy())
        np.save(os.path.join(path, 'data_z'), self.Z.detach().numpy())
        np.save(os.path.join(path, 'graph'), self.G.detach().numpy())
        np.save(os.path.join(path, 'graph_w'), self.w.detach().numpy())


        if self.func_type == "linear":
            np.save(os.path.join(path, 'linear_coeff'), self.f.coeff)

    def sample_graph(self, instantaneous: bool) -> torch.Tensor:
        """
        Sample a random matrix that will be used as an adjacency matrix
        The diagonal is set to 1.
        Args:
            instantaneous: if True, sample a DAG for G[0]
        Returns:
            A Tensor of tau graphs between the Z (shape: tau x (d x d_z) x (d x d_z))
        """
        prob_tensor = torch.ones((self.tau + 1, self.d * self.d_z, self.d * self.d_z)) * self.prob
        # set diagonal to 1 of the graph G_{t-1}
        if self.fixed_diagonal:
            prob_tensor[-2, torch.arange(prob_tensor.size(1)), torch.arange(prob_tensor.size(2))] = 1

        G = torch.bernoulli(prob_tensor)

        if instantaneous:
            # for G_t sample a DAG
            G[-1], self.causal_order = self.sample_dag()
        else:
            # no instantaneous links, so set G_t to 0
            G[-1] = 0
            self.causal_order = None

        return G


    def sample_dag(self) -> Tuple[torch.Tensor, list]:
        """
        Sample a random DAG that will be used as an adjacency matrix
        for instantaneous connections
        Returns:
            A Tensor of tau graphs, size: (tau, d, num_neighbor x d)
            and a list containing the causal order of the variables
        """
        prob_tensor = torch.ones((1, self.d_z, self.d_z)) * self.prob
        # set all elements on and above the diagonal as 0
        prob_tensor = torch.tril(prob_tensor, diagonal=-1)

        G = torch.bernoulli(prob_tensor)

        # permutation
        causal_order = torch.randperm(self.d_z)
        G = G[:, causal_order]
        G = G[:, :, causal_order]
        assert is_acyclic(G[0])

        return G, causal_order

    def init_mlp_weight(self, model):
        """Function to initialize MLP's weight from a normal.
        In practice, gives more interesting functions than defaul
        initialization
        Args:
            model: a pytorch model (MLP for now)
        """
        if isinstance(model, torch.nn.modules.Linear):
            torch.nn.init.normal_(model.weight.data, mean=0., std=1)

    def sample_mlp(self, n_timesteps: int):
        """Sample a MLP that outputs the parameters for the distributions of Z
        Args:
            n_timesteps: Number of previous timesteps considered
        """
        mlps = nn.ModuleList()
        num_first_layer = n_timesteps * self.d * self.d_z
        num_last_layer = 2

        if self.non_linearity == "relu":
            nonlin = nn.ReLU()
        elif self.non_linearity == "leaky_relu":
            nonlin = nn.LeakyReLU()
        else:
            raise NotImplementedError("Nonlinearity is not implemented yet")

        for i_d in range(self.d):
            for k in range(self.d_z):
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

    def sample_w(self) -> torch.Tensor:
        """Sample matrices that are non-negative and orthogonal.
        They are the links between Z and X.
        Returns:
            A tensor w (shape: d, d_x, d_z)
        """
        mask = torch.zeros((self.d, self.d_x, self.d_z))
        for i in range(self.d):
            # assign d_xs uniformly to a cluster k
            cluster_assign = np.random.choice(self.d_z, size=self.d_x - self.d_z)
            cluster_assign = np.append(cluster_assign, np.arange(self.d_z))
            np.random.shuffle(cluster_assign)
            mask[i, np.arange(self.d_x), cluster_assign] = 1

        w = torch.empty((self.d, self.d_x, self.d_z)).uniform_(0.2, 1)
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
        # add a 'buffer' of 100 steps to make sure we get the stationary
        # distribution
        mixing_time = 10
        self.t += mixing_time

        # initialize Z for the first timesteps
        self.Z = torch.zeros((self.n, self.t, self.d, self.d_z))
        self.Z_mu = torch.zeros((self.n, self.t, self.d, self.d_z))
        self.X = torch.zeros((self.n, self.t, self.d, self.d_x))

        # sample graphs
        self.G = self.sample_graph(self.hp.instantaneous)

        # sample mechanisms
        if self.func_type == "mlp":
            self.f = [None]
            for t in range(1, self.tau + 1):
                self.f.append(self.sample_mlp(t))
        elif self.func_type == "add_nonlinear":
            self.f = StationaryMechanisms(self.G.numpy(), self.tau, self.d, self.d_z,
                                          self.causal_order,
                                          instantaneous=self.instantaneous,
                                          radius_correct=self.radius_correct,
                                          func_type=self.func_type,
                                          noise_z_std=self.noise_z_std)
        elif self.func_type == "linear":
            self.f = StationaryMechanisms(self.G.numpy(), self.tau, self.d, self.d_z,
                                          self.causal_order,
                                          instantaneous=self.instantaneous,
                                          radius_correct=self.radius_correct,
                                          func_type=self.func_type,
                                          noise_z_std=self.noise_z_std)
        elif self.func_type == "logistic_map":
            if self.tau > 1:
                raise ValueError("tau should equal 1 when using logistic maps")
            if not self.fixed_diagonal:
                raise ValueError("fixed_diagonal should equal True when using logistic maps")
            self.f = LogisticMechanisms(self.G.numpy(), self.tau, self.d, self.d_z,
                                        self.causal_order,
                                        instantaneous=self.instantaneous,
                                        radius_correct=self.radius_correct,
                                        func_type=self.func_type,
                                        noise_z_std=self.noise_z_std)
        elif self.func_type == "nonlinear":
            self.f = NonAdditiveStationaryMechanisms(self.G.numpy(), self.tau, self.d, self.d_z,
                                                     self.causal_order,
                                                     instantaneous=self.instantaneous,
                                                     radius_correct=self.radius_correct,
                                                     noise_z_std=self.noise_z_std)
        else:
            raise NotImplementedError("the only fct types are NN and stationary")

        # sample observational model
        self.w = self.sample_w()
        if self.nonlinear_mixing:
            self.mask = (self.w != 0)[0] * 1.
            self.mixing_f = MixingFunctions(self.mask, self.d_x, self.d_z)
            self.mask = self.mask / np.linalg.norm(self.mask, axis=0)

        for i_n in range(self.n):

            # sample the latent Z
            for t in range(self.t):
                if t == 0:
                    # for the first, sample from N(0, 1)
                    self.Z[i_n, t].normal_(0, self.noise_z_std)
                else:
                    # for the x first steps (< tau), apply the x first
                    # mechanisms
                    if t < self.tau + 1:
                        idx = t
                        g = self.G[-t-1:]
                        z = self.Z[i_n, :t+1]
                    else:
                        idx = -1
                        g = self.G
                        z = self.Z[i_n, t - self.tau:t + 1]

                    # nn_input = (g * z).view(-1)
                    if self.func_type == "mlp":
                        for i_d in range(self.d):
                            for k in range(self.d_z):
                                    nn_input = (z.view(z.shape[0], -1) * g[:, k + i_d * self.d_z]).view(1, -1)
                                    params = self.f[idx][k + i_d * self.d_z](nn_input)

                                    std = torch.ones_like(params[:, 1]) * 0.0001
                                    dist = distr.normal.Normal(params[:, 0], std)
                                    self.Z[i_n, t, i_d, k] = dist.rsample()
                    elif self.func_type == "add_nonlinear" or self.func_type == "linear":
                        self.Z_mu[i_n, t], noise = self.f.apply(g, z, t)
                        self.Z[i_n, t] = self.Z_mu[i_n, t] + noise
                    elif self.func_type == "nonlinear":
                        self.Z_mu[i_n, t], noise = self.f.apply(g, z, t)
                        self.Z[i_n, t] = self.Z_mu[i_n, t] + noise
                    elif self.func_type == "logistic_map":
                        self.Z_mu[i_n, t], noise = self.f.apply(g, z, t)
                        self.Z[i_n, t] = self.Z_mu[i_n, t]

            # sample the data X (= WZ + noise)
            if self.nonlinear_mixing:
                z = self.Z.reshape(1, self.Z.shape[1], 1, self.Z.shape[-1])
                z = z.repeat(1, 1, self.w.shape[1], 1)
                self.X_masked = self.mask * z[0]
                self.X_mu = self.mixing_f(self.X_masked)
                self.X_mu = self.X_mu.reshape(1, self.X_mu.shape[0], 1, self.X_mu.shape[1])
            else:
                self.X_mu = torch.einsum('dxz, ntdz -> ntdx', self.w, self.Z)
            self.X = self.X_mu + torch.normal(0, self.noise_x_std, size=self.X.size())

        self.t -= mixing_time

        self.X = self.X[:, mixing_time:]
        print(self.X.shape)
        self.X_mu = self.X_mu[:, mixing_time:]
        self.Z = self.Z[:, mixing_time:]
        self.Z_mu = self.Z_mu[:, mixing_time:]
        return self.X, self.Z


    def compute_metrics(self):
        metrics = {}
        # WARNING: these metrics are not all valid when graph contains
        # instantaneous connections!
        # get recons term
        px_distr = distr.normal.Normal(self.X_mu, self.noise_x_std)
        metrics["recons"] = torch.mean(torch.sum(px_distr.log_prob(self.X), dim=3)).item()

        # get KL term
        # encode q(Zt | Xt)
        q_mu = torch.einsum('dxz, ntdx -> ntdz', self.w, self.X)
        q = distr.normal.Normal(q_mu, self.noise_z_std)

        # get p(Zt | Z<t)
        p = distr.normal.Normal(self.Z_mu, self.noise_z_std)
        kl = torch.sum(distr.kl_divergence(q, p), 3).mean().item()
        metrics["kl"] = kl

        # get MCC
        # mcc = np.corrcoef(p.sample().numpy().reshape(self.n, -1),
        #                   self.Z.numpy().reshape(self.n, -1))
        # metrics["mcc_stoch"] = mcc[0, 1]

        mcc = np.corrcoef(self.Z_mu.reshape(self.n, -1),
                          self.Z.numpy().reshape(self.n, -1))
        metrics["mcc"] = mcc[0, 1]

        # ELBO with GT model
        metrics["elbo"] = metrics["recons"] - metrics["kl"]
        print(metrics)
        self.best_metrics = metrics


class DataGeneratorWithoutLatent:
    """
    Code use to generate synthetic data without latent variables
    """
    def __init__(self, hp):
        self.hp = hp
        self.n = hp.n
        self.d = hp.d
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
            self.t = hp.t

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
        # prob_tensor = torch.ones((self.tau, self.d, self.num_neigh * self.d)) * self.prob
        prob_tensor = torch.ones((self.tau, self.d, self.num_neigh * self.d)) * self.prob

        if diagonal:
            # set diagonal to 1
            if self.fixed_diagonal:
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
                        if self.d_x == 1:
                            # print(self.G.size())
                            # print(self.X.size())
                            # print(self.X[i_n, t - self.tau:t + t1, :, :self.d].size())
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

if __name__ == "__main__":
    sample_stationary_coeff(2, 3, 2)
