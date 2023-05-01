import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_hidden: int,
                 num_input: int,
                 num_output: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output

        module_dict = OrderedDict()

        # create model layer by layer
        in_features = num_input
        out_features = num_hidden
        if num_layers == 0:
            out_features = num_output

        module_dict['lin0'] = nn.Linear(in_features, out_features)

        for layer in range(num_layers):
            in_features = num_hidden
            out_features = num_hidden

            if layer == num_layers - 1:
                out_features = num_output

            module_dict[f'nonlin{layer}'] = nn.LeakyReLU()
            module_dict[f'lin{layer+1}'] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class LinearMasked(nn.Module):
    def __init__(self, adj, W):
        super().__init__()
        # torch.zeros_like
        self.weights = nn.Parameter(torch.Tensor(np.zeros_like(adj)))
        self.W = torch.Tensor(W)
        self.adj = torch.Tensor(adj)


    def forward(self, x_):
        # out = torch.matmul(data, self.adj * self.weights)
        linear_weights = self.adj * self.weights
        z = torch.einsum("nxt,xz->nzt", x_, self.W)

        x = z[:, :, :-1]
        y = x_[:, :, -1]

        # TODO: make sure it is column that are parents
        z_hat = torch.einsum("ndt,cdt->nc", x, linear_weights)
        y_hat = torch.einsum("nz,xz->nx", z_hat, self.W)
        return torch.mean(torch.sum(0.5 * ((y - y_hat))**2, dim=1))


class NNMasked(nn.Module):
    def __init__(self, adj, W, d_z, num_hidden: int = 8, num_layers: int = 2):
        super().__init__()
        self.d_z = d_z
        self.W = torch.Tensor(W)
        self.adj = torch.Tensor(adj[:, :, 1:])
        self.mlps = nn.ModuleList(MLP(num_layers, num_hidden, d_z, 1) for i in range(d_z))

    def forward(self, x_):
        z = torch.einsum("nxt,xz->nzt", x_, self.W)
        z_hat = torch.zeros((x_.shape[0], self.d_z))
        x = z[:, :, :-1]
        y = x_[:, :, -1]

        for i in range(self.d_z):
            z_hat[:, i] = self.mlps[i]((x * self.adj[i]).reshape(x.shape[0], -1)).squeeze()

        y_hat = torch.einsum("nz,xz->nx", z_hat, self.W)

        return torch.mean(torch.sum(0.5 * ((y - y_hat))**2, dim=1))


def train(adj, data, W, idx_train, idx_valid, max_iter, batch_size, linear=True,
          num_hidden=8, num_layers=2):
    tau = adj.shape[-1]
    best_valid_score = -np.inf
    full_patience = 100
    flag_max_iter = True
    if linear:
        model = LinearMasked(adj, W)
        optimizer = torch.optim.Adagrad(model.parameters())
    else:
        model = NNMasked(adj, W, adj.shape[0], num_hidden, num_layers)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)

    for iter in range(max_iter):
        x = sample(data, idx_train, batch_size, tau)

        loss = model(x)
        train_score = -loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x = sample(data, idx_valid, None, tau)
            valid_score = -model(x).item()

        if valid_score > best_valid_score + 1e-4:
            best_valid_score = valid_score
            # compute best model training score
            x = sample(data, idx_train, None, tau)
            best_train_score = -model(x).item()
            # restore patience
            patience = full_patience
        else:
            patience -= 1

        if iter % 10 == 0:
            print(f"Iteration: {iter}, score_train: {train_score:.5f} , score_valid : {valid_score:.5f}, \
                  best_train_score : {best_train_score:.5f}, best_valid_score {best_valid_score:.5f}, \
                  patience: {patience}")
        if patience == 0:
            flag_max_iter = False
            break

    # print(model.weights)

    return -best_train_score, -best_valid_score, flag_max_iter


def sample(data, idx_train, n, tau):
    samples = np.zeros((idx_train.shape[0], data.shape[1], tau))
    # remove idx too close to the end of the time-serie
    # idx_train = idx_train[:-(tau + 1)]

    if n is not None:
        sampled_idx = np.random.choice(idx_train, n)
    else:
        sampled_idx = idx_train

    for i, idx in enumerate(sampled_idx):
        samples[i] = data[idx: idx + tau].T

    return torch.from_numpy(samples).float()
