import torch
import torch.nn as nn
import numpy as np


class LinearMasked(nn.Module):
    def __init__(self, adj):
        super().__init__()
        # torch.zeros_like
        self.weights = nn.Parameter(torch.Tensor(np.zeros_like(adj)))
        self.adj = torch.Tensor(adj)

    def forward(self, x):
        # out = torch.matmul(data, self.adj * self.weights)
        w = self.adj * self.weights
        x = x[:, :, :-1]
        y = x[:, :, -1]

        # TODO: make sure it is column that are parents
        y_hat = torch.einsum("ndt,cdt->nc", x, w)
        return torch.mean(torch.sum(0.5 * ((y - y_hat))**2, dim=1))


def train(adj, data, idx_train, idx_valid, max_iter, batch_size):
    tau = adj.shape[-1]
    best_valid_score = -np.inf
    full_patience = 100
    flag_max_iter = True
    model = LinearMasked(adj)
    optimizer = torch.optim.Adagrad(model.parameters())

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

        if iter % 100 == 0:
            print("Iteration: {iter}, score_train: {train_score:.5f} , score_valid : {valid_score:.5f}, \
                  best_train_score : {best_train_score:.5f}, best_valid_score {best_valid_score:.5f}, \
                  patience: {patience}"
        if patience == 0:
            flag_max_iter = False
            break

    print(model.weights)

    return -best_train_score, -best_valid_score, flag_max_iter


def sample(data, idx_train, n, tau):
    samples = np.zeros((data.shape[0], data.shape[1], tau + 1))
    # remove idx too close to the end of the time-serie
    idx_train = idx_train[:-(tau + 1)]

    if n is not None:
        sampled_idx = np.random.choice(idx_train, n)
    else:
        sampled_idx = idx_train

    for i, idx in enumerate(sampled_idx):
        samples[i] = data[idx: idx + tau + 1].T

    return torch.from_numpy(samples).float()
