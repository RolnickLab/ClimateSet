import os
import torch
import numpy as np
from typing import Tuple


class DataLoader:
    # TODO: adapt for interventions
    def __init__(self,
                 ratio_train: float,
                 ratio_valid: float,
                 data_path: str,
                 latent: bool,
                 instantaneous: bool,
                 tau: int):
        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        if ratio_valid == 0:
            self.ratio_valid = 1 - ratio_train
        assert ratio_train + ratio_valid <= 1

        self.data_path = data_path
        self.latent = latent
        self.instantaneous = instantaneous
        self.tau = tau

        # Load and split the data
        self._load_data()
        self.n = self.x.shape[0]
        self.d = self.x.shape[2]
        self.d_x = self.x.shape[3]
        self._split_data()

    def _load_data(self):
        self.x = np.load(os.path.join(self.data_path, 'data_x.npy'))
        if self.latent:
            self.z = np.load(os.path.join(self.data_path, 'data_z.npy'))
            self.gt_w = np.load(os.path.join(self.data_path, 'graph_w.npy'))
            self.gt_graph = np.load(os.path.join(self.data_path, 'graph_z.npy'))
        else:
            self.gt_graph = np.load(os.path.join(self.data_path, 'graph.npy'))

    def _split_data(self):
        t_max = self.x.shape[1]

        # TODO: be more general
        if self.n == 1:
            n_train = int(t_max * self.ratio_train)
            n_valid = int(t_max * self.ratio_valid)
            self.idx_train = np.arange(n_train)
            self.idx_valid = np.arange(n_train - self.tau, n_train + n_valid)
            self.x_train = self.x[:, self.idx_train]
            self.x_valid = self.x[:, self.idx_valid]
        else:
            n_train = int(self.n * self.ratio_train)
            n_valid = int(self.n * self.ratio_valid)
            self.idx_train = np.arange(n_train)
            self.idx_valid = np.arange(n_train - self.tau, n_train + n_valid)
            np.random.shuffle(self.idx_train)
            np.random.shuffle(self.idx_valid)
            self.x_train = self.x[self.idx_train]
            self.x_valid = self.x[self.idx_valid]

    def _sample(self, dataset: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if dataset.shape[0] <= 0:
            __import__('ipdb').set_trace()

        if self.instantaneous:
            x = np.zeros((batch_size, self.tau + 1, self.d, self.d_x))
            t1 = 1
        else:
            x = np.zeros((batch_size, self.tau, self.d, self.d_x))
            t1 = 0
        y = np.zeros((batch_size, self.d, self.d_x))

        if self.n == 1:
            random_idx = np.random.choice(np.arange(self.tau, dataset.shape[1]), replace=False, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = dataset[0, idx - self.tau:idx + t1]
                y[i] = dataset[0, idx]
        else:
            random_idx = np.random.choice(np.arange(dataset.shape[0]), replace=False, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = dataset[idx, 0:self.tau + t1]
                y[i] = dataset[idx, self.tau]
        return torch.tensor(x), torch.tensor(y)

    def sample_train(self, batch_size: int) -> torch.Tensor:
        return self._sample(self.x_train, batch_size)

    def sample_valid(self, batch_size: int) -> torch.Tensor:
        return self._sample(self.x_valid, batch_size)
