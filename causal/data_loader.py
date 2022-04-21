import os
import torch
import numpy as np
from typing import Tuple


class DataLoader:
    # TODO: adapt if have multiple timeseries
    # TODO: adapt for interventions
    def __init__(self,
                 ratio_train: float,
                 ratio_valid: float,
                 data_path: str,
                 latent: bool,
                 tau: int):
        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        if ratio_valid == 0:
            self.ratio_valid = 1 - ratio_train
        assert ratio_train + ratio_valid <= 1

        self.data_path = data_path
        self.latent = latent
        self.tau = tau

        # Load and split the data
        self._load_data()
        self.d = self.x.shape[1]
        self.d_x = self.x.shape[2]
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
        n = self.x.shape[0]
        n_train = int(n * self.ratio_train)
        n_valid = int(n * self.ratio_valid)
        self.idx_train = np.arange(n_train)
        self.idx_valid = np.arange(n_train - self.tau, n_train + n_valid)
        self.x_train = self.x[self.idx_train]
        self.x_valid = self.x[self.idx_valid]

    def _sample(self, dataset: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if dataset.shape[0] <= 0:
            __import__('ipdb').set_trace()
        random_idx = np.random.choice(np.arange(self.tau, dataset.shape[0]), replace=False, size=batch_size)
        x = np.zeros((batch_size, self.tau, self.d, self.d_x))
        y = np.zeros((batch_size, self.d, self.d_x))

        for i, idx in enumerate(random_idx):
            x[i] = dataset[idx - self.tau:idx]
            y[i] = dataset[idx]
        return torch.tensor(x), torch.tensor(y)

    def sample_train(self, batch_size: int) -> torch.Tensor:
        return self._sample(self.x_train, batch_size)

    def sample_valid(self, batch_size: int) -> torch.Tensor:
        return self._sample(self.x_valid, batch_size)
