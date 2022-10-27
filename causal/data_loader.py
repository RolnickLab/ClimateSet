import os
import torch
import tables
import numpy as np
from typing import Tuple
from geopy import distance


class DataLoader:
    """
    Load data, can deal with numpy and hdf5 files.
    For numpy, keep the file in memory, whereas for hdf5 sample only the
    desired examples.
    Usual dimension of data: (n, t, d, d_x)
    """
    # TODO: adapt for interventions
    def __init__(self,
                 ratio_train: float,
                 ratio_valid: float,
                 data_path: str,
                 data_format: str,
                 latent: bool,
                 no_gt: bool,
                 debug_gt_w: bool,
                 instantaneous: bool,
                 tau: int):
        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        self.data_path = data_path
        self.data_format = data_format
        self.latent = latent
        self.no_gt = no_gt
        self.debug_gt_w = debug_gt_w
        self.instantaneous = instantaneous
        self.tau = tau

        self.gt_graph = None
        self.z = None  # use with latent model
        self.gt_w = None
        self.coordinates = None  # used when using real-world data
        self.d_z = 0

        # Load and split the data
        self._load_data()
        self._split_data()

    def _load_data(self):
        """
        Open the data files and the ground-truth graph if it exists
        """
        if self.data_format == "numpy":
            self.x = np.load(os.path.join(self.data_path, 'data_x.npy'))

            if not self.no_gt:
                self.gt_graph = np.load(os.path.join(self.data_path, 'graph.npy'))
                if self.latent:
                    self.z = np.load(os.path.join(self.data_path, 'data_z.npy'))
                    self.gt_w = np.load(os.path.join(self.data_path, 'graph_w.npy'))
        elif self.data_format == "hdf5":
            f = tables.open_file(os.path.join(self.data_path, 'data.h5'), mode='r')
            self.x = f.root.data

        # names for X, Z dimensions
        self.n = self.x.shape[0]
        self.d = self.x.shape[2]
        self.d_x = self.x.shape[3]
        if not self.no_gt and self.latent:
            self.d_z = self.z.shape[3]

        # use coordinates if using real-world datasets
        if self.no_gt:
            self.coordinates = np.load(os.path.join(self.data_path, 'coordinates.npy'))

    def _split_data(self):
        """
        Determine the indices for training and validation sets
        """
        t_max = self.x.shape[1]

        # TODO: be more general, n > 1 with different t
        if self.n == 1:
            self.n_train = int(t_max * self.ratio_train)
            self.n_valid = int(t_max * self.ratio_valid)
            self.idx_train = np.arange(self.tau, self.n_train)
            self.idx_valid = np.arange(self.n_train - self.tau, self.n_train + self.n_valid)
        else:
            self.n_train = int(self.n * self.ratio_train)
            self.n_valid = int(self.n * self.ratio_valid)
            self.idx_train = np.arange(self.tau, self.n_train)
            self.idx_valid = np.arange(self.n_train - self.tau, self.n_train + self.n_valid)
            np.random.shuffle(self.idx_train)
            np.random.shuffle(self.idx_valid)

    def sample(self, batch_size: int, valid: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_size: the number of examples in a minibatch
            valid: if True, sample from validation set
        Returns:
            x, y, z: tensors of the data x, the data to predict y, and the latent variables z
        """

        # initiliaze empty arrays
        if self.instantaneous:
            x = np.zeros((batch_size, self.tau + 1, self.d, self.d_x))
            if not self.no_gt and self.latent:
                z = np.zeros((batch_size, self.tau + 2, self.d, self.d_z))
            else:
                z = None
            t1 = 1
        else:
            x = np.zeros((batch_size, self.tau, self.d, self.d_x))
            if not self.no_gt and self.latent:
                z = np.zeros((batch_size, self.tau + 1, self.d, self.d_z))
            else:
                z = None
            t1 = 0
        y = np.zeros((batch_size, self.d, self.d_x))

        if valid:
            dataset_idx = self.idx_valid
        else:
            dataset_idx = self.idx_train

        if self.n == 1:
            # if there is only one long timeserie
            random_idx = np.random.choice(dataset_idx, replace=False, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = self.x[0, idx - self.tau:idx + t1]
                y[i] = self.x[0, idx + t1]
                if not self.no_gt and self.latent:
                    z[i] = self.z[0, idx - self.tau:idx + t1 + 1]
        else:
            # if there are multiple timeseries
            random_idx = np.random.choice(dataset_idx, replace=False, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = self.x[idx, 0:self.tau + t1]
                y[i] = self.x[idx, self.tau + t1]
                if not self.no_gt and self.latent:
                    z[i] = self.z[idx, 0:self.tau + t1 + 1]

        # convert to torch tensors
        x_ = torch.tensor(x)
        y_ = torch.tensor(y)
        if not self.no_gt and self.latent:
            z_ = torch.tensor(z)
        else:
            z_ = z

        return x_, y_, z_

    def get_geodisic_distances(self, coordinates: np.ndarray):
        """
        Calculate the distance matrix between every pair of coordinates.
        Use the geodesic distance with the WGS-84 model.

        might be too slow for 10000 grid locations...
        """
        d = np.zeros((self.d_x, self.d_x))
        for i, c1 in enumerate(coordinates):
            for j, c2 in enumerate(coordinates):
                if i == j:
                    d[i, j] = d[j, i] = 0
                else:
                    # by default, use the WGS-84 model
                    d[i, j] = d[j, i] = distance.geodesic(c1, c2).km

        return d
