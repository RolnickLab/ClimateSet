# phillipe's code

import os
import torch
import tables  # for handling the h5py files
import numpy as np
from typing import Tuple
import xarray as xr

# from geopy import distance


class DataLoader:
    """

    Load data, can deal with numpy and hdf5 files.
    For numpy, keep the file in memory, whereas for hdf5 sample only the
    desired examples.


    for us:
        - the dataloader should probably look at a list of vars and years to consider and get all the respective files from the directory and load together on demand
        - so we should have a directory of .h5 files and not a singe file (OPENQUESTION: if time and grid stays the same, this might lead to lots redundant data, but assuming storing everything in one will get too big and inflexible)
        - to do in preprocessing: create .h5 files

    to do here:
        - read out correct dimensionalities from our own h5 files
        - scrapping together from list of years, scenarios, vars etc.

    - in: .h5 files


    Usual dimension of data: (n, t, d, d_x)
    n = number of time series i.e. years
    t = number of time steps ie. frequency (12 for months...)
    d = number of physical variables
    d_x = the number of grid locations
    """

    # TODO: adapt for interventions
    def __init__(
        self,
        ratio_train: float,
        ratio_valid: float,
        data_path: str,
        data_format: str,  # 'numpy' or 'hdf5'
        latent: bool,  # if we use a model that has latent variables e.g. spatial aggregation
        no_gt: bool,  # if we have a ground-truth causal graph to compare with, If True, does not use any ground-truth for plotting and metrics
        debug_gt_w: bool,  # If true, use the ground truth graph (use only to debug)"
        instantaneous: bool,  # Use instantaneous connections
        tau: int,
    ):  # time window size

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
        self.k = 0

        # Load and split the data
        self._load_data()
        self._split_data()

    def _load_data(self):
        """
        Open the data files and the ground-truth graph if it exists
        """
        if self.data_format == "numpy":
            self.x = np.load(os.path.join(self.data_path, "data_x.npy"))

            if not self.no_gt:
                self.gt_graph = np.load(os.path.join(self.data_path, "graph.npy"))
                if self.latent:
                    self.z = np.load(os.path.join(self.data_path, "data_z.npy"))
                    self.gt_w = np.load(os.path.join(self.data_path, "graph_w.npy"))

        elif self.data_format == "hdf5":
            f = tables.open_file(os.path.join(self.data_path, "data_phi.h5"), mode="r")
            print(f)
            print(f.root)
            ds_phi = xr.open_dataset(
                os.path.join(self.data_path, "data_phi.h5"), mode="r"
            )
            print(ds_phi)
            ds = xr.open_dataset(
                "/home/charlie/Documents/MILA/causalpaca/data/data_processed/CMIP6/ssp126/pr/250_km/mon/2016/CMIP6_NorESM2-LM_r1i1p1f1_ssp126_pr_250_km_mon_gn_2016.h5",
                mode="r",
            )
            print(ds)
            self.x = f.root.data

        # names for X, Z dimensions
        self.n = self.x.shape[0]  #  num time series
        self.d = self.x.shape[2]  # number of physical variables
        self.d_x = self.x.shape[3]  # number of grid location
        if not self.no_gt and self.latent:
            self.k = self.z.shape[3]

        print(f"n {self.n} d {self.d} d_x {self.d_x}")

        # use coordinates if using real-world datasets
        if self.no_gt:
            self.coordinates = np.load(os.path.join(self.data_path, "coordinates.npy"))
            print("got coords")
            # TODO: decide if keep or not
            # self.distances = self.get_geodisic_distances(self.coordinates)

    def _split_data(self):
        """
        Determine the indices for training and validation sets
        """
        t_max = self.x.shape[1]
        print("tmax", t_max)

        # TODO: be more general
        if self.n == 1:
            self.n_train = int(t_max * self.ratio_train)
            self.n_valid = int(t_max * self.ratio_valid)
            self.idx_train = np.arange(self.tau, self.n_train)
            self.idx_valid = np.arange(
                self.n_train - self.tau, self.n_train + self.n_valid
            )
            # self.x_train = self.x[:, self.idx_train]
            # self.x_valid = self.x[:, self.idx_valid]
            # if self.latent:
            #     self.z_train = self.z[:, self.idx_train]
            #     self.z_valid = self.z[:, self.idx_valid]
        else:
            self.n_train = int(self.n * self.ratio_train)
            self.n_valid = int(self.n * self.ratio_valid)
            self.idx_train = np.arange(self.tau, self.n_train)
            self.idx_valid = np.arange(
                self.n_train - self.tau, self.n_train + self.n_valid
            )
            np.random.shuffle(self.idx_train)
            np.random.shuffle(self.idx_valid)
            # self.x_train = self.x[self.idx_train]
            # self.x_valid = self.x[self.idx_valid]
            # if self.latent:
            #     self.z_train = self.z[self.idx_train]
            #     self.z_valid = self.z[self.idx_valid]

    def sample(self, batch_size: int, valid: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_size: the number of examples in a minibatch
            valid: if True, sample from validation set
        Returns:
            x: tensor of the data with time-series (batch_size, tau, d, d_x)
            y: data to predict (batch_size, d, d_x)
            z: latent variables ??

        """

        # initiliaze empty arrays
        if (
            self.instantaneous
        ):  # if we have instantaneous connections consider x_{t} .. x_{t-i}

            # we want to sample over time
            x = np.zeros((batch_size, self.tau + 1, self.d, self.d_x))

            if (
                not self.no_gt and self.latent
            ):  # if we have ground truth and are dealing with latents
                z = np.zeros((batch_size, self.tau + 2, self.d, self.k))
            else:
                z = None
            t1 = 1
        else:  # only consider x_{t-1} ... x_{t-i}
            x = np.zeros((batch_size, self.tau, self.d, self.d_x))
            if not self.no_gt and self.latent:
                z = np.zeros((batch_size, self.tau + 1, self.d, self.k))
            else:
                z = None
            t1 = 0
        # targets without time-windows
        y = np.zeros((batch_size, self.d, self.d_x))

        if valid:
            dataset_idx = self.idx_valid
        else:
            dataset_idx = self.idx_train

        print("dataset index", dataset_idx)

        if self.n == 1:
            # if there is only one long

            #  we sample random indexes along the itme dimension of the dataset, as many as specified by batch size
            random_idx = np.random.choice(dataset_idx, replace=False, size=batch_size)
            print("batch size", batch_size)
            print("random indx")
            print(random_idx)

            for i, idx in enumerate(random_idx):
                print(i, idx)

                # from the index we go and select the time window around it
                # TODO: question: what happens if we e.g. select an index smaller than tau, we would slice back to the end -> does not make sense in real world data
                # nevermind, that happens in setting up the dataset_idx
                # fill up the batch
                x[i] = self.x[0, idx - self.tau : idx + t1]
                y[i] = self.x[0, idx + t1]
                if not self.no_gt and self.latent:
                    z[i] = self.z[0, idx - self.tau : idx + t1 + 1]
        else:
            # if there are multiple timeseries
            random_idx = np.random.choice(dataset_idx, replace=False, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = self.x[idx, 0 : self.tau + t1]
                y[i] = self.x[idx, self.tau + t1]
                if not self.no_gt and self.latent:
                    z[i] = self.z[idx, 0 : self.tau + t1 + 1]

        # convert to torch tensors
        x_ = torch.tensor(x)
        y_ = torch.tensor(y)
        if not self.no_gt and self.latent:
            z_ = torch.tensor(z)
        else:
            z_ = z

        # print(x_, y_, z_)

        return x_, y_, z_


if __name__ == "__main__":
    ratio_train = 0.8
    ratio_valid = 0.2
    data_path = "/home/charlie/Documents/MILA/causalpaca/data/test_data/"
    data_format = "hdf5"

    data_loader = DataLoader(
        ratio_train,
        ratio_valid,
        data_path,
        data_format,
        latent=True,
        no_gt=True,
        debug_gt_w=False,
        instantaneous=False,
        tau=4,
    )

    x, y, z = data_loader.sample(8, False)

    print(x.size(), y.size())
