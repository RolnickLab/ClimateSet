# phillipe's code

import os
import torch
import tables  # for handling the h5py files
import numpy as np
from typing import Tuple
import xarray as xr
from mother_data.utils.helper_funcs import get_keys_from_value
from mother_data.utils.constants import VAR_SOURCE_LOOKUP, RES_TO_CHUNKSIZE

# from geopy import distance


class DataLoader:
    def __init__(
        self,
        ratio_train: float,
        ratio_valid: float,
        data_path: str,
        latent: bool,  # if we use a model that has latent variables e.g. spatial aggregation
        no_gt: bool,  # if we have a ground-truth causal graph to compare with, If True, does not use any ground-truth for plotting and metrics
        debug_gt_w: bool,  # If true, use the ground truth graph (use only to debug)"
        instantaneous: bool,  # Use instantaneous connections
        tau: int,  # time window size
        years: [str],
        experiments: [str],
        vars: [str],
        resolution="250_km",
        freq="mon",
    ):
        """
        Simple DataLoader (extended from Phillipe to fit real climate model data.)
        Resulting dimension of data: (n, t, d, d_x)
        n = number of time series i.e. years
        t = numbernumber of time steps ie. frequency (12 for months...)
        d = number of physical variables
        d_x = the number of grid locations

        @params:
            ratio_train (float): ratio of the dataset to be used for training
            ratio_valid (float): ratio of the dataset to be used for validation
            data_path (str): path to the preprocessed data
            latent (bool): if we use a model that has latent variables e.g. spatial aggregation
            no_gt (bool): if we have a ground-truth causal graph to compare with, If True, does not use any ground-truth for plotting and metrics
            debug_gt_w (bool): If true, use the ground truth graph (use only to debug)"
            instantaneous (bool): Use instantaneous connections
            tau (int): time window size
            years ([str]): list of years to take data from
            experiments ([str]): list of experiments to take data from
            vars ([str]): list of variables to include
            resolution (str): desired nominal resolution
            freq (str): desired temporal resolution
        """

        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        self.data_path = data_path
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

        self.years = years
        self.experiments = experiments
        self.resolution = resolution
        self.freq = freq

        vars = [v.replace(" ", "_").replace("-", "_") for v in vars]

        self.model_vars = []
        self.forcing_vars = []
        for v in vars:
            t = get_keys_from_value(VAR_SOURCE_LOOKUP, v)
            if t == "model":
                self.model_vars.append(v)
            elif t == "raw":
                self.forcing_vars.append(v)

            else:
                print(
                    f"WARNING: unknown source type for var {v}. Not supported. Skipping."
                )

        # Load and split the data
        self._load_data()
        self._split_data()

    def _load_data(self, shuffle=True):
        """
        Open and store the data files.

        @params:
            shuffle (bool): if the data should be shuffled before loading
        """

        # build correct path
        res_freq = f"/{self.resolution}/{self.freq}/"

        n = len(self.experiments) * len(self.years)
        # num of target vars
        d_m = len(self.model_vars)
        # num_of forcings
        d_f = len(self.forcing_vars)
        t = RES_TO_CHUNKSIZE[self.freq]

        d_x = (96, 144)  # TODO NOM RES TO GRID LOOKUP
        data = []

        # handle cmip6 target variables
        if d_m >= 1:
            # empty array for storing
            model_data = np.zeros((n, t, d_m, *d_x))
            for i, v in enumerate(self.model_vars):
                for j, e in enumerate(self.experiments):
                    for k, y in enumerate(self.years):
                        # create path names
                        path = f"{self.data_path}CMIP6/{e}/{v}{res_freq}{y}/"
                        f_name = os.listdir(path)[0]

                        model_data[j + k, :, i, :] = np.asarray(
                            tables.open_file(path + f_name, mode="r").root._f_get_child(
                                v
                            )
                        )

            data.append(model_data)

        #  handle input4mips variables
        if d_f >= 1:
            forcing_data = np.zeros((n, t, d_f, *d_x))
            for i, v in enumerate(self.forcing_vars):
                for j, e in enumerate(self.experiments):
                    for k, y in enumerate(self.years):
                        path = f"{self.data_path}input4mips/{e}/{v}{res_freq}{y}/"
                        f_name = os.listdir(path)[0]

                        forcing_data[j + k, :, i, :] = np.asarray(
                            tables.open_file(path + f_name, mode="r").root._f_get_child(
                                v
                            )
                        )

            data.append(forcing_data)

        # combine all variables
        self.data = np.concatenate(data)
        # read out dimensions
        self.n = self.data.shape[0]
        self.t = self.data.shape[1]
        self.d = self.data.shape[2]
        # flatten out grid cells
        self.data = np.reshape(self.data, (self.n, self.t, self.d, -1))
        self.d_x = self.data.shape[-1]

        if shuffle:
            np.random.shuffle(self.data)
        print("DATA shape", self.data.shape)

        # use coordinates if using real-world datasets
        # if self.no_gt:
        #    self.coordinates = np.load(os.path.join(self.data_path, 'coordinates.npy'))
        #    print("got coords")
        #    # TODO: decide if keep or not
        #    # self.distances = self.get_geodisic_distances(self.coordinates)

    def _split_data(self):
        """
        Determine the indices for training and validation sets.
        """
        t_max = (
            self.t
        )  # should in general equal the chunksize e.g. monthly data -> max 12 per year

        # only on time series
        if self.n == 1:
            # index along the time dimension
            self.n_train = int(t_max * self.ratio_train)
            self.n_valid = int(t_max * self.ratio_valid)
            self.idx_train = np.arange(self.tau, self.n_train)
            self.idx_valid = np.arange(
                self.n_train - self.tau, self.n_train + self.n_valid
            )

            # if self.latent:
            #     self.z_train = self.z[:, self.idx_train]
            #     self.z_valid = self.z[:, self.idx_valid]

        #  handle multiple time series
        else:
            # number of time series to train on
            self.n_train = int(self.n * self.ratio_train)
            # number of validation time series
            self.n_valid = int(self.n * self.ratio_valid)
            self.idx_train = np.arange(self.n_train)
            self.idx_valid = np.arange(self.n_train, self.n_train + self.n_valid)

            np.random.shuffle(self.idx_train)
            np.random.shuffle(self.idx_valid)

            # if self.latent:
            #     self.z_train = self.z[self.idx_train]
            #     self.z_valid = self.z[self.idx_valid]

    def sample(self, batch_size: int, valid: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sampling sliced time series from the trainin / validation dataset.

        @params:
            batch_size (int): the number of examples in a minibatch
            valid (bool): if True, sample from validation set
        @returns:
            x (tensor): of the data with time-series (batch_size, tau, d, d_x)
            y (tensor): data to predict (batch_size, d, d_x)

        """

        # initiliaze empty arrays
        if (
            self.instantaneous
        ):  # if we have instantaneous connections consider x_{t} .. x_{t-i}
            x = np.zeros((batch_size, self.tau + 1, self.d, self.d_x))
            t1 = 1  #  modifier

            """ depricated
            if not self.no_gt and self.latent: # if we have ground truth and are dealing with latents
                z = np.zeros((batch_size, self.tau + 2, self.d, self.k))
            else:
                z = None
            """

        else:  # only consider x_{t-1} ... x_{t-i}
            x = np.zeros((batch_size, self.tau, self.d, self.d_x))
            t1 = 0
            """ deprecated (?)
            if not self.no_gt and self.latent:
                z = np.zeros((batch_size, self.tau + 1, self.d, self.k))
            else:
                z = None
            """

        # targets without time-windows
        y = np.zeros((batch_size, self.d, self.d_x))

        if valid:
            dataset_idx = self.idx_valid
        else:
            dataset_idx = self.idx_train

        # considering a single long time series
        if self.n == 1:

            # select a random index along the time dimension

            # allow replacement here because short time series #TODO clarify if we want that and when / when not
            random_idx = np.random.choice(dataset_idx, replace=True, size=batch_size)
            for i, idx in enumerate(random_idx):

                # first dimension is 1
                x[i, :] = np.squeeze(self.data)[idx - self.tau : idx + t1, :, :]
                y[i, :] = np.squeeze(self.data)[idx + t1]

        # if there are multiple timeseries
        else:

            # with replacement cause bigger batch_size than number of time series
            random_idx = np.random.choice(dataset_idx, replace=True, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = self.data[idx, 0 : self.tau + t1]
                y[i] = self.data[idx, self.tau + t1]

                """
                if not self.no_gt and self.latent:
                    z[i] = self.z[idx, 0:self.tau + t1 + 1]
                """

        # convert to torch tensors
        x_ = torch.tensor(x)
        y_ = torch.tensor(y)

        """
        if not self.no_gt and self.latent:
            z_ = torch.tensor(z)
        else:
            z_ = z

        #print(x_, y_, z_)
        """
        return x_, y_


if __name__ == "__main__":
    ratio_train = 0.8
    ratio_valid = 0.2
    data_path = "/home/charlie/Documents/MILA/causalpaca/data/data_processed/"  # /home/charlie/Documents/MILA/causalpaca/data/test_data/"
    years = ["2016"]  # ,"2020"]
    experiments = ["ssp126"]  # , "ssp370"]
    vars = ["pr", "tas"]

    data_loader = DataLoader(
        ratio_train,
        ratio_valid,
        data_path,
        latent=True,
        no_gt=True,
        debug_gt_w=False,
        instantaneous=False,
        tau=4,
        years=years,
        experiments=experiments,
        vars=vars,
    )

    x, y = data_loader.sample(8, False)

    print(x.size(), y.size())
