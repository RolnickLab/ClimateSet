import os
import tables
import numpy as np


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
            self.x = np.asarray(self.x)
            self.x= np.swapaxes(self.x, 0, 1)


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

        if self.n == 1:
            idx_train = []
            idx_valid = []
            for i in range(t_max // 100):
                start = i * 100
                idx_train.extend(range(start + self.tau, start + int(100 * self.ratio_train)))
                idx_valid.extend(range(start + int(100 * self.ratio_train), start + 100))
            self.idx_train = np.array(idx_train)
            self.idx_valid = np.array(idx_valid)
            self.n_train = self.idx_train.shape[0]  #int(t_max * self.ratio_train)
            self.n_valid = self.idx_valid.shape[0]  #int(t_max * self.ratio_valid)

            # train = X first, valid = 1 - X last examples
            # self.idx_train = np.arange(self.tau, self.n_train)
            # self.idx_valid = np.arange(self.n_train, self.n_train + self.n_valid)
        else:
            self.n_train = int(self.n * self.ratio_train)
            self.n_valid = int(self.n * self.ratio_valid)
            self.idx_train = np.arange(self.tau, self.n_train)
            self.idx_valid = np.arange(self.n_train - self.tau, self.n_train + self.n_valid)
            np.random.shuffle(self.idx_train)
            np.random.shuffle(self.idx_valid)
