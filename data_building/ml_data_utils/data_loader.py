# phillipe's code
import os
import torch
import tables  # for handling the h5py files
import numpy as np
#from typing import Concatenate, Tuple
import xarray as xr
from data_generation.utils.helper_funcs import get_keys_from_value
from data_generation.parameters.constants import VAR_SOURCE_LOOKUP, RES_TO_CHUNKSIZE

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
        years: [str], #TODO: different years for different experiments? (historical??)
        experiments_train: [str], # we want to split the datasets according to experiments
        experiments_valid: [str],
        vars: [str],
        resolution="250_km",
        freq="mon",
        model="NorESM2-LM",
        ensemble_members=["r1i1p1f1", "r2i1p1f1"]
    ):
        """
        Simple DataLoader (extended from Phillipe to fit real climate model data.)
        Resulting dimension of data: (n, t, d, d_x)
        n = number of time series i.e. years
        t = numbernumber of time steps ie. frequency (12 for months...)
        d = number of physical variables
        d_x = the number of grid locations

        @params:
            ratio_train (float): ratio of the dataset to be used for training -> DEPRICATED
            ratio_valid (float): ratio of the dataset to be used for validation -> DEPRICATED
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
        self.experiments = experiments_train+experiments_valid
        self.experiments_train=experiments_train
        self.experiments_valid=experiments_valid
        self.resolution = resolution
        self.freq = freq
        self.model = model
        self.ensemble_members = ensemble_members

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
        all_black_years=[]
        self.model_data={}
        self.model_exp_to_index={}
        n=0


        if len(self.model_vars)>=1:
            for m in self.ensemble_members:
                member_data, member_exp_to_index, green_years= self._load_data_model_ensemble(m)
                self.model_data[m]=member_data
                print("member data shape", member_data.shape)
                self.model_ex_to_index[m]=member_exp_to_index
                all_black_years.append(*green_years)
                n+=member_data.shape[0]
                self.t=member_data.shape[1]


        if len(self.forcing_vars)>=1:
            self.forcing_data, self.forcing_exp_to_index=self._load_data_forcing()
            n+=self.forcing_data.shape[0]
            self.t = self.forcing_data.shape[1]

        self.n = n

        self.d=len(self.model_vars)+len(self.forcing_vars)
        self.dx=self.forcing_data.shape[-1]

        #self._load_data()
        self._split_data() #-> DEPRICATED

    def _load_data(self, member="r1i1p1f1"):

        """
        DEPRICATED
        Open and store the data files.

        @returns:
            data (np.array): Data is sorted by experiments and has the size (num_exp*years_per_exp, t, d, d_x)
            exp_to_index (Dict([str]: Tuple(int))): Dictonary storing the respective indexes of each experiment along the first dimension of the data (n)
        """

        # build correct path
        res_freq = f"/{self.resolution}/{self.freq}/"

        #n = len(self.experiments) * len(self.years)

        # num of target vars
        d_m = len(self.model_vars)

        data = []




        # 1 bacth = all years for a specific scenario
        # track available data per experiments
        data_per_exp=[]

        # dictionary to store indexes of data per experiment

        exp_to_indexes = {}

        # initial index for experiment
        start_counter=0

        # iterate through all experiments
        for  e in self.experiments:

            # flag if we have enough variables for the experiment/y pair
            enough_data=True

            # track data per year
            data_per_y=[]
            # counter for number of years with sufficient data (all variables present)
            year_counter=0

            # iterate throuhg years
            for y in self.years:

                # track data per variable
                data_per_var=[]

                # if variables are missing, skip the years
                enough_data=True


                if d_m >= 1:

                    for i, v in enumerate(self.model_vars):

                            # empty array for storing -> one array per experiment
                            #model_data = np.zeros((, t, d_m, *d_x)) we don't know in advance cause we might be missing years / vars e.g. historical

                                # create path names
                                path = f"{self.data_path}CMIP6/{self.model}/{member}/{e}/{v}{res_freq}{y}/"

                                isExist = os.path.exists(path)

                                if not isExist:
                                    print(f"WARNING: Experiment, variable, year pair does not exist: {e,v,y}. Skipping.")
                                    enough_data=False
                                    continue



                                # todo : insert is exist checkup

                                f_list = os.listdir(path)
                                if len(f_list)==0:
                                    print(f"WARNING: No data available for the pair {e, v}. Skipping.")
                                    enough_data=False
                                    continue
                                f_name = f_list[0]

                                # make daat per var stackable along the var dimension (2)
                                data_per_var.append(np.expand_dims(np.asarray(
                                    tables.open_file(path + f_name, mode="r").root._f_get_child(
                                        v
                                    )),1))





                    #  handle input4mips variables
                    if d_f >= 1:
                        #forcing_data = np.zeros((n, t, d_f, *d_x))
                        for j, v in enumerate(self.forcing_vars):

                                path = f"{self.data_path}input4mips/{e}/{v}{res_freq}{y}/"

                                isExist = os.path.exists(path)
                                if not isExist:
                                    print(f"WARNING: Experiment, variable, year pair does not exist: {e,v,y}. Skipping.")
                                    enough_data=False
                                    continue


                                f_name = os.listdir(path)[0]
                                print("appending")
                                # make data per var stackable along the var dimension (2)
                                data_per_var.append(np.expand_dims(np.asarray(
                                    tables.open_file(path + f_name, mode="r").root._f_get_child(
                                        v
                                    )
                                ),1))

                if enough_data:
                        print("HAving enough data")
                        # concatenate over variables
                        array_vars=np.concatenate(data_per_var, axis=1)
                        # make stackable along num_time series dimension
                        array_vars=np.expand_dims(array_vars, 0)

                        data_per_y.append(array_vars)
                        year_counter+=1
                else:
                        print(f"WARNING: Not all vars available for the specified year: {y}. Skipping.")
                        continue

            if len(data_per_y)==0:
                print(f"WARNING: No data for year {y}. Skipping.")
                continue
            else:
                array_years=np.concatenate(data_per_y,0)

                data_per_exp.append(array_years)
                exp_to_indexes[e]=(start_counter, start_counter+year_counter)
                start_counter=start_counter+year_counter

        # in the end the whole dataset should be indexable by experiment / num years
        # so wee need a mapping for experiment to indexes e.g "ssp126" -> index 0-5 (5 years), "ssp370" = index 5-15
        #data.append(forcing_data)


        self.exp_to_indexes=exp_to_indexes
        # combine all variables
        if len(data_per_exp)==0:
            print("WARNING: No data available. Please check your specifications and make sure you have all variables for all years specfiied.")
            raise ValueError
        data = np.concatenate(data_per_exp)

        # read out dimensions
        n = data.shape[0]
        t = .data.shape[1]
        d = .data.shape[2]
        # flatten out grid cells
        data = np.reshape(data, (n, t, d, -1))
        d_x = data.shape[-1]


        return data, exp_to_indexes

    def _load_data_forcing(self, years):
        """
        Open and store the data files.

        @returns:
            data (np.array): Data is sorted by experiments and has the size (num_exp*years_per_exp, t, d, d_x)
            exp_to_index (Dict([str]: Tuple(int))): Dictonary storing the respective indexes of each experiment along the first dimension of the data (n)
        """

        # build correct path
        res_freq = f"/{self.resolution}/{self.freq}/"


        data = []


        # 1 bacth = all years for a specific scenario
        # track available data per experiments
        data_per_exp=[]

        # dictionary to store indexes of data per experiment

        exp_to_indexes = {}

        # initial index for experiment
        start_counter=0

        # iterate through all experiments
        for  e in self.experiments:

            # flag if we have enough variables for the experiment/y pair
            enough_data=True

            # track data per year
            data_per_y=[]
            # counter for number of years with sufficient data (all variables present)
            year_counter=0

            # iterate throuhg years
            for y in self.years:

                # track data per variable
                data_per_var=[]

                # if variables are missing, skip the years
                enough_data=True


                #forcing_data = np.zeros((n, t, d_f, *d_x))
                for j, v in enumerate(self.forcing_vars):

                                path = f"{self.data_path}input4mips/{e}/{v}{res_freq}{y}/"

                                isExist = os.path.exists(path)
                                if not isExist:
                                    print(f"WARNING: Experiment, variable, year pair does not exist: {e,v,y}. Skipping.")
                                    enough_data=False
                                    continue


                                f_name = os.listdir(path)[0]
                                print("appending")
                                # make data per var stackable along the var dimension (2)
                                data_per_var.append(np.expand_dims(np.asarray(
                                    tables.open_file(path + f_name, mode="r").root._f_get_child(
                                        v
                                    )
                                ),1))

                if enough_data:
                        print("HAving enough data")
                        # concatenate over variables
                        array_vars=np.concatenate(data_per_var, axis=1)
                        # make stackable along num_time series dimension
                        array_vars=np.expand_dims(array_vars, 0)

                        data_per_y.append(array_vars)
                        year_counter+=1
                else:
                        print(f"WARNING: Not all vars available for the specified year: {y}. Skipping.")
                        continue

            if len(data_per_y)==0:
                print(f"WARNING: No data for year {y}. Skipping.")
                continue
            else:
                array_years=np.concatenate(data_per_y,0)

                data_per_exp.append(array_years)
                exp_to_indexes[e]=(start_counter, start_counter+year_counter)
                start_counter=start_counter+year_counter

        # in the end the whole dataset should be indexable by experiment / num years
        # so wee need a mapping for experiment to indexes e.g "ssp126" -> index 0-5 (5 years), "ssp370" = index 5-15
        #data.append(forcing_data)


        exp_to_indexes=exp_to_indexes
        # combine all variables
        if len(data_per_exp)==0:
            print("WARNING: No data available. Please check your specifications and make sure you have all variables for all years specfiied.")
            raise ValueError
        data = np.concatenate(data_per_exp)

        # read out dimensions
        n = data.shape[0]
        t = data.shape[1]
        d = data.shape[2]
        # flatten out grid cells
        data = np.reshape(data, (n, t, d, -1))
        d_x = data.shape[-1]


        return data, exp_to_indexes



    def _load_data_model_ensemble(self, member):
        """
        Open and store the data files.

        @returns:
            data (np.array): Data is sorted by experiments and has the size (num_exp*years_per_exp, t, d, d_x)
            exp_to_index (Dict([str]: Tuple(int))): Dictonary storing the respective indexes of each experiment along the first dimension of the data (n)
        """

        # build correct path
        res_freq = f"/{self.resolution}/{self.freq}/"

        #n = len(self.experiments) * len(self.years)

        # num of target vars
        d_m = len(self.model_vars)

        t = RES_TO_CHUNKSIZE[self.freq]


        data = []

        # 1 bacth = all years for a specific scenario
        # track available data per experiments
        data_per_exp=[]

        # dictionary to store indexes of data per experiment

        exp_to_indexes = {}

        # initial index for experiment
        start_counter=0

        # iterate through all experiments
        for  e in self.experiments:

            # flag if we have enough variables for the experiment/y pair
            enough_data=True

            # track data per year
            data_per_y=[]
            # counter for number of years with sufficient data (all variables present)
            year_counter=0

            # iterate throuhg years
            for y in self.years:

                # track data per variable
                data_per_var=[]

                # if variables are missing, skip the years
                #enough_data=True


                if d_m >= 1:

                    for i, v in enumerate(self.model_vars):

                            # empty array for storing -> one array per experiment
                            #model_data = np.zeros((, t, d_m, *d_x)) we don't know in advance cause we might be missing years / vars e.g. historical

                                # create path names
                                path = f"{self.data_path}CMIP6/{self.model}/{member}/{e}/{v}{res_freq}{y}/"

                                isExist = os.path.exists(path)

                                if not isExist:
                                    print(f"WARNING: Experiment, variable, year pair does not exist: {e,v,y}. Skipping.")
                                    enough_data=False
                                    continue



                                # todo : insert is exist checkup

                                f_list = os.listdir(path)
                                if len(f_list)==0:
                                    print(f"WARNING: No data available for the pair {e, v}. Skipping.")
                                    enough_data=False
                                    continue
                                f_name = f_list[0]

                                # make daat per var stackable along the var dimension (2)
                                data_per_var.append(np.expand_dims(np.asarray(
                                    tables.open_file(path + f_name, mode="r").root._f_get_child(
                                        v
                                    )),1))

                if enough_data:
                        # concatenate over variables
                        array_vars=np.concatenate(data_per_var, axis=1)
                        # make stackable along num_time series dimension
                        array_vars=np.expand_dims(array_vars, 0)

                        data_per_y.append(array_vars)
                        year_counter+=1
                else:
                        print(f"WARNING: Not all vars available for the specified year: {y}. Skipping.")
                        continue

            if len(data_per_y)==0:
                print(f"WARNING: No data for year {y}. Skipping.")
                continue
            else:
                array_years=np.concatenate(data_per_y,0)

                data_per_exp.append(array_years)
                exp_to_indexes[e]=(start_counter, start_counter+year_counter)
                start_counter=start_counter+year_counter

        # in the end the whole dataset should be indexable by experiment / num years
        # so wee need a mapping for experiment to indexes e.g "ssp126" -> index 0-5 (5 years), "ssp370" = index 5-15
        #data.append(forcing_data)


        self.exp_to_indexes=exp_to_indexes
        # combine all variables
        if len(data_per_exp)==0:
            print("WARNING: No data available. Please check your specifications and make sure you have all variables for all years specfiied.")
            raise ValueError
        self.data = np.concatenate(data_per_exp)

        # read out dimensions
        n = self.data.shape[0]
        t = self.data.shape[1]
        d = self.data.shape[2]
        # flatten out grid cells
        self.data = np.reshape(self.data, (n, t, d, -1))

        return data, exp_to_indexes

        """
        OLD CODE save
         # handle cmip6 target variables

        if d_m >= 1:
            model_data = np.zeros((n, t, d_m, *d_x))

            for i, v in enumerate(self.model_vars):


                for j, e in enumerate(self.experiments):

                    # empty array for storing -> one array per experiment sot

                    for k, y in enumerate(self.years):
                        # create path names
                        path = f"{self.data_path}CMIP6/{e}/{v}{res_freq}{y}/"
                        print(path)
                        f_name = os.listdir(path)[0]

                        model_data[j + k, :, i, :] = np.asarray(
                            tables.open_file(path + f_name, mode="r").root._f_get_child(
                                v
                            )
                        )
                        data_per_exp.append()

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
        print("DATA shape", self.data.shape)"""

        # use coordinates if using real-world datasets
        # if self.no_gt:
        #    self.coordinates = np.load(os.path.join(self.data_path, 'coordinates.npy'))
        #    print("got coords")
        #    # TODO: decide if keep or not
        #    # self.distances = self.get_geodisic_distances(self.coordinates)

    def _split_data(self):
        """ DEPRICATED
        Determine the indices for training and validation sets.


        In our case, we should split along the experiments.....
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

    def sample(self, batch_size=None, drop_remainder=False, valid=bool):
        """
        Sampling sliced time series from the train / validation dataset.

        @params:
            batch_size (int): if batch size is none, the batch_size equals the number of years available for each experiment
            valid (bool): if True, sample from validation dataset
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

    def sample_per_experiment_member(self, batch_size: int, valid: bool = False):
        """
        Sampling sliced time series from the trainin / validation dataset.
        Returns a batch for each experiment / ensembl member pair present in the respective dataset.

        @params:
            batch_size (int): the number of examples in a minibatch
            valid (bool): if True, sample from validation set
        @returns:
            x (tensor): batch of the data with time-window (num_experiments, batch_size, tau, d, d_x)
            y (tensor): data to predict (num_experiments, batch_size, d, d_x)

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
            experiments=self.experiments_valid
        else:
            experiments=self.experiments_train

        num_exp=len(experiments)

        full_x=np.zeros((num_exp, batch_size, self.tau, self.d, self.d_x))
        full_y=np.zeros((num_exp, batch_size, self.d, self.d_x))

        for i,exp in enumerate(experiments):
            for j, m in enumerate(self.ensemble_members):
                have_model_data=False
                have_forcing_data=False
                # cmip data
                if len(self.model_vars)>=1:

                    try:
                        start, end = self.model_exp_to_index[m][exp]
                    except KeyError:
                        aspect="validation" if valid else "training"
                        print(f"WARNING: Experiment {exp} member r{m} pair not existent in {aspect} dataset.")
                        continue

                    model_data_per_batch=self.model_data[m][start:end, :]
                    model_dataset_index=np.arrange(start, end)
                    model_n=start-end
                    have_model_data=True

                if len(self.forcing_vars)>=1:

                    try:
                        start, end = self.forcing_exp_to_index[exp]
                    except KeyError:
                        aspect="validation" if valid else "training"
                        print(f"WARNING: Experiment {exp} not existent in forcing {aspect} dataset.")
                        continue

                    forcing_data_per_batch=self.forcing_data[start:end, :]
                    forcing_dataset_index=np.arrange(start, end)
                    forcing_n=start-end
                    have_forcing_data=True


                # input4 mips data
                try:
                    start, end = self.exp_to_indexes[exp]
                except KeyError:
                    aspect="validation" if valid else "training"

                    print(f"WARNING: Experiment {exp} not existent in {aspect} dataset.")

                    if ((i+1)==len(experiments)) & (len(data_per_batch)==0):
                        print("WARNING: No data found, please confirm your experiment selection. Aborting.")
                        raise exit(0)
                    continue

                data_per_batch = self.data[start:end, :]
                dataset_idx = np.arange(start, end)
                n=start-end
                # single time-series (one year), select random index along the time dimension
                if n==1:
                    random_idx=np.random.choice(dataset_idx, replace=True, size=batch_size)
                    for i, idx in enumerate(random_idx):
                        # first dimension is 1
                        x[i, :] = np.squeeze(self.data_per_batch)[idx - self.tau : idx + t1, :, :]
                        y[i, :] = np.squeeze(self.data_per_batch)[idx + t1]
                # multiple time series
                else:
                    # with replacement cause bigger batch_size than number of time series
                    random_idx = np.random.choice(dataset_idx, replace=True, size=batch_size)
                    for i, idx in enumerate(random_idx):
                        x[i] = self.data[idx, 0 : self.tau + t1]
                        y[i] = self.data[idx, self.tau + t1]
                full_x[i:]=x
                full_y[i:]=y
        # convert to torch tensors
        full_x_ = torch.tensor(full_x)
        full_y_ = torch.tensor(full_y)

        return full_x_, full_y_

    def sample_per_experiment(self, batch_size: int, valid: bool = False):
        """
        Sampling sliced time series from the trainin / validation dataset.
        Returns a batch for each experiment present in the respective dataset.

        @params:
            batch_size (int): the number of examples in a minibatch
            valid (bool): if True, sample from validation set
        @returns:
            x (tensor): batch of the data with time-window (num_experiments, batch_size, tau, d, d_x)
            y (tensor): data to predict (num_experiments, batch_size, d, d_x)

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
            experiments=self.experiments_valid
        else:
            experiments=self.experiments_train

        num_exp=len(experiments)

        full_x=np.zeros((num_exp, batch_size, self.tau, self.d, self.d_x))
        full_y=np.zeros((num_exp, batch_size, self.d, self.d_x))

        for i,exp in enumerate(experiments):
            try:
                start, end = self.exp_to_indexes[exp]
            except KeyError:
                aspect="validation" if valid else "training"

                print(f"WARNING: Experiment {exp} not existent in {aspect} dataset.")

                if ((i+1)==len(experiments)) & (len(data_per_batch)==0):
                    print("WARNING: No data found, please confirm your experiment selection. Aborting.")
                    raise exit(0)
                continue

            data_per_batch = self.data[start:end, :]
            dataset_idx = np.arange(start, end)
            n=start-end
            # single time-series (one year), select random index along the time dimension
            if n==1:
                random_idx=np.random.choice(dataset_idx, replace=True, size=batch_size)
                for i, idx in enumerate(random_idx):
                    # first dimension is 1
                    x[i, :] = np.squeeze(self.data_per_batch)[idx - self.tau : idx + t1, :, :]
                    y[i, :] = np.squeeze(self.data_per_batch)[idx + t1]
            # multiple time series
            else:
                # with replacement cause bigger batch_size than number of time series
                random_idx = np.random.choice(dataset_idx, replace=True, size=batch_size)
                for i, idx in enumerate(random_idx):
                    x[i] = self.data[idx, 0 : self.tau + t1]
                    y[i] = self.data[idx, self.tau + t1]
            full_x[i:]=x
            full_y[i:]=y
        # convert to torch tensors
        full_x_ = torch.tensor(full_x)
        full_y_ = torch.tensor(full_y)

        return full_x_, full_y_



if __name__ == "__main__":
    ratio_train = 0.8
    ratio_valid = 0.2
    data_path = "/home/charlie/Documents/MILA/causalpaca/data/data_processed/"  # /home/charlie/Documents/MILA/causalpaca/data/test_data/"
    years = [str(y) for y in np.arange(2015, 2056,5)]
    print(years)  # ,"2020"]
    experiments_train = ["ssp126"]  # , "ssp370"]
    experiments_valid= ["ssp126"]
    vars = ["pr", "tas"]#, "BC_em_anthro", "CH4_em_anthro", "CO2_em_anthro", "SO2_em_anthro"]

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
        experiments_train=experiments_train,
        experiments_valid=experiments_valid,
        vars=vars,

    )

    x, y = data_loader.sample_per_experiment(8)

    print(x.size(), y.size())

    x,y = data_loader.sample(8)

    print(x.shape, y.shape)
