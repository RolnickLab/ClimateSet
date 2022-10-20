# phillipe's code

import os
import torch
import tables  # for handling the h5py files
import numpy as np
#from typing import Concatenate, Tuple
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
        instantaneous: bool,  # Use instantaneous connections
        tau: int,  # time window size
        years: [str], #TODO: different years for different experiments? (historical??)
        experiments_train: [str], # we want to split the datasets according to experiments
        experiments_valid: [str],
        vars: [str],
        resolution="250_km",
        freq="mon",
        model="NorESM2-LM",
        ensemble_members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
        load_mix=True,

    ):
        """
        Simple DataLoader.
        Loads forcnig and model data separatley for storing efficiency, except when load_mix is set to True: In this case, data is thrown together to a single dataset that allows sampling with mixing experiments and ensemble members.
        
            n = number of time series i.e. years
            t = numbernumber of time steps ie. frequency (12 for months...)
            d = number of physical variables
            d_x = the number of grid locations
            num_exp= number of available experiments
            num_ensemble = number of available ensemble members
    
        Resulting dimension of data: 
        if load_mix: 
            data: np.array(shape=(num_exp x num_ensemble x n, t, d, d_x)) 
        
        else:
            2 parts of the dataset:
            1)CMIP6
                model_data : Dict[member]=np.array(shape=(num_exp x num_enesmble x n, t, model_vars, d_x))
                model_exp_to_index: Dict[member]=Dict[experiment]=(start, end) indexses for data indexing
            2) input4mips
                forcing_data: np.array(shape=(num_exp x n, t, forcing_vars, d_x))
                forcing_exp_to_index: Dict[experiment]=(start,end)

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
            model (str): name of model used in trainnig # TODO: extend to multiple models
            ensemble_members ([str]): list of ensemble members to include
            load_mix (bool): if True, ensemble member and experiments are mixed in sampling
        """

        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        self.data_path = data_path
        self.latent = latent #TODO: not yet handled
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
        self.load_mix = load_mix



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
        
        self.model_data={}
        self.model_exp_to_index={}
        n=0
        consider_forcing=(len(self.forcing_vars)>=1)
        

        self.data=[]

        all_avail_years=[]
        if consider_forcing:
            self.forcing_data, self.forcing_exp_to_index, avail_years =self._load_data_forcing()
            n+=self.forcing_data.shape[0]
            self.t = self.forcing_data.shape[1]
            self.d_x=self.forcing_data.shape[-1]
            all_avail_years=list(set(all_avail_years+avail_years))
        else:
            all_avail_years=self.years

        
        if len(self.model_vars)>=1:
            for m in self.ensemble_members:
                member_data, member_exp_to_index = self._load_data_model_ensemble(m, all_avail_years)
                if member_data is None:
                    continue
                self.model_data[m]=member_data
                self.model_exp_to_index[m]=member_exp_to_index
                n+=member_data.shape[0]
                self.t=member_data.shape[1]

                if consider_forcing:
                    assert self.d_x==member_data.shape[-1], "WARNING: Spatial dimensions of forcing and member data does not match!"
                else:
                    self.d_x=member_data.shape[-1]


                if load_mix:
                    if consider_forcing:
                        for e in self.experiments:
                            if e in (self.forcing_exp_to_index.keys() & member_exp_to_index.keys()):
                                m_start, m_end = member_exp_to_index[e]
                                f_start, f_end = self.forcing_exp_to_index[e]
                                self.data.append(np.stack([self.forcing_data[f_start:f_end], member_data[m_start:m_end]], axis=2))
                    else:
                        self.data.append(member_data)

        self.d=len(self.model_vars)+len(self.forcing_vars)
        if load_mix:
            self.data=np.concatenate(self.data, 0)
            
            self.n = self.data.shape[0]
            self.d = self.data.shape[2]
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
        t = data.shape[1]
        d = data.shape[2]
        # flatten out grid cells
        data = np.reshape(data, (n, t, d, -1))
        d_x = data.shape[-1]
    

        return data, exp_to_indexes

    def _load_data_forcing(self):
        """
        Open and store the data files for forcing variables.
        
        @returns:
            data (np.array): Data is sorted by experiments and has the size (num_exp*years_per_exp, t, d, d_x)
            exp_to_index (Dict([str]: Tuple(int))): Dictonary storing the respective indexes of each experiment along the first dimension of the data (n)
        """

        # build correct path
        res_freq = f"/{self.resolution}/{self.freq}/"

        avail_years=[] 
    
        # 1 bacth = all years for a specific scenario
        # track available data per experiments             
        data_per_exp=[]

        # dictionary to store indexes of data per experiment

        exp_to_indexes = {}

        # initial index for experiment
        start_counter=0

        # track available years per experiment
        exp_avail_years=[]

        # iterate through all experiments
        for  e in self.experiments:

            # flag if we have enough variables for the experiment/y pair
            enough_data=True
    
            # track data per year
            data_per_y=[]
            # counter for number of years with sufficient data (all variables present)
            year_counter=0
            
            # iterate through years
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
                                    print(path)
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
                        exp_avail_years.append(y)
                else: 
                        print(f"WARNING: Not all vars available for the specified year: {y}. Skipping.")
                        continue

            if len(data_per_y)==0:
                print(f"WARNING: No data for year {y}. Skipping.")
                continue
            else:
                array_years=np.concatenate(data_per_y,0)
             
                 # in the end the whole dataset should be indexable by experiment / num years
            # so wee need a mapping for experiment to indexes e.g "ssp126" -> index 0-5 (5 years), "ssp370" = index 5-15
        
                data_per_exp.append(array_years)
                exp_to_indexes[e]=(start_counter, start_counter+year_counter)
                start_counter=start_counter+year_counter
                avail_years = list(set(avail_years+exp_avail_years))

       
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
        

        return data, exp_to_indexes, avail_years

    def _load_data_model_ensemble(self, member, years):
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

          
            # track data per year
            data_per_y=[]
            # counter for number of years with sufficient data (all variables present)
            year_counter=0
            
            # iterate throuhg years
            for y in years:
            
                # track data per variable 
                data_per_var=[]

                # if variables are missing, skip the years
                #enough_data=True
            
                # flag if we have enough variables for the experiment/y pair
                enough_data=True
    
                if d_m >= 1:

                    for i, v in enumerate(self.model_vars):

                            # empty array for storing -> one array per experiment
                            #model_data = np.zeros((, t, d_m, *d_x)) we don't know in advance cause we might be missing years / vars e.g. historical
                        
                                # create path names
                                path = f"{self.data_path}CMIP6/{self.model}/{member}/{e}/{v}{res_freq}{y}/"
                                print(path)
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

            
        exp_to_indexes=exp_to_indexes
        # combine all variables
        if len(data_per_exp)==0:
            print(f"WARNING: No data available for member {member}. Please check your specifications and make sure you have all variables for all years specfiied.")
            return None, None
        data = np.concatenate(data_per_exp)
      
        # read out dimensions
        n = data.shape[0]
        t = data.shape[1]
        d = data.shape[2]
        # flatten out grid cells
        data = np.reshape(data, (n, t, d, -1))
   
      
        return data, exp_to_indexes
     

    def _split_data(self):
        """ 
        Determine the indices for training and validation sets. Only used if load_mix is True.
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

    def sample_mix(self, batch_size=None, valid=bool):
        """
        Sampling sliced time series from the train / validation dataset.
        This method mixes different experiments and ensemble members in each batch. 

        @params:
            batch_size (int): if batch size is none, the batch_size equals the number of years available for each experiment
            valid (bool): if True, sample from validation dataset
        @returns:
            x (tensor): of the data with time-series (batch_size, tau, d, d_x)
            y (tensor): data to predict (batch_size, d, d_x) 

        """

        assert self.load_mix, "You want to sample mixed data but due to storage efficiency it is not yet loaded. Please initialize the data_loader with load_mix=True to load mixed data."

        # initiliaze empty arrays
        if (
            self.instantaneous
        ):  # if we have instantaneous connections consider x_{t} .. x_{t-i}
            x = np.zeros((batch_size, self.tau + 1, self.d, self.d_x))
            t1 = 1  #  modifier

        else:  # only consider x_{t-1} ... x_{t-i}
            x = np.zeros((batch_size, self.tau, self.d, self.d_x))
            t1 = 0       

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

        return x_, y_

    def sample(self, batch_size: int, valid: bool = False):
        """
        Sampling sliced time series from the training / validation dataset.
        Returns a batch for each experiment / ensemble member pair present in the respective dataset.
        Each batch contains a sample over the present years contained in the dataset, years may be repeated dependent on the choice of batch_size and the number of available years.
        If only a single years exist per experiment / member pairs, sampling happens along the time axis.


        @params:
            batch_size (int): the number of examples in a minibatch
            valid (bool): if True, sample from validation set
        @returns:
            x (tensor): batch of the data with time-window (num_experimentsxnum_ensemble members, batch_size, tau, d, d_x), please note that only fully complete experiments/semble members pairs will be considered
            y (tensor): data to predict (num_experimentsxnum_ensemble_members, batch_size, d, d_x), please note that only fully complete experiment/ensemble member pairs will be considered

        """

        # initiliaze empty arrays
        if (
            self.instantaneous
        ):  # if we have instantaneous connections consider x_{t} .. x_{t-i}
            x = np.zeros((batch_size, self.tau + 1, self.d, self.d_x))
            t1 = 1  #  modifier


        else:  # only consider x_{t-1} ... x_{t-i}
            x = np.zeros((batch_size, self.tau, self.d, self.d_x))
            t1 = 0
        
        # targets without time-windows
        y = np.zeros((batch_size, self.d, self.d_x))

        if valid:
            experiments=self.experiments_valid
        else:
            experiments=self.experiments_train

        # list tracking all samples
        full_x=[]
        full_y=[]

        # list tracking samples per experiment
        full_x_exp=[]
        full_y_exp=[]

        for i,exp in enumerate(experiments):
                
                have_forcing_data=False
                x_forcing, y_forcing = x.copy(), y.copy()
                forcing_n = 0

                # if we have forcing variables, sample forcing data only once per experiment

                if len(self.forcing_vars)>=1:

                    # ckeck existence
                    try:
                        # retrieve index given experiment
                        start, end = self.forcing_exp_to_index[exp]
                    except KeyError:
                        aspect="validation" if valid else "training"
                        print(f"WARNING: Experiment {exp} not existent in forcing {aspect} dataset.")
                        continue

                    # extract experiment data from all forcing data
                    forcing_data_per_batch=self.forcing_data[start:end, :]
                    
                    forcing_n=end-start
                    have_forcing_data=True
                

                    if not(forcing_data_per_batch.shape[-1]!=self.d_x):
                        print(f"WARNING: Spatial resolution for forcing data do not match, got {forcing_data_per_batch.shape[-1]} but should be {self.d_x}. Skiping.")
                        continue
                    
                 
                    # one big time series = sample along time dimension
                    if forcing_n==1:
                        random_idx_forcing=np.random.choice(forcing_data_per_batch.shape[1], replace=True, size=batch_size)
                        
                        for i,jdx in enumerate(zip(random_idx_forcing)):
                            x_forcing=np.squeez(forcing_data_per_batch)[jdx-self.tau:jdx+t1, :, :]
                            y_forcing=np.squeeze(forcing_data_per_batch)[jdx + t1]

                    # multiple time series (e.g. several years), sample over years
                    elif forcing_n>1:
                        random_idx_forcing = np.random.choice(forcing_data_per_batch.shape[0], replace=True, size=batch_size)
                        
                        for i, idx in enumerate(random_idx_forcing):
                            x_forcing[i] = forcing_data_per_batch[idx, 0 : self.tau + t1]
                            y_forcing[i] = forcing_data_per_batch[idx, self.tau + t1]

                    
                # given that we have model data 
                if len(self.model_vars)>=1:
                    
                    # sample model data for each ensemleb member
                    for  m in self.ensemble_members:

                        have_model_data=False
                        model_n=0
                        x_model, y_model = x.copy(), y.copy()
                  
                        # trying to retreive indexes for member experiment pair
                        try:
                            start, end = self.model_exp_to_index[m][exp]
                        except KeyError:
                            aspect="validation" if valid else "training"
                            print(f"WARNING: Experiment {exp} member r{m} pair not existent in {aspect} dataset.")
                            continue

                        # get data per experiment / member
                        model_data_per_batch=self.model_data[m][start:end, :]
                        
                        model_n=end-start
                        have_model_data=True
                       
                        if model_data_per_batch.shape[-1]!=self.d_x:
                            print(f"WARNING: Spatial resolution for model data {exp} {m} do not match, got {model_data_per_batch.shape[-1]} but should be {self.d_x}. Skiping.")
                            continue

                        # one big time series = sample along time dimension
                        if model_n==1:
                            random_idx_model=np.random.choice(model_data_per_batch.shape[1], replace=True, size=batch_size)
                            for i,jdx in enumerate(zip(random_idx_model)):
                                x_model=np.squeez(model_data_per_batch)[jdx-self.tau:jdx+t1, :, :]
                                y_model=np.squeeze(model_data_per_batch)[jdx + t1]
                        # multiple time series (e.g. several years), sample over years
                        elif model_n>1:
                            random_idx_model = np.random.choice(model_data_per_batch.shape[0], replace=True, size=batch_size)
                            for i, idx in enumerate(random_idx_model):
                                x_model[i] = model_data_per_batch[idx, 0 : self.tau + t1]
                                y_model[i] = model_data_per_batch[idx, self.tau + t1]

                      

                        if (have_model_data & have_forcing_data):
                            if not(model_n==forcing_n):
                                print(f"WARNING: Not the same number of samples experiment {exp} and member {m} for forcings and model variables. Skipping ")
                                continue
                            # join together along data var dimension
                            x_member_exp=np.concatenate([x_forcing, x_model], axis=2)
                            y_member_exp=np.concatenate([y_forcing, y_model], axis=2)

                        elif have_forcing_data:
                            x_member_exp=x_forcing
                            y_member_exp=y_forcing
                        elif have_model_data:
                            x_member_exp=x_model
                            y_member_exp=y_model
                        else:
                            print(f"WARNING: No data for experiment {exp} and member {m} for forcings and model variables. Skipping.")
                            continue

                        # track data per member
                        full_x_exp.append(x_member_exp)
                        full_y_exp.append(y_member_exp)

                # give that we have no model data
                else:
                    full_x_exp.append(x_forcing)
                    full_y_exp.append(y_forcing)

                # track data per experiment
               
                full_x_exp_a=np.stack(full_x_exp, axis=0)
                full_y_exp_a=np.stack(full_y_exp, axis=0)
               
                full_x.append(full_x_exp_a)
                full_y.append(full_y_exp_a)


        if (len(full_x)==0) or (len(full_y)==0):
                    print(f"WARNING: Apperantly no data available that fits the specifications. Please check your dataset and specifications.")
                    exit(0)

        # join all samples together
        full_x=np.concatenate(full_x, axis=0)
        full_y=np.concatenate(full_y, axis=0)
        x_, y_ = torch.tensor(full_x), torch.tensor(full_y)

        return x_,y_
                                   

    def sample_per_experiment(self, batch_size: int, valid: bool = False):
        """
        DEPRICATED 
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
    years = [str(y) for y in np.arange(2020, 2051,10)]
    print(years)  # ,"2020"]
    experiments_train = [ "ssp370", "ssp126"]  # , "ssp370"]
    experiments_valid= ["ssp126"]
    vars = ["pr"] #, "CH4_em_anthro", "CO2_em_anthro", "SO2_em_anthro"]

    data_loader = DataLoader(
        ratio_train,
        ratio_valid,
        data_path,
        latent=True,
        instantaneous=False,
        tau=4,
        years=years,
        experiments_train=experiments_train,
        experiments_valid=experiments_valid,
        vars=vars,
        load_mix=True
       
    )
    x, y = data_loader.sample(16)

    #x, y = data_loader.sample_per_experiment(8)

    print(x.size(), y.size())

    x,y = data_loader.sample_mix(8)

    print(x.shape, y.shape)
