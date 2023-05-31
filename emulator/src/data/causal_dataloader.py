import copy
import logging
import os
import glob
import pickle
import shutil
import zipfile
from typing import Dict, Optional, List, Callable, Tuple, Union

import numpy as np
import xarray as xr
import torch
from torch import Tensor


class CausalDataset(torch.utils.data.dataloader):
    """ 
    
    
    Use first ensemble member for now 
    Option to use multile ensemble member later
    Give option for which variable to use
    Load 3 scenarios for train/val: Take this as a list
    Process and save this as .npz in $SLURM_TMPDIR
    Load these in train/val/test Dataloader functions
    Keep one scenario for testing

    # Input shape (85 * 12, 4, 144, 96)
    # Target shape (85 * 12, 1, 144, 96)
    """
    def __init__(
            self,
            years: str, 
            mode: str = "train", # Train or test maybe
            data_dir: Optional[str] = '/home/venka97/scratch/causalpaca/data/CMIP6/',
            output_save_dir: Optional[str] = '/home/venka97/scratch/causal_savedata',
            climate_model: Union[str, List[str]] = 'NorESM2-LM',
            num_ensembles: int = 1, # 1 for first ensemble, -1 for all
            train_experiments: List[str] = ['ssp126','ssp370','ssp585'],
            test_experiments: Union[str, List[str]] = ['ssp245'],
            variables: Union[str, List[str]] = 'pr',
            load_data_into_mem: bool = True, # Keeping this true be default for now
    ):

        self.mode = mode
        self.root_dir = data_dir
        self.output_save_dir = output_save_dir
        self.input_nc_files = []
        self.output_nc_files = []

        #TO-DO: This is just getting the list of .nc files for targets. Put this logic in a function and get input list as well.
        # In a function, we can call CausalDataset() instance for train and test separately to load the data

        if isinstance(climate_model, str):
            self.root_dir = os.path.join(self.root_dir, climate_model)
        else:
            # Logic for multiple climate models, not sure how to load/create dataset yet
            pass

        if num_ensembles == 1:
            ensembles = os.listdir(self.root_dir)
            self.ensemble_dir =  os.path.join(self.root_dir, ensembles[0]) # Taking first ensemble member
        
        # Add code here for adding files for input nc data
        # Similar to the loop below for output files

        # List of output files
        for exp in train_experiments:
            for var in variables:
                var_dir = os.path.join(self.ensemble_dir, exp, variables, '250_km/mon')
                files = glob.glob(var_dir + '/**/*.nc', recursive=True)
                self.output_nc_files += files

        # Got all the files paths at this point, now open and merge


        if isinstance(year, int): 
            years = years
        else:
            years = self.get_years_list(years, give_list=True) # Can use this to split data into train/val eg. 2015-2080 train. 2080-2100 val.
        self.n_years = len(years)

        # Split the data here using n_years if needed,
        # else do random split logic here


        # Check here if os.path.isfile($SCRATCH/data.npz) exists
        # if it does, use self._reload data(path)
        if os.path.isfile():
            pass

        else:
            self.raw_data_input = self.load_data_into_mem(self.input_nc_files) #currently don't have input paths etc
            self.raw_data_output = self.load_data_into_mem(self.output_nc_files) 

            self.input_path = self.save_data_into_disk(self.raw_data_input, self.mode, 'input')
            self.output_path = self.save_data_into_disk(self.raw_data_output, self.mode, 'output')

            self.copy_to_slurm(input_path)
            self.copy_to_slurm(output_path)

            # Call _reload_data here with self.input_path and self.output_path
            # self.X = self._reload_data(input_path)
            # self.Y = self._reload_data(output_path)
            # Write a normalize transform to calculate mean and std
            # Either normalized whole array here or per instance getitem, that maybe faster

            # Now X and Y is ready for getitem

        def load_data_into_mem(self, paths: List[str]): #-> np.ndarray():
            temp_data = xr.open_mfdataset(paths).compute() #.compute is not necessary but eh, doesn't hurt
            temp_data = temp_data.to_array().to_numpy() # Should be of shape (1, 1036, 96, 144)
            temp_data = temp_data.squeeze() # (1036, 96, 144)
            temp_data = temp_data.reshape(-1, 12, 96, 144) # Don't hardcode this shape here later, get dims from CONSTANTS etc.
            return temp_data # (86, 12, 96, 144). Desired shape where 86 can be the batch dimension. Can get items of shape (batch_size, 12, 96, 144)


        def save_data_into_disk(self, data:np.ndarray, mode: str, file: str) -> str:
            fname = '_' + mode + '_' + file + '.npz'
            np.savez(os.path.join(self.output_save_dir, fname))
            return os.path.join(self.output_save_dir, fname) 


        def copy_to_slurm(self, fname):
            pass
            # Need to re-write this depending on which directory structure we want

            # if 'SLURM_TMPDIR' in os.environ:
            #     print('Copying the datato SLURM_TMPDIR')

            #     input_dir = os.environ['SLURM_TMPDIR'] + '/input'
            #     os.makedirs(os.path.dirname(in_dir), exist_ok=True)

            #     shutil.copyfile(self.input_fname, input_dir)
            #     self.input_path = h5_path_new_in

            #     h5_path_new_out = os.environ['SLURM_TMPDIR'] + '/output_' + self._filename
            #     shutil.copyfile(self._out_path, h5_path_new_out)
            #     self._out_path = h5_path_new_out

        def _reload_data(self, fname):
            try:
                in_data = np.load(fname, allow_pickle=True)
            except zipfile.BadZipFile as e:
                log.warning(f"{fname} was not properly saved or has been corrupted.")
                raise e
            try:
                in_files = in_data.files
            except AttributeError:
                return in_data

            if len(in_files) == 1:
                return in_data[in_files[0]]
            else:
                return {k: in_data[k] for k in in_files}


        def get_years_list(self, years:str, give_list: Optional[bool] = False):
            """
            Get a string of type 20xx-21xx.
            Split by - and return min and max years.
            Can be used to split train and val.
            """
            assert len(years) != 9, "Years string must be in the format xxxx-yyyy eg. 2015-2100 with string length 9. Please check the year string."
            splits = years.split('-')
            min_year, max_year = int(splits[0]), int(splits[1])

            if give_list:
                return np.arange(min_year, max_year + 1, step=1) 
            return min_year, max_year


        def get_dataset_statistics(self, data):
            pass
        # Load the saved data here and calculate the mean & std for normalizing the input
        # Return and normalized before __getitem__()
        
        def __getitem__(self, index) -> (Dict[str, Tensor], Dict[str, Tensor]):  # Dict[str, Tensor]):
            if not self._load_h5_into_mem:
                pass
                # X = raw_Xs
                # Y = raw_Ys
            else:
                #TO-DO: Need to write Normalizer transform and To-Tensor transform
                # Doing norm and to-tensor per-instance here. 
                X_norm = self.input_transforms(self.X[index]) 
                Y_norm = self.output_transforms(self.Y[index])

            sample = {"input": X_norm, "label": Y_norm}

            return sample

        # @property
        # def name(self):
        #     return self._name.upper()

        def __str__(self):
            s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
            return s

        def __len__(self):
            return self.dataset_size