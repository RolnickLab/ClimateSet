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

from emulator.src.utils.utils import get_logger
from emulator.src.data.constants import LON, LAT, SEQ_LEN, INPUT4MIPS_TEMP_RES, CMIP6_TEMP_RES, INPUT4MIPS_NOM_RES, CMIP6_NOM_RES, DATA_DIR, OPENBURNING_MODEL_MAPPING, NO_OPENBURNING_VARS
log = get_logger()

# I think we should have 2 dataset classes, one for input4mips and one for cmip6 (as the latter is model dependent)
# Also, we should create separate datasets for testing and training and validation, to make the splitting easier
"""
- base data set: implements copy to slurm, get item etc pp
- cmip6 data set: model wise
- input4mips data set: same per model

- from datamodule create one of these per train/test/val
"""

class ClimateDataset(torch.utils.data.Dataset):
        def __init__(self,
            years: Union[int,str] = "2015-2020", 
            mode: str = "train", # Train or test maybe # deprecated
            #input4mips_data_dir: Optional[str] ='/scratch/venka97/causalpaca_load/',  #'/home/venka97/scratch/causalpaca/data/',#'/home/venka97/scratch/causalpaca/data/CMIP6/',
            #cmip6_data_dir:  Optional[str] = '/scratch/venka97/causalpaca_processed/',
            output_save_dir: Optional[str] = '/home/mila/c/charlotte.lange/scratch/causalpaca/emulator/DATA',#'/home/venka97/scratch/causal_savedata',
            climate_model: str = 'NorESM2-LM', # implementing single model only for now
            num_ensembles: int = 1, # 1 for first ensemble, -1 for all
            scenarios: Union[List[str], str] = ['ssp126','ssp370','ssp585'],
            historical_years: Union[Union[int, str], None] = "1850-1900",
            #train_experiments: List[str] = ['ssp126','ssp370','ssp585'],
            #test_experiments: Union[str, List[str]] = ['ssp245'],
            out_variables: Union[str, List[str]] = 'pr',
            in_variables: Union[str, List[str]] = ['BC_sum','SO2_sum', 'CH4_sum', 'CO2_sum'],
            seq_to_seq: bool = True, #TODO: implement if false
            channels_last: bool = True,
            load_data_into_mem: bool = True, # Keeping this true be default for now
            input_transform = None, # TODO: implement
            input_normalization = None, #TODO: implement
            output_transform = None,
            output_normalization = None,
            *args, **kwargs,
            
            ):

            super().__init__()
            #self.data_dir = data_dir
            self.output_save_dir = output_save_dir
            
            self.channels_last = channels_last
            self.load_data_into_mem = load_data_into_mem

            if isinstance(in_variables, str):
                in_variables=[in_variables]
            if isinstance(out_variables, str):
                out_variables=[out_variables]
            if isinstance(scenarios, str):
                scenarios=[scenarios]

            self.scenarios=scenarios   

            if isinstance(years, int): 
                self.years = years
            else:
                self.years = self.get_years_list(years, give_list=True) # Can use this to split data into train/val eg. 2015-2080 train. 2080-2100 val.

            if historical_years is None:
                self.historical_years=[]
            elif isinstance(historical_years, int): 
                self.historical_years = historical_years
            else:
                self.historical_years = self.get_years_list(historical_years, give_list=True) # Can use this to split data into train/val eg. 2015-2080 train. 2080-2100 val.
            self.n_years = len(self.years) + len(self.historical_years)

            openburning_specs=OPENBURNING_MODEL_MAPPING[climate_model]


            ds_kwargs = dict(
                scenarios=scenarios,
                years=self.years,
                historical_years=self.historical_years,
                channels_last=channels_last,
                openburning_specs = openburning_specs,
                mode=mode,
                output_save_dir=output_save_dir,
                seq_to_seq=seq_to_seq,
            )
            # creates on cmip and on input4mip dataset
            print("creating input4mips")
            self.input4mips_ds = Input4MipsDataset(variables=in_variables, **ds_kwargs)
            print("creating cmip6")
            #self.cmip6_ds=self.input4mips_ds
            self.cmip6_ds=CMIP6Dataset(climate_model=climate_model, num_ensembles=num_ensembles, variables=out_variables, **ds_kwargs)

        
        # this operates variable vise now.... #TODO: sizes for input4mips / adapt to mulitple vars
        def load_data_into_mem(self, paths: List[List[str]], num_vars, channels_last=True, seq_to_seq=True): #-> np.ndarray():
            array_list =[]
            print("lengh paths", len(paths))
            
            for vlist in paths:
                print("length_paths_list", len(vlist))
                temp_data = xr.open_mfdataset(vlist, concat_dim='time', combine='nested').compute() #.compute is not necessary but eh, doesn't hurt
                temp_data = temp_data.to_array().to_numpy() # Should be of shape (vars, 1036*num_scenarios, 96, 144)
                #temp_data = temp_data.squeeze() # (1036*num_scanarios, 96, 144)
                print("temp data sahpe", temp_data.shape)
                array_list.append(temp_data)
            temp_data = np.concatenate(array_list, axis=0)
            print("temp data sahpe", temp_data.shape)
            temp_data = temp_data.reshape(num_vars,-1, SEQ_LEN, LON, LAT)
            print("temp data sahpe", temp_data.shape)
            if seq_to_seq==False:
                temp_data=temp_data[:,:,-1,:,:] # only take last time step
                temp_data=np.expand_dims(temp_data, axis=2)
                print("seq to 1 temp data sahpe", temp_data.shape)
            if channels_last:
                temp_data = temp_data.transpose((1,2,3,4,0))
            else:
                temp_data = temp_data.transpose((1,2,0,3,4))
            print("final temp data shape", temp_data.shape)
            return temp_data # (86*num_scenarios!, 12, vars, 96, 144). Desired shape where 86*num_scenaiors can be the batch dimension. Can get items of shape (batch_size, 12, 96, 144) -> #TODO: confirm that one item should be one year of one scenario


        def save_data_into_disk(self, data:np.ndarray,fname:str, output_save_dir: str) -> str:
            
            np.savez(os.path.join(output_save_dir, fname), data=data)
            return os.path.join(output_save_dir, fname) 

        def get_save_name_from_kwargs(self, mode:str, file:str,kwargs: Dict):
            fname =""
            for k in kwargs:
                if isinstance(kwargs[k], List):
                    fname+=f"{k}_"+"_".join(kwargs[k])+'_'
                else:
                    fname+=f"{k}_{kwargs[k]}_"
            fname += mode + '_' + file + '.npz'
            print(fname)
            return fname


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
            if len(years)!=9:
                log.warn("Years string must be in the format xxxx-yyyy eg. 2015-2100 with string length 9. Please check the year string.")
                raise ValueError
            splits = years.split('-')
            min_year, max_year = int(splits[0]), int(splits[1])

            if give_list:
                return np.arange(min_year, max_year + 1, step=1) 
            return min_year, max_year


        def get_dataset_statistics(self, data):
            pass
        # Load the saved data here and calculate the mean & std for normalizing the input
        # Return and normalized before __getitem__()
        
        def __getitem__(self, index):  # Dict[str, Tensor]):

            # access data in input4mips and cmip6 datasets
            raw_Xs = self.input4mips_ds[index]
            raw_Ys= self.cmip6_ds[index]
            #raw_Ys = self.cmip6_ds[index]
            if not self.load_data_into_mem:
                X = raw_Xs
                Y = raw_Ys
            else:
                #if self.in
                #TO-DO: Need to write Normalizer transform and To-Tensor transform
                # Doing norm and to-tensor per-instance here. 
                #X_norm = self.input_transforms(self.X[index]) 
                #Y_norm = self.output_transforms(self.Y[index])
                X = raw_Xs
                Y = raw_Ys

            return X,Y

        # @property
        # def name(self):
        #     return self._name.upper()

        def __str__(self):
            s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
            return s

        def __len__(self):
            print(self.input4mips_ds.lenght,self.cmip6_ds.lenght)
            assert self.input4mips_ds.lenght == self.cmip6_ds.lenght, "Datasets not of same length"
            return self.input4mips_ds.lenght


class CMIP6Dataset(ClimateDataset):
    """ 
    
    Use first ensemble member for now 
    Option to use multile ensemble member later
    Give option for which variable to use
    Load 3 scenarios for train/val: Take this as a list
    Process and save this as .npz in $SLURM_TMPDIR
    Load these in train/val/test Dataloader functions
    Keep one scenario for testing
    # Target shape (85 * 12, 1, 144, 96) # ! * num_scenarios!!
    """
    def __init__( # inherits all the stuff from Base
            self,
            years: Union[int,str], 
            historical_years: Union[int,str],
            #mode: str = "train", # Train or test maybe # deprecated
            data_dir: Optional[str] = DATA_DIR,
            #output_save_dir: Optional[str] = '/home/venka97/scratch/causal_savedata',
            climate_model: str = 'NorESM2-LM',
            num_ensembles: int = 1, # 1 for first ensemble, -1 for all
            scenarios: List[str] = ['ssp126','ssp370','ssp585'],
            #train_experiments: List[str] = ['ssp126','ssp370','ssp585'],
            #test_experiments: Union[str, List[str]] = ['ssp245'],
            variables: List[str] = ['pr'],
            mode: str = 'train',
            output_save_dir: str = "",
            channels_last: bool = True,
            seq_to_seq: bool = True,
            *args, **kwargs,
            #load_data_into_mem: bool = True, # Keeping this true be default for now
    ):

        #self.mode = mode
        self.root_dir = os.path.join(data_dir, "targets/CMIP6")
        #self.output_save_dir = output_save_dir
        self.input_nc_files = []
        self.output_nc_files = []

        fname_kwargs = dict(
            climate_model = climate_model,
            num_ensembles = num_ensembles,
            years=f"{years[0]}-{years[-1]}", 
            historical_years= f"{historical_years[0]}-{historical_years[-1]}",
            variables = variables,
            scenarios = scenarios,
            channels_last = channels_last,
            seq_to_seq=seq_to_seq
        )
     

        #TO-DO: This is just getting the list of .nc files for targets. Put this logic in a function and get input list as well.
        # In a function, we can call CausalDataset() instance for train and test separately to load the data

        if isinstance(climate_model, str):
            self.root_dir = os.path.join(self.root_dir, climate_model)
        else:
            # Logic for multiple climate models, not sure how to load/create dataset yet
            log.warn("Data loader not yet implemented for multiple climate models.")
            raise NotImplementedError

        if num_ensembles == 1:
            ensembles = os.listdir(self.root_dir)
            self.ensemble_dir =  os.path.join(self.root_dir, ensembles[0]) # Taking first ensemble member
        else:
            log.warn("Data loader not yet implemented for mulitple ensemble members.")
            raise NotImplementedError

       
        

        # Split the data here using n_years if needed,
        # else do random split logic here


        # Check here if os.path.isfile($SCRATCH/data.npz) exists
        # if it does, use self._reload data(path)
        fname = self.get_save_name_from_kwargs(mode=mode, file='target', kwargs=fname_kwargs)


        if os.path.isfile(os.path.join(output_save_dir, fname)): # we first need to get the name here to test that...
            self.data_path=os.path.join(output_save_dir, fname)
            print("path exiists, reloading")
            self.Data = self._reload_data(self.data_path)

        else:
            # Add code here for adding files for input nc data
            # Similar to the loop below for output files

            # Got all the files paths at this point, now open and merge

            # List of output files
            files_per_var=[]
            for var in variables:
                output_nc_files=[]
        
                for exp in scenarios:
                    if exp=='historical':
                        get_years=historical_years
                    else:
                        get_years=years
                    for y in get_years:
                        var_dir = os.path.join(self.ensemble_dir, exp, var, f'{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}') 
                        print(var_dir)
                        files = glob.glob(var_dir + f'/*.nc', recursive=True)
                        # loads all years! implement plitting
                        output_nc_files += files
                files_per_var.append(output_nc_files)

            #self.raw_data_input = self.load_data_into_mem(self.input_nc_files) #currently don't have input paths etc
            self.raw_data = self.load_data_into_mem(files_per_var, num_vars=len(variables), channels_last=channels_last, seq_to_seq=seq_to_seq) 

            #self.input_path = self.save_data_into_disk(self.raw_data_input, self.mode, 'input')
            self.data_path = self.save_data_into_disk(self.raw_data, fname, output_save_dir)

            #self.copy_to_slurm(self.input_path)
            self.copy_to_slurm(self.data_path)

            # Call _reload_data here with self.input_path and self.output_path
            # self.X = self._reload_data(input_path)
            self.Data = self._reload_data(self.data_path)
            # Write a normalize transform to calculate mean and std
            # Either normalized whole array here or per instance getitem, that maybe faster

            # Now X and Y is ready for getitem
        self.lenght=self.Data.shape[0]
    def __getitem__(self, index):
        return self.Data[index]



class Input4MipsDataset(ClimateDataset):
    """ 
    Loads all scenarios for a given var / for all vars
    """
    def __init__( # inherits all the stuff from Base
            self,
            years: Union[int,str], 
            historical_years: Union[int,str],
            data_dir: Optional[str] = DATA_DIR,
            variables:  List[str] = ['BC_sum'],
            scenarios: List[str] = ['ssp126','ssp370','ssp585'],
            channels_last: bool = True,
            openburning_specs : Tuple[str] = ("no_fires", "no_fires"),
            mode : str = 'train',
            output_save_dir : str = "",
            *args, **kwargs,
        ):

        self.channels_last=channels_last

        #self.mode = mode
        self.root_dir = os.path.join(data_dir, "inputs/input4mips")
        #self.output_save_dir = output_save_dir
        self.input_nc_files = []
        self.output_nc_files = []

        fname_kwargs = dict(
            years=f"{years[0]}-{years[-1]}", 
            historical_years= f"{historical_years[0]}-{historical_years[-1]}",
            variables = variables,
            scenarios = scenarios,
            channels_last = channels_last,
            openburning_specs =openburning_specs,
            seq_to_seq=True
        )

        
        historical_openburning, ssp_openburning = openburning_specs

        # Split the data here using n_years if needed,
        # else do random split logic here
        fname = self.get_save_name_from_kwargs(mode=mode, file='input', kwargs=fname_kwargs)

        # Check here if os.path.isfile($SCRATCH/data.npz) exists #TODO: check if exists on slurm
        # if it does, use self._reload data(path)
        if os.path.isfile(os.path.join(output_save_dir, fname)): # we first need to get the name here to test that...
            self.data_path=os.path.join(output_save_dir, fname)
            print("path exists, reloading")
            self.Data = self._reload_data(self.data_path)
           
        else:
            files_per_var=[]
            for var in variables:
                output_nc_files=[]
                for exp in scenarios: # TODO: implement getting by years! also sub seletction for historical years
                    print(var, exp)
                    if var in NO_OPENBURNING_VARS:
                        filter_path_by=''
                        print("CO2 found")
                    elif exp=='historical':
                        filter_path_by=historical_openburning
                    else:
                        filter_path_by=ssp_openburning
                    print("filter path", filter_path_by)
                    var_dir = os.path.join(self.root_dir, exp, var, f'{INPUT4MIPS_NOM_RES}/{INPUT4MIPS_TEMP_RES}')
                    files = glob.glob(var_dir + f'/**/*{filter_path_by}*.nc', recursive=True)
                
                    print(files)
                    #break
                    output_nc_files += files
                files_per_var.append(output_nc_files)

            #self.raw_data_input = self.load_data_into_mem(self.input_nc_files) #currently don't have input paths etc
            self.raw_data = self.load_data_into_mem(files_per_var, num_vars=len(variables), channels_last=self.channels_last, seq_to_seq=True) # we always want the full sequence for input4mips

            #self.input_path = self.save_data_into_disk(self.raw_data_input, self.mode, 'input')
            self.data_path = self.save_data_into_disk(self.raw_data, fname, output_save_dir)

            #self.copy_to_slurm(self.input_path)
            self.copy_to_slurm(self.data_path)

            # Call _reload_data here with self.input_path and self.output_path
            # self.X = self._reload_data(input_path)
            self.Data = self._reload_data(self.data_path)
            # Write a normalize transform to calculate mean and std
            # Either normalized whole array here or per instance getitem, that maybe faster

            # Now X and Y is ready for getitem
        print("sDATA shape", self.Data.shape)
        self.lenght=self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]


if __name__=="__main__":
    ds = ClimateDataset(seq_to_seq=False)
    #for (i,j) in ds:
        #print("i:", i.shape)
        #print("j:", j.shape)
    print(len(ds))