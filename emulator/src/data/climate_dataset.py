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
from emulator.src.data.constants import LON, LAT, SEQ_LEN, INPUT4MIPS_TEMP_RES, CMIP6_TEMP_RES, INPUT4MIPS_NOM_RES, CMIP6_NOM_RES, DATA_DIR, OPENBURNING_MODEL_MAPPING, NO_OPENBURNING_VARS, AVAILABLE_MODELS_FIRETYPE
log = get_logger()

"""
- base data set: implements copy to slurm, get item etc pp
- cmip6 data set: model-member wise
- input4mips data set: same per model-member pairing (but unique to each openburning spec)

- from datamodule create one of these per train/test/val
"""

class ClimateDataset(torch.utils.data.Dataset):
        def __init__(self,
            years: Union[int,str] = "2015-2020", 
            mode: str = "train+val", # Train or test maybe # deprecated
            output_save_dir: Optional[str] = DATA_DIR, 
            climate_model: str = 'NorESM2-LM', # implementing single model, for mulitple models use SuperClimateDataset
            num_ensembles: int = 1, # 1 for first ensemble, -1 for all
            scenarios: Union[List[str], str] = ['ssp126','ssp370','ssp585'],
            historical_years: Union[Union[int, str], None] = "1850-1900",
            out_variables: Union[str, List[str]] = 'pr',
            in_variables: Union[str, List[str]] = ['BC_sum','SO2_sum', 'CH4_sum', 'CO2_sum'],
            seq_to_seq: bool = True, #TODO: implement if false
            seq_len: int = 12,
            channels_last: bool = False,
            load_data_into_mem: bool = True, # Keeping this true be default for now
            input_transform = None, # TODO: implement
            input_normalization = 'z-norm', #TODO: implement
            output_transform = None,
            output_normalization = 'z-norm',
            *args, **kwargs,
            
            ):

            super().__init__()

            self.test_dir = output_save_dir
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
            self.num_ensembles=num_ensembles

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
            self.n_years = len(self.years) + len(self.historical_years) if 'historical' in self.scenarios else len(self.years)

            if climate_model in AVAILABLE_MODELS_FIRETYPE:
                openburning_specs=OPENBURNING_MODEL_MAPPING[climate_model]
            else:
                openburning_specs=OPENBURNING_MODEL_MAPPING["other"]


            ds_kwargs = dict(
                scenarios=scenarios,
                years=self.years,
                historical_years=self.historical_years,
                channels_last=channels_last,
                openburning_specs = openburning_specs,
                mode=mode,
                output_save_dir=output_save_dir,
                seq_to_seq=seq_to_seq,
                seq_len=seq_len
            )
            # creates on cmip and on input4mip dataset
            print("Creating input4mips...")
            self.input4mips_ds = Input4MipsDataset(variables=in_variables, **ds_kwargs)
            print("Creating cmip6...")
            self.cmip6_ds=CMIP6Dataset(climate_model=climate_model, num_ensembles=num_ensembles, variables=out_variables, **ds_kwargs)

        
        # this operates variable vise now.... 
        def load_into_mem(self, paths: List[List[str]], num_vars, channels_last=True, seq_to_seq=True, seq_len=12): #-> np.ndarray():
           
            array_list =[]
            for vlist in paths:
                print("Number of files per var:", len(vlist))
                temp_data = xr.open_mfdataset(vlist, concat_dim='time', combine='nested').compute() #.compute is not necessary but eh, doesn't hurt
                temp_data = temp_data.to_array().to_numpy() # Should be of shape (vars, years*ensemble_members*num_scenarios, lon, lat)
                array_list.append(temp_data)
            temp_data = np.concatenate(array_list, axis=0)

            if seq_len!=SEQ_LEN:
                print("Choosing a sequence length greater or lesser than the data sequence length.")
                new_num_years = int(np.floor(temp_data.shape[1]/seq_len/len(self.scenarios)))
                
                # divide by scenario num and seq len, round
                # multiply with scenario num an dseq len to get correct shape
                new_shape_one=new_num_years*len(self.scenarios)
                assert new_shape_one*seq_len > temp_data.shape[1], f"New sequence length {seq_len} greater than available years {temp_data.shape[1]}!"
                print(f"New sequence length: {seq_len} Dropping {temp_data.shape[1]-(new_shape_one*seq_len)} years")
                temp_data=temp_data[:,:(new_shape_one*seq_len),:]
                
            else:
                new_shape_one=int(temp_data.shape[1]/seq_len)

            temp_data = temp_data.reshape(num_vars,new_shape_one, seq_len, LON, LAT) # num_vars, num_scenarios*num_remainding_years, seq_len,lon,lat)
            if seq_to_seq==False:
                temp_data=temp_data[:,:,-1,:,:] # only take last time step
                temp_data=np.expand_dims(temp_data, axis=2)
            if channels_last:
                temp_data = temp_data.transpose((1,2,3,4,0))
            else:
                temp_data = temp_data.transpose((1,2,0,3,4))
            return temp_data # (years*num_scenarios, seq_len, vars, lon, lat)


        def save_data_into_disk(self, data:np.ndarray,fname:str, output_save_dir: str) -> str:
            
            np.savez(os.path.join(output_save_dir, fname), data=data)
            return os.path.join(output_save_dir, fname) 

        def get_save_name_from_kwargs(self, mode:str, file:str,kwargs: Dict):
            fname =""
            #print("KWARGs:", kwargs)

            if file == 'statistics':
                # only cmip 6
                if 'climate_model' in kwargs:
                    fname += kwargs['climate_model'] + '_'
                if 'num_ensembles' in kwargs:
                    fname += str(kwargs['num_ensembles']) + '_'
                # all
                fname += '_'.join(kwargs['variables']) +'_' #+ '_' + kwargs['input_normalization']
                

            else:

                for k in kwargs:
                    if isinstance(kwargs[k], List):
                        fname+=f"{k}_"+"_".join(kwargs[k])+'_'
                    else:
                        fname+=f"{k}_{kwargs[k]}_"
            
            if file == 'statistics':
                fname += mode + '_' + file + '.npy'
            else:
                fname += mode + '_' + file + '.npz'

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


        def get_dataset_statistics(self, data, mode, type='z-norm', mips= 'cmip6'):
            if mode == 'train' or mode == 'train+val':
                if type == 'z-norm':
                    mean, std = self.get_mean_std(data)
                    return mean, std
                elif type == 'minmax':
                    min_val, max_val = self.get_min_max(data)
                    return min_val, max_val
                else:
                    print('Normalizing of type {0} has not been implemented!'.format(type))
            else:
                print('In testing mode, skipping statistics calculations.')


        def get_mean_std(self, data):
            # data shape (years*scenarios, seq, vars, lon, lat)
            if self.channels_last:
                data = np.moveaxis(data, -1, 0)
            else: 
                data = np.moveaxis(data, 2, 0) 
            vars_mean = np.mean(data, axis=(1, 2, 3, 4)) 
            vars_std = np.std(data, axis=(1, 2, 3, 4))
            vars_mean = np.expand_dims(vars_mean, (1, 2, 3, 4)) # Shape of mean & std (4, 1, 1, 1, 1)
            vars_std = np.expand_dims(vars_std, (1, 2, 3, 4))
            return vars_mean, vars_std

        def get_min_max(self, data):

            if self.channels_last:
                data = np.moveaxis(data, -1, 0)
            else:
                data = np.moveaxis(data, 2, 0) # shape (258, 12, 4, 96, 144) -> (4, 258, 12, 96, 144) easier to calulate statistics
            vars_max = np.max(data, axis=(1, 2, 3, 4)) # shape (258, 12, 4, 96, 144)
            vars_min = np.min(data, axis=(1, 2, 3, 4))
            vars_max = np.expand_dims(vars_max, (1, 2, 3, 4)) # shape of mean & std (4, 1, 1, 1, 1)
            vars_min= np.expand_dims(vars_min, (1, 2, 3, 4))
            return vars_min, vars_max

        def normalize_data(self, data, stats, type='z-norm'):
            # Only implementing z-norm for now
            # z-norm: (data-mean)/(std + eps); eps=1e-9
            # min-max = (v - v.min()) / (v.max() - v.min())

            print('Normalizing data...')
            if self.channels_last:
                data = np.moveaxis(data, -1, 0) # vars from last to 0 (num_vars, years, seq_len, lon, lat)
            else:
                data = np.moveaxis(data, 2, 0) # shape (years, seq_len, num_vars, lon, lat) -> (num_vars, years, seq_len, lon, lat) 
            
            print("mean", stats['mean'].shape, 'std', stats['std'].shape)
            norm_data = (data - stats['mean'])/(stats['std'])

            if self.channels_last:
                norm_data = np.moveaxis(norm_data, 0, -1)
            else:     
                norm_data = np.moveaxis(norm_data, 0, 2) # Switch back to (years, seq_len, num_vars, 96, 144)
            
            return norm_data

        def write_dataset_statistics(self, fname, stats):
            np.save(os.path.join(self.output_save_dir, fname), stats, allow_pickle=True)            
            return os.path.join(self.output_save_dir, fname) 

        def load_dataset_statistics(self, fname, mode, mips):
            if 'train_' in fname:
                fname = fname.replace('train', 'train+val')
            elif 'test' in fname:
                fname = fname.replace('test', 'train+val')

            stats_data = np.load(os.path.join(self.output_save_dir, fname), allow_pickle=True).item()
                
            return stats_data      
        
        def __getitem__(self, index):  # Dict[str, Tensor]):

            # access data in input4mips and cmip6 datasets
            # for mulitple ensemble members, we have to repeat the input4mips dataset!
            if index>=self.input4mips_ds.length-1:
                index=index-self.input4mips_ds.length # just shifiting back by one time the full dataset
            raw_Xs = self.input4mips_ds[index]
            raw_Ys= self.cmip6_ds[index]
            if not self.load_data_into_mem:
                X = raw_Xs
                Y = raw_Ys
            else:
                #TO-DO: Need to write Normalizer transform and To-Tensor transform
                # Doing norm and to-tensor per-instance here. 
                #X_norm = self.input_transforms(self.X[index]) 
                #Y_norm = self.output_transforms(self.Y[index])
                X = raw_Xs
                Y = raw_Ys

            return X,Y

        def __str__(self):
            s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
            return s

        def __len__(self):
            print('Input4mips', self.input4mips_ds.length, 'CMIP6 data' , self.cmip6_ds.length)
            # cmip must be num_ensemble members times input4mips
            assert self.input4mips_ds.length*self.num_ensembles == self.cmip6_ds.length, f"CMIP6 must be num_ensembles times the length of input4mips. Got {self.cmip6_ds.length} and {self.input4mips_ds.length}"
            return self.cmip6_ds.length


class CMIP6Dataset(ClimateDataset):
    """ 
    CMIP6 Dataset. Containing data for single climate models but potentially multiple ensemble members.
    Iiterating overy every member.
    """
    def __init__( # inherits all the stuff from Base
            self,
            years: Union[int,str], 
            historical_years: Union[int,str],
            data_dir: Optional[str] = DATA_DIR,
            climate_model: str = 'NorESM2-LM',
            num_ensembles: int = 1, # 1 for first ensemble, -1 for all
            scenarios: List[str] = ['ssp126','ssp370','ssp585'],
            variables: List[str] = ['pr'],
            mode: str = 'train',
            output_save_dir: str = "",
            channels_last: bool = True,
            seq_to_seq: bool = True,
            seq_len: int = 12,
            *args, **kwargs,
    ):

        self.mode = mode
        self.output_save_dir = output_save_dir
        self.root_dir = os.path.join(data_dir, "outputs/CMIP6")

        self.input_nc_files = []
        self.output_nc_files = []

        self.scenarios=scenarios
        self.channels_last=channels_last

        fname_kwargs = dict(
            climate_model = climate_model,
            num_ensembles = num_ensembles,
            years=f"{years[0]}-{years[-1]}", 
            historical_years= f"{historical_years[0]}-{historical_years[-1]}",
            variables = variables,
            scenarios = scenarios,
            channels_last = channels_last,
            seq_to_seq=seq_to_seq,
            seq_=seq_len
        )
     
        if isinstance(climate_model, str):
            self.root_dir = os.path.join(self.root_dir, climate_model)
        else:
            log.warn("For loading multiple climate models, please make sure to use the Super Climate Dataset Class.")
            raise NotImplementedError

        if num_ensembles == 1:
            ensembles = os.listdir(self.root_dir)
            self.ensemble_dir =  [os.path.join(self.root_dir, ensembles[0])] # Taking first ensemble member
        else:
            print("Multiple ensembles", num_ensembles)
            self.ensemble_dir = []
            ensembles = os.listdir(self.root_dir)
            for i,folder in enumerate(ensembles):
                self.ensemble_dir.append(os.path.join(self.root_dir, folder)) # Taking multiple ensemble members
                if i==(num_ensembles-1):
                    break # if num_ensemble ==-1 we take all
            
        # Check here if os.path.isfile($SCRATCH/data.npz) exists
        # if it does, use self._reload data(path)
        fname = self.get_save_name_from_kwargs(mode=mode, file='target', kwargs=fname_kwargs)

        if os.path.isfile(os.path.join(output_save_dir, fname)): # we first need to get the name here to test that...
            self.data_path=os.path.join(output_save_dir, fname)
            print("path exists, reloading")
            self.Data = self._reload_data(self.data_path)

            # Load stats and normalize
            stats_fname = self.get_save_name_from_kwargs(mode=mode, file='statistics', kwargs=fname_kwargs)
            
            stats = self.load_dataset_statistics(os.path.join(self.output_save_dir, stats_fname), mode=self.mode, mips='cmip6')
            self.Data = self.normalize_data(self.Data, stats)

        else:
            # Getting list of file names per variable for open and merging
            files_per_var=[]
            for var in variables:
             
                output_nc_files=[]
                for exp in scenarios:
                    if exp=='historical':
                        get_years=historical_years
                    else:
                        get_years=years
                    for y in get_years:
                        for em in self.ensemble_dir:
                            var_dir = os.path.join(em, exp, var, f'{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}') 
                            files = glob.glob(var_dir + f'/*.nc', recursive=True)
                            if len(files)==0:
                                print("No files for this scenario, year, ensemble member pairing:", exp, y, em)
                                exit(0)
                            # loads all years!
                            output_nc_files += files
                files_per_var.append(output_nc_files)
            self.raw_data = self.load_into_mem(files_per_var, num_vars=len(variables), channels_last=channels_last, seq_to_seq=seq_to_seq, seq_len=seq_len) 

            if self.mode == 'train' or self.mode == 'train+val':
                stats_fname = self.get_save_name_from_kwargs(mode=mode, file='statistics', kwargs=fname_kwargs)

                if os.path.isfile(stats_fname):
                    print('Stats file already exists! Loading from memory.')
                    stats = self.load_statistics_data(stats_fname)
                    self.norm_data = self.normalize_data(self.raw_data, stats)

                else:    
                    stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips='cmip6')
                    stats = {'mean': stat1, 'std': stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    
                    save_file_name = self.write_dataset_statistics(stats_fname, stats)
                    print("WROTE STATISTICS", save_file_name)

                self.norm_data = self.normalize_data(self.raw_data, stats)


            elif self.mode == 'test':
                stats_fname = self.get_save_name_from_kwargs(mode='train+val', file='statistics', kwargs=fname_kwargs)
                save_file_name = os.path.join(self.output_save_dir, fname)
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips='cmip6')
                self.norm_data = self.normalize_data(self.raw_data, stats)

            self.data_path = self.save_data_into_disk(self.raw_data, fname, output_save_dir)

            self.copy_to_slurm(self.data_path)

            self.Data = self.norm_data # ready for getitem
        self.length=self.Data.shape[0]



    def __getitem__(self, index):
        return self.Data[index]



class Input4MipsDataset(ClimateDataset):
    """ 
    Loads all scenarios for a given variable.
    """
    def __init__(
            self,
            years: Union[int,str], 
            historical_years: Union[int,str],
            data_dir: Optional[str] = DATA_DIR,
            variables:  List[str] = ['BC_sum'],
            scenarios: List[str] = ['ssp126','ssp370','ssp585'],
            channels_last: bool = False,
            openburning_specs : Tuple[str] = ("no_fires", "no_fires"),
            mode : str = 'train',
            output_save_dir : str = "",
            seq_to_seq: bool = True,
            seq_len: int = 12,
            *args, **kwargs,
        ):

        self.channels_last=channels_last

        self.mode = mode
        self.root_dir = os.path.join(data_dir, 'inputs/input4mips')
        self.output_save_dir = output_save_dir
        self.input_nc_files = []
        self.output_nc_files = []

        self.scenarios=scenarios
        fname_kwargs = dict(
            years=f"{years[0]}-{years[-1]}", 
            historical_years= f"{historical_years[0]}-{historical_years[-1]}",
            variables = variables,
            scenarios = scenarios,
            channels_last = channels_last,
            openburning_specs =openburning_specs,
            seq_to_seq=seq_to_seq,
            seq_len=seq_len
        )

        
        historical_openburning, ssp_openburning = openburning_specs

        fname = self.get_save_name_from_kwargs(mode=mode, file='input', kwargs=fname_kwargs)

        if os.path.isfile(os.path.join(output_save_dir, fname)): # we first need to get the name here to test that...
            self.data_path=os.path.join(output_save_dir, fname)
            print("path exists, reloading")
            self.Data = self._reload_data(self.data_path)

            # Load stats and normalize
            stats_fname = self.get_save_name_from_kwargs(mode=mode, file='statistics', kwargs=fname_kwargs)
            print(stats_fname)
            stats = self.load_dataset_statistics(os.path.join(self.output_save_dir, stats_fname), mode=self.mode, mips='input4mips')
            self.Data = self.normalize_data(self.Data, stats)
           
      
        else:
            files_per_var=[]
            for var in variables:
                output_nc_files=[]
        
                for exp in scenarios:
                    if exp=='historical':
                        get_years=historical_years
                    else:
                        get_years=years
                    for y in get_years:
                        var_dir = os.path.join(self.root_dir, exp, var, f'{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}') 

                output_nc_files=[]
                for exp in scenarios: # TODO: implement getting by years! also sub seletction for historical years
                    if var in NO_OPENBURNING_VARS:
                        filter_path_by=''
                    elif exp=='historical':
                        filter_path_by=historical_openburning
                        get_years=historical_years
                    else:
                        filter_path_by=ssp_openburning
                        get_years=years


                    for y in get_years:
                        var_dir = os.path.join(self.root_dir, exp, var, f'{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}')
                        files = glob.glob(var_dir + f'/**/*{filter_path_by}*.nc', recursive=True)
                        output_nc_files += files
                files_per_var.append(output_nc_files)

            self.raw_data = self.load_into_mem(files_per_var, num_vars=len(variables), channels_last=self.channels_last, seq_to_seq=True ,seq_len=seq_len) # we always want the full sequence for input4mips

            if self.mode == 'train' or self.mode == 'train+val':
                stats_fname = self.get_save_name_from_kwargs(mode=mode, file='statistics', kwargs=fname_kwargs)

                if os.path.isfile(stats_fname):
                    print('Stats file already exists! Loading from mempory.')
                    stats = self.load_statistics_data(stats_fname)
                    self.norm_data = self.normalize_data(self.raw_data, stats)

                else:    
                    stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips='cmip6')
                    stats = {'mean': stat1, 'std': stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    save_file_name = self.write_dataset_statistics(stats_fname, stats)

                self.norm_data = self.normalize_data(self.raw_data, stats)


            elif self.mode == 'test':
                stats_fname = self.get_save_name_from_kwargs(mode='train+val', file='statistics', kwargs=fname_kwargs) #Load train stats cause we don't calculcate norm stats for test.
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips='input4mips')
                self.norm_data = self.normalize_data(self.raw_data, stats)

            self.data_path = self.save_data_into_disk(self.raw_data, fname, output_save_dir)

            self.copy_to_slurm(self.data_path)

            self.Data = self.norm_data
        self.length=self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]


if __name__=="__main__":
    # FGOALS-g3 MPI-ESM1-2-HR
    ds = ClimateDataset(seq_to_seq=True, in_variables=['BC_sum','SO2_sum', 'CH4_sum'], scenarios=["historical","ssp370"],climate_model="MPI-ESM1-2-HR", seq_len=12, num_ensembles=2, channels_last=False)
    #for (i,j) in ds:
        #print("i:", i.shape)
        #print("j:", j.shape)
    print(len(ds))

    for i,(x,y) in enumerate(ds):
        print(i)