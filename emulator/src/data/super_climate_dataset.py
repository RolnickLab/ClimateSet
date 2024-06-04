import copy
import logging
import os
import glob
import pickle
import shutil
import zipfile
from typing import Dict, Optional, List, Callable, Tuple, Union
import copy
import numpy as np
import xarray as xr
import torch
from torch import Tensor
import threading


from emulator.src.utils.utils import get_logger, all_equal, map_variables_targetmip
from emulator.src.data.constants import (
    LON,
    LAT,
    SEQ_LEN,
    INPUT4MIPS_TEMP_RES,
    CMIP6_TEMP_RES,
    INPUT4MIPS_NOM_RES,
    CMIP6_NOM_RES,
    DATA_DIR,
    OPENBURNING_MODEL_MAPPING,
    NO_OPENBURNING_VARS,
    AVAILABLE_MODELS_FIRETYPE,
)
log = get_logger()
from abc import ABC, abstractmethod

class ABC_Climate_Dataset(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for climate datasets.

    Attributes:
        mode (str): Mode of the dataset, either 'train', 'val', or 'test'.
    """

    @abstractmethod
    def __init__(self, mode: str = "train+val"):
        """
        Initializes the dataset with the given mode.

        Args:
            mode (str): Mode of the dataset, either 'train', 'val', or 'test'. Default is 'train+val'.
        """
        self.mode = mode

    @abstractmethod
    def __getitem__(self, index):
        """Abstract method to get an item by index."""
        pass

    def get_save_name_from_kwargs(self, mode: str, file: str, kwargs: Dict) -> str:
        """
        Generates a save file name based on given mode, file type, and additional keyword arguments.

        Args:
            mode (str): Mode of the dataset.
            file (str): File type.
            kwargs (Dict): Additional arguments to include in the file name.

        Returns:
            str: Generated file name.
        """
        fname = ""

        if file == "statistics":
            # Only CMIP6
            if "climate_model" in kwargs:
                fname += kwargs["climate_model"] + "_"
            if "num_ensembles" in kwargs:
                fname += str(kwargs["num_ensembles"]) + "_"
            # All variables
            fname += "_".join(kwargs["variables"]) + "_"
        else:
            for k, v in kwargs.items():
                if isinstance(v, list):
                    fname += f"{k}_" + "_".join(map(str, v)) + "_"
                else:
                    fname += f"{k}_{v}_"

        if file == "statistics":
            fname += f"{mode}_{file}.npy"
        else:
            fname += f"{mode}_{file}.npz"

        return fname

    def _reload_data(self, fname: str):
        """
        Reloads data from a file.

        Args:
            fname (str): File name.

        Returns:
            The reloaded data.
        """
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

    def load_dataset_statistics(self, fname: str, mode: str, mips: str):
        """
        Loads dataset statistics from a file.

        Args:
            fname (str): File name.
            mode (str): Mode of the dataset.
            mips (str): MIPS type.

        Returns:
            dict: Loaded statistics data.
        """
        if "train_" in fname:
            fname = fname.replace("train", "train+val")
        elif "test" in fname:
            fname = fname.replace("test", "train+val")

        stats_data = np.load(os.path.join(self.output_save_dir, fname), allow_pickle=True).item()
        return stats_data

    def normalize_data(self, data: np.ndarray, stats: dict, norm_type: str = "z-norm") -> np.ndarray:
        """
        Normalizes data using given statistics and normalization type.

        Args:
            data (np.ndarray): Data to normalize.
            stats (dict): Statistics for normalization.
            norm_type (str): Type of normalization. Default is 'z-norm'.

        Returns:
            np.ndarray: Normalized data.
        """
        if self.channels_last:
            data = np.moveaxis(data, -1, 0)  # Move variables from last to first axis
        else:
            data = np.moveaxis(data, 2, 0)  # Move variables to first axis

        norm_data = (data - stats["mean"]) / stats["std"]

        if self.channels_last:
            norm_data = np.moveaxis(norm_data, 0, -1)
        else:
            norm_data = np.moveaxis(norm_data, 0, 2)  # Switch back to original shape

        return norm_data

    def load_into_mem(
        self, paths: List[List[str]], num_vars: int, channels_last: bool = True, 
        seq_to_seq: bool = True, seq_len: int = 12
    ) -> np.ndarray:
        """
        Loads dataset into memory.

        Args:
            paths (List[List[str]]): List of paths to the data files.
            num_vars (int): Number of variables.
            channels_last (bool): If True, channels are last. Default is True.
            seq_to_seq (bool): If True, uses sequence-to-sequence format. Default is True.
            seq_len (int): Length of the sequence. Default is 12.

        Returns:
            np.ndarray: Loaded data.
        """
        array_list = []
        for vlist in paths:
            temp_data = xr.open_mfdataset(vlist, concat_dim="time", combine="nested").compute()
            temp_data = temp_data.to_array().to_numpy()
            array_list.append(temp_data)
        temp_data = np.concatenate(array_list, axis=0)

        if seq_len != SEQ_LEN:
            new_num_years = int(np.floor(temp_data.shape[1] / seq_len / len(self.scenarios)))
            new_shape_one = new_num_years * len(self.scenarios)
            assert new_shape_one * seq_len > temp_data.shape[1], (
                f"New sequence length {seq_len} greater than available years {temp_data.shape[1]}!"
            )
            temp_data = temp_data[:, : (new_shape_one * seq_len), :]
        else:
            new_shape_one = int(temp_data.shape[1] / seq_len)

        temp_data = temp_data.reshape(num_vars, new_shape_one, seq_len, LON, LAT)

        if not seq_to_seq:
            temp_data = temp_data[:, :, -1, :, :]  # Only take last time step
            temp_data = np.expand_dims(temp_data, axis=2)

        if channels_last:
            temp_data = temp_data.transpose((1, 2, 3, 4, 0))
        else:
            temp_data = temp_data.transpose((1, 2, 0, 3, 4))

        return temp_data

    def write_dataset_statistics(self, fname: str, stats: dict) -> str:
        """
        Writes dataset statistics to a file.

        Args:
            fname (str): File name.
            stats (dict): Statistics data.

        Returns:
            str: Path to the saved file.
        """
        np.save(os.path.join(self.output_save_dir, fname), stats, allow_pickle=True)
        return os.path.join(self.output_save_dir, fname)

    def save_data_into_disk(self, data: np.ndarray, fname: str, output_save_dir: str) -> str:
        """
        Saves data into disk.

        Args:
            data (np.ndarray): Data to save.
            fname (str): File name.
            output_save_dir (str): Directory to save the file.

        Returns:
            str: Path to the saved file.
        """
        np.savez(os.path.join(output_save_dir, fname), data=data)
        return os.path.join(output_save_dir, fname)

    def copy_to_slurm(self, fname: str):
        """
        Copies data to SLURM directory.

        Args:
            fname (str): File name.
        """
        # TODO: Implement SLURM copy logic
        pass

    def get_dataset_statistics(self, data: np.ndarray, mode: str, norm_type: str = "z-norm", mips: str = "cmip6"):
        """
        Gets dataset statistics.

        Args:
            data (np.ndarray): Data to calculate statistics for.
            mode (str): Mode of the dataset.
            norm_type (str): Type of normalization. Default is 'z-norm'.
            mips (str): MIPS type. Default is 'cmip6'.

        Returns:
            Tuple: Mean and standard deviation or min and max values.
        """
        if mode in ["train", "train+val"]:
            if norm_type == "z-norm":
                return self.get_mean_std(data)
            elif norm_type == "minmax":
                return self.get_min_max(data)
            else:
                print(f"Normalization of type {norm_type} has not been implemented!")
        else:
            print("In testing mode, skipping statistics calculations.")

    def get_mean_std(self, data: np.ndarray):
        """
        Calculates mean and standard deviation of the data.

        Args:
            data (np.ndarray): Data to calculate statistics for.

        Returns:
            Tuple: Mean and standard deviation.
        """
        if self.channels_last:
            data = np.moveaxis(data, -1, 0)
        else:
            data = np.moveaxis(data, 2, 0)
        
        vars_mean = np.mean(data, axis=(1, 2, 3, 4))
        vars_std = np.std(data, axis=(1, 2, 3, 4))

        vars_mean = np.expand_dims(vars_mean, (1, 2, 3, 4))
        vars_std = np.expand_dims(vars_std, (1, 2, 3, 4))

        return vars_mean, vars_std

    def get_min_max(self, data: np.ndarray):
        """
        Calculates min and max values of the data.

        Args:
            data (np.ndarray): Data to calculate statistics for.

        Returns:
            Tuple: Min and max values.
        """
        if self.channels_last:
            data = np.moveaxis(data, -1, 0)
        else:
            data = np.moveaxis(data, 2, 0)
        
        vars_max = np.max(data, axis=(1, 2, 3, 4))
        vars_min = np.min(data, axis=(1, 2, 3, 4))

        vars_max = np.expand_dims(vars_max, (1, 2, 3, 4))
        vars_min = np.expand_dims(vars_min, (1, 2, 3, 4))

        return vars_min, vars_max

    def __len__(self) -> int:
        """
        Returns the length of the dataset based on the mode.

        Returns:
            int: Length of the dataset.
        """
        if self.mode == 'train' or self.mode == 'test':
            return len(self.index_manager.train_indexes)
        elif self.mode == 'val':
            return len(self.index_manager.val_indexes)
        else:
            print(f"Unknown mode: {self.mode}")
            raise ValueError
    

class SuperClimateDataset(ABC_Climate_Dataset):
    """
    Class to efficiently load data for multi-model multi-ensemble experiments.
    Input4MIPS data is only loaded once (per desired fire-model as specified by the climate models)

    """
    def __init__(
        self,
        index_manager = None,
        years: Union[int, str] = "2015-2020",
        mode: str = "train+val",
        input4mips_data_dir: Optional[str] = DATA_DIR,
        cmip6_data_dir: Optional[str] = DATA_DIR,
        output_save_dir: Optional[str] = DATA_DIR,
        climate_models: List[str] = ["NorESM2-LM", "EC-Earth3-Veg-LR"],
        num_ensembles: Union[List[int], int] = 1,  # 1 for first ensemble, -1 for all
        scenarios: Union[List[str], str] = ["ssp126", "ssp370", "ssp585"],
        historical_years: Union[Union[int, str], None] = "1950-1955",
        out_variables: Union[str, List[str]] = "pr",
        in_variables: Union[str, List[str]] = [
            "BC_sum",
            "SO2_sum",
            "CH4_sum",
            "CO2_sum",
        ],
        seq_to_seq: bool = True,
        seq_len: int = 12,
        channels_last: bool = False,
        load_data_into_mem: bool = True,  # Keeping this true be default for now
        input_transform=None,  # TODO: implement
        input_normalization="z-norm",  # TODO: implement
        output_transform=None,
        output_normalization="z-norm",
        val_split: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.index_manager = index_manager
        self.output_save_dir = output_save_dir
        self.channels_last = channels_last
        self.load_data_into_mem = load_data_into_mem

        if isinstance(in_variables, str):
            in_variables = [in_variables]
        if isinstance(out_variables, str):
            out_variables = [out_variables]
        if isinstance(scenarios, str):
            scenarios = [scenarios]
        self.num_in=len(in_variables)
        self.num_out=len(out_variables)

    
        # remap in variables / out vars to input4mip and CMIP
        # than in final get item, map back
        in_variables_im, out_variables_im, x_indexes, y_indexes = map_variables_targetmip(in_variables, out_variables)
        
        
        self.x_indexes=x_indexes
        self.y_indexes=y_indexes
        

        self.scenarios = scenarios

        if isinstance(num_ensembles, int):
            self.index_manager.num_ensembles = [num_ensembles] * len(climate_models)
        else:
            self.index_manager.num_ensembles = num_ensembles

        if isinstance(years, int):
            self.years = years
        else:
            self.years = self.get_years_list(
                years, give_list=True
            )  # Can use this to split data into train/val eg. 2015-2080 train. 2080-2100 val.

        if historical_years is None:
            self.historical_years = []
        elif isinstance(historical_years, int):
            self.historical_years = historical_years
        else:
            self.historical_years = self.get_years_list(
                historical_years, give_list=True
            )  # Can use this to split data into train/val eg. 2015-2080 train. 2080-2100 val.

        self.n_years = (
            len(self.years) + len(self.historical_years)
            if "historical" in self.scenarios
            else len(self.years)
        )

        # we need to create a mapping from climate model to respective input4mips ds (because openburning specs might differ)

        ds_kwargs = dict(
            scenarios=scenarios,
            years=self.years,
            historical_years=self.historical_years,
            channels_last=channels_last,
            mode=mode,
            output_save_dir=output_save_dir,
            seq_to_seq=seq_to_seq,
            seq_len=seq_len,
            data_dir=input4mips_data_dir,
        )

        # compute val/train indexes based on fraction
        # First set to test, but generally it will be set to either Val or Training first and only if it is not set at all, it will be set to to test automatically
        self.mode = "test"
        self.val_indexes=[]
        self.reset_val_index=None
        self.train_indexes=np.arange(self.index_manager.total_length)
        self.reset_train_index=0


    def set_mode(self, train: bool = False, indexes=None,reset_index=None, test = False):
        if test: 
            if train: 
                raise Exception("Train parametere should not be True when Test is True!")
            pass
        elif train:
            log.info("Setting to train.")
            self.mode='train'
            if ((indexes is not None) and (reset_index is not None)): 
                self.train_indexes = indexes
                self.reset_train_index = reset_index
        else:
            log.info("Setting to val.")
            self.mode='val'
            if ((indexes is not None) and (reset_index is not None)): 
                self.val_indexes = indexes
                self.reset_val_index = reset_index

    def get_years_list(self, years: str, give_list: Optional[bool] = False):
        """
        Get a string of type 20xx-21xx.
        Split by - and return min and max years.
        Can be used to split train and val.

        """
        if len(years) != 9:
            log.warn(
                "Years string must be in the format xxxx-yyyy eg. 2015-2100 with string length 9. Please check the year string."
            )
            raise ValueError
        splits = years.split("-")
        min_year, max_year = int(splits[0]), int(splits[1])

        if give_list:
            return np.arange(min_year, max_year + 1, step=1)
        return min_year, max_year

    
    def __getitem__(self, index):  # Dict[str, Tensor]):
        # check mode and reset the index
        if self.mode=='train':   
            index=self.train_indexes[index]
            # check that everything is reset if index is zero (starting again)

            if index == self.reset_train_index:
                self.index_manager.reset_index()
        
        elif self.mode=='val':

            index=self.val_indexes[index]
            # check that everything is reset if index is zero (starting again)
            if index == self.reset_val_index:
                self.index_manager.reset_index()
        elif self.mode=='test':
            # no need to reset in testing
            index=index
        else:
            raise ValueError
        
        self.index_manager.increment_cmip6_index(index)
      
        Y = self.index_manager.get_raw_ys(index)
        X = self.index_manager.get_raw_xs(index)



        # convert cmip model index to overall model num
        model_id = self.index_manager.climate_models[self.index_manager.cmip6_model_index]
        # return which climate model index the batch belongs to
        return X, Y, model_id

    def __str__(self):
        s = f" Super Emulator dataset: {len(self.index_manager.climate_models)} climate models with {self.index_manager.num_ensembles} ensemble members and {self.n_years} years used, with a total size of {len(self)} examples (in, out)."
        return s

    
    def __len__(self):
        if self.mode=='train' or self.mode=='test':
            return len(self.train_indexes)
        elif self.mode=='val':
            return len(self.val_indexes)
        # elif self.mode=='train+val':
        #     return self.get_initial_length()
        else:
            print("Unknown mode.", self.mode)
            raise ValueError



class CMIP6Dataset(ABC_Climate_Dataset):
    """
    Super CMIP6 Dataset. Containing data for multiple climate models and potentially multiple ensemble members.
    Iiterating overy every member-model pair.
    """

    def __init__(  # inherits all the stuff from Base
        self,
        data_dir,  # dir leading to a specific ensemble member
        years: Union[int, str],
        historical_years: Union[int, str],
        climate_model: str = "NorESM2-LM",
        scenarios: List[str] = ["ssp126", "ssp370", "ssp585"],
        variables: List[str] = ["pr"],
        mode: str = "train",
        output_save_dir: str = "",
        channels_last: bool = True,
        seq_to_seq: bool = True,
        seq_len: int = 12,
        *args,
        **kwargs,
    ):
        self.mode = mode
        self.output_save_dir = output_save_dir

        self.input_nc_files = []
        self.output_nc_files = []

        self.scenarios = scenarios
        self.channels_last = channels_last

        fname_kwargs = dict(
            climate_model=climate_model,
            ensemble_member=data_dir.split("/")[-1],
            years=f"{years[0]}-{years[-1]}",
            historical_years=f"{historical_years[0]}-{historical_years[-1]}",
            variables=variables,
            scenarios=scenarios,
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
            seq_len=seq_len,
        )

        # Check here if os.path.isfile($SCRATCH/data.npz) exists
        # if it does, use self._reload data(path)
        fname = self.get_save_name_from_kwargs(
            mode=mode, file="target", kwargs=fname_kwargs
        )
        if os.path.isfile(
            os.path.join(output_save_dir, fname)
        ):  # we first need to get the name here to test that...
            self.data_path = os.path.join(output_save_dir, fname)
            print("path exists, reloading")
            self.Data = self._reload_data(self.data_path)

            # Load stats and normalize
            stats_fname = self.get_save_name_from_kwargs(
                mode=mode, file="statistics", kwargs=fname_kwargs
            )

            stats = self.load_dataset_statistics(
                os.path.join(self.output_save_dir, stats_fname),
                mode=self.mode,
                mips="cmip6",
            )
            self.Data = self.normalize_data(self.Data, stats)

        else:
            # List of output files
            files_per_var = []
            for var in variables:
                output_nc_files = []

                for exp in scenarios:
                    if exp == "historical":
                        get_years = historical_years
                    else:
                        get_years = years
                    for y in get_years:
                        # we only have one ensemble here
                        var_dir = os.path.join(
                            data_dir, exp, var, f"{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}"
                        )
                        files = glob.glob(var_dir + f"/*.nc", recursive=True)
                        if len(files) == 0:
                            print(
                                "No files for this climate model, ensemble member, var, year ,scenario:",
                                climate_model,
                                data_dir.split("/")[-1],
                                var,
                                y,
                                exp,
                            )
                            print("Exiting! Please fix the data issue.")
                            exit(0)
                        # loads all years! implement splitting
                        output_nc_files += files
                files_per_var.append(output_nc_files)

            self.raw_data = self.load_into_mem(
                files_per_var,
                num_vars=len(variables),
                channels_last=channels_last,
                seq_to_seq=seq_to_seq,
                seq_len=seq_len,
            )
            if self.mode == "train" or self.mode == "train+val":
                stats_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=fname_kwargs
                )

                if os.path.isfile(fname):
                    print("Stats file already exists! Loading from memory.")
                    stats = self.load_statistics_data(stats_fname)
                    self.norm_data = self.normalize_data(self.raw_data, stats)

                else:
                    stat1, stat2 = self.get_dataset_statistics(
                        self.raw_data, self.mode, mips="cmip6"
                    )
                    stats = {"mean": stat1, "std": stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    save_file_name = self.write_dataset_statistics(stats_fname, stats)
                    print("WROTE STATISTICS", save_file_name)

                self.norm_data = self.normalize_data(self.raw_data, stats)

            elif self.mode == "test":
                stats_fname = self.get_save_name_from_kwargs(
                    mode="train+val", file="statistics", kwargs=fname_kwargs
                )
                save_file_name = os.path.join(self.output_save_dir, fname)
                stats = self.load_dataset_statistics(
                    stats_fname, mode=self.mode, mips="cmip6"
                )
                self.norm_data = self.normalize_data(self.raw_data, stats)

            self.data_path = self.save_data_into_disk(
                self.raw_data, fname, output_save_dir
            )

            self.copy_to_slurm(self.data_path)

            self.Data = self.norm_data
        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]

    def __len__(self):
        return len(self.Data)


class Input4MipsDataset(ABC_Climate_Dataset):
    """
    Loads all scenarios for a given var / for all vars
    """

    def __init__(  # inherits all the stuff from Base
        self,
        years: Union[int, str],
        historical_years: Union[int, str],
        data_dir: Optional[str] = DATA_DIR,
        variables: List[str] = ["BC_sum"],
        scenarios: List[str] = ["ssp126", "ssp370", "ssp585"],
        channels_last: bool = False,
        openburning_specs: Tuple[str] = ("no_fires", "no_fires"),
        mode: str = "train",
        output_save_dir: str = "",
        seq_to_seq: bool = True,
        seq_len: int = 12,
        *args,
        **kwargs,
    ):
        self.channels_last = channels_last

        self.mode = mode
        self.root_dir = os.path.join(data_dir, "inputs/input4mips")
        self.output_save_dir = output_save_dir
        self.input_nc_files = []
        self.output_nc_files = []

        self.scenarios = scenarios
        fname_kwargs = dict(
            years=f"{years[0]}-{years[-1]}",
            historical_years=f"{historical_years[0]}-{historical_years[-1]}",
            variables=variables,
            scenarios=scenarios,
            channels_last=channels_last,
            openburning_specs=openburning_specs,
            seq_to_seq=seq_to_seq,
            seq_len=seq_len,
        )

        historical_openburning, ssp_openburning = openburning_specs

        fname = self.get_save_name_from_kwargs(
            mode=mode, file="input", kwargs=fname_kwargs
        )

        # Check here if os.path.isfile($SCRATCH/data.npz) exists #TODO: check if exists on slurm
        # if it does, use self._reload data(path)
        if os.path.isfile(
            os.path.join(output_save_dir, fname)
        ):  # we first need to get the name here to test that...
            self.data_path = os.path.join(output_save_dir, fname)
            print("path exists, reloading")
            self.Data = self._reload_data(self.data_path)

            # Load stats and normalize
            stats_fname = self.get_save_name_from_kwargs(
                mode=mode, file="statistics", kwargs=fname_kwargs
            )
            stats = self.load_dataset_statistics(
                os.path.join(self.output_save_dir, stats_fname),
                mode=self.mode,
                mips="input4mips",
            )
            self.Data = self.normalize_data(self.Data, stats)

        else:
            files_per_var = []
            for var in variables:
                output_nc_files = []

                for exp in scenarios:
                    if exp == "historical":
                        get_years = historical_years
                    else:
                        get_years = years
                    for y in get_years:
                        var_dir = os.path.join(
                            self.root_dir,
                            exp,
                            var,
                            f"{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}",
                        )

                output_nc_files = []
                for exp in scenarios:
                    if var in NO_OPENBURNING_VARS:
                        filter_path_by = ""

                    elif exp == "historical":
                        filter_path_by = historical_openburning
                        get_years = historical_years
                    else:
                        filter_path_by = ssp_openburning
                        get_years = years

                    for y in get_years:
                        var_dir = os.path.join(
                            self.root_dir,
                            exp,
                            var,
                            f"{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}",
                        )
                        files = glob.glob(
                            var_dir + f"/**/*{filter_path_by}*.nc", recursive=True
                        )
                        output_nc_files += files
                files_per_var.append(output_nc_files)

            self.raw_data = self.load_into_mem(
                files_per_var,
                num_vars=len(variables),
                channels_last=self.channels_last,
                seq_to_seq=True,
                seq_len=seq_len,
            )  # we always want the full sequence for input4mips

            if self.mode == "train" or self.mode == "train+val":
                stats_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=fname_kwargs
                )

                if os.path.isfile(stats_fname):
                    print("Stats file already exists! Loading from mempory.")
                    stats = self.load_statistics_data(stats_fname)
                    self.norm_data = self.normalize_data(self.raw_data, stats)

                else:
                    stat1, stat2 = self.get_dataset_statistics(
                        self.raw_data, self.mode, mips="cmip6"
                    )
                    stats = {"mean": stat1, "std": stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    save_file_name = self.write_dataset_statistics(stats_fname, stats)

                self.norm_data = self.normalize_data(self.raw_data, stats)

            elif self.mode == "test":
                stats_fname = self.get_save_name_from_kwargs(
                    mode="train+val", file="statistics", kwargs=fname_kwargs
                )  # Load train stats cause we don't calculcate norm stats for test.
                stats = self.load_dataset_statistics(
                    stats_fname, mode=self.mode, mips="input4mips"
                )
                self.norm_data = self.normalize_data(self.raw_data, stats)

            self.data_path = self.save_data_into_disk(
                self.raw_data, fname, output_save_dir
            )

            self.copy_to_slurm(self.data_path)

            self.Data = self.norm_data

        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]

    def __len__(self):
        return len(self.Data)


if __name__ == "__main__":
    print("dataset_loaded")
    # ds = SuperClimateDataset(
    #     seq_to_seq=True,
    #     in_variables=["BC_sum", "SO2_sum", "CH4_sum"],
    #     scenarios=["ssp126", "ssp370"],
    #     climate_models=["MPI-ESM1-2-HR", "GFDL-ESM4", "NorESM2-LM"],
    #     seq_len=12,
    #     num_ensembles=1,
    #     years="2015-2021",
    #     channels_last=False,
    # )
    # # for (i,j) in ds:
    # # print("i:", i.shape)
    # # print("j:", j.shape)
    # print(ds)
    # print(len(ds))
    # # in_len, out_len = len(ds)
    # # print(in_len, out_len)

    # for i, (x, y, index) in enumerate(ds):
    #     print("iteration", i)
    #     #    print("x", x.shape)
    #     #    print("y", y.shape)
    #     print("model index", index)
