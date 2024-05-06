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

from emulator.src.utils.utils import get_logger, all_equal
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


class SuperClimateDataset(torch.utils.data.Dataset):
    """
    Class to efficiently load data for multi-model multi-ensemble experiments.
    Input4MIPS data is only loaded once (per desired fire-model as specified by the climate models)

    """
    def __init__(
        self,
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

        self.output_save_dir = output_save_dir
        self.channels_last = channels_last
        self.load_data_into_mem = load_data_into_mem

        if isinstance(in_variables, str):
            in_variables = [in_variables]
        if isinstance(out_variables, str):
            out_variables = [out_variables]
        if isinstance(scenarios, str):
            scenarios = [scenarios]

        self.scenarios = scenarios
        self.climate_models = climate_models

        if isinstance(num_ensembles, int):
            self.num_ensembles = [num_ensembles] * len(climate_models)
        else:
            self.num_ensembles = num_ensembles

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
        self.openburning_specs = [
            OPENBURNING_MODEL_MAPPING[climate_model]
            if climate_model in AVAILABLE_MODELS_FIRETYPE
            else OPENBURNING_MODEL_MAPPING["other"]
            for climate_model in climate_models
        ]

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

        # creates list of input4mips (one per unique openburning spec)
        self.input4mips_ds = dict()
        for spec in set(self.openburning_specs):
            self.input4mips_ds[spec] = Input4MipsDataset(
                variables=in_variables, openburning_specs=spec, **ds_kwargs
            )

        # we create one CMIP6 dataset per model-ensemble member pair for easier iteration
        self.cmip6_ds_model = []
        self.cmip6_model_index = 0
        self.cmip6_member_index = 0
        self.input4mips_shift = 0

        # switch out data directory
        ds_kwargs.pop("data_dir")
        for climate_model, num_ensembles, openburning_specs in zip(
            self.climate_models, self.num_ensembles, self.openburning_specs
        ):
            cmip6_ds_model_member = []
            # get ensemble dir list
            root_dir = os.path.join(cmip6_data_dir, "outputs/CMIP6")
            if isinstance(climate_model, str):
                root_dir = os.path.join(root_dir, climate_model)

            if num_ensembles == 1:
                ensembles = os.listdir(root_dir)
                ensemble_dir = [
                    os.path.join(root_dir, ensembles[0])
                ]  # Taking first ensemble member
            else:
                ensemble_dir = []
                ensembles = os.listdir(root_dir)
                for i, folder in enumerate(ensembles):
                    ensemble_dir.append(
                        os.path.join(root_dir, folder)
                    )  # Taking multiple ensemble members
                    if i == (num_ensembles - 1):
                        break

            for em in ensemble_dir:
                # create on ds for each ensemble member
                cmip6_ds_model_member.append(
                    CMIP6Dataset(
                        data_dir=em,
                        climate_model=climate_model,
                        openburning_specs=openburning_specs,
                        variables=out_variables,
                        **ds_kwargs,
                    )
                )

            self.cmip6_ds_model.append(cmip6_ds_model_member)

        # for index shift we need the cum sum of the lengths
        lengths_model = []
        for model in self.cmip6_ds_model:
            lengths_member = []
            for member in model:
                lengths_member.append(member.length)
            lengths_model.append(lengths_member)

        lengths_model = np.asarray(lengths_model)
        # roll by one to get the correct shift
        lengths_model = np.roll(lengths_model, 1)
        lengths_model[0][0] = 0
        index = np.cumsum(lengths_model).reshape(lengths_model.shape)
        self.index_shifts = index
        # compute val/train indexes based on fraction
        total_length = self.get_initial_length()
        if mode=='test':
            # no split in testing
            print("no split in testing")
            self.val_indexes=[]
            self.reset_val_index=None
            self.mode="test"
            self.train_indexes=np.arange(total_length)
            self.reset_train_index=0
        else:
            self.val_indexes=np.sort(np.random.choice(total_length, int(np.round(val_split*total_length)), replace=False))
            self.reset_val_index=self.val_indexes[0]
            self.train_indexes=np.delete(np.arange(total_length), self.val_indexes)
            self.reset_train_index=self.train_indexes[0]
            print("val indexes", self.val_indexes, "reset", self.reset_val_index)
            print("train_indexes", self.train_indexes, "reset", self.reset_train_index)
            self.mode="train" # by default use train indexes


    def set_mode(self, train: bool = False):
        if train:
            log.info("Setting to train.")
            self.mode='train'
        else:
            log.info("Setting to val.")
            self.mode='val'
        return copy.deepcopy(self)

    # this operates variable vise now...
    def load_into_mem(
        self,
        paths: List[List[str]],
        num_vars,
        channels_last=True,
        seq_to_seq=True,
        seq_len=12,
    ):  # -> np.ndarray():
        array_list = []
        for vlist in paths:
            temp_data = xr.open_mfdataset(
                vlist, concat_dim="time", combine="nested"
            ).compute()  # .compute is not necessary but eh, doesn't hurt
            temp_data = temp_data.to_array().to_numpy()
            array_list.append(temp_data)
        temp_data = np.concatenate(array_list, axis=0)

        if seq_len != SEQ_LEN:
            print(
                "Choosing a sequence length greater or lesser than the data sequence length."
            )
            new_num_years = int(
                np.floor(temp_data.shape[1] / seq_len / len(self.scenarios))
            )

            # divide by scenario num and seq len, round
            # multiply with scenario num an dseq len to get correct shape
            new_shape_one = new_num_years * len(self.scenarios)
            assert (
                new_shape_one * seq_len > temp_data.shape[1]
            ), f"New sequence length {seq_len} greater than available years {temp_data.shape[1]}!"
            print(
                f"New sequence length: {seq_len} Dropping {temp_data.shape[1]-(new_shape_one*seq_len)} years"
            )
            temp_data = temp_data[:, : (new_shape_one * seq_len), :]

        else:
            new_shape_one = int(temp_data.shape[1] / seq_len)

        temp_data = temp_data.reshape(
            num_vars, new_shape_one, seq_len, LON, LAT
        )  # num_vars, num_scenarios*num_remainding_years, seq_len,lon,lat)

        if seq_to_seq == False:
            temp_data = temp_data[:, :, -1, :, :]  # only take last time step
            temp_data = np.expand_dims(temp_data, axis=2)

        if channels_last:
            temp_data = temp_data.transpose((1, 2, 3, 4, 0))
        else:
            temp_data = temp_data.transpose((1, 2, 0, 3, 4))

        return temp_data  # (years*num_scenarios, seq_len, vars, 96, 144)

    def save_data_into_disk(
        self, data: np.ndarray, fname: str, output_save_dir: str
    ) -> str:
        np.savez(os.path.join(output_save_dir, fname), data=data)
        return os.path.join(output_save_dir, fname)

    """
        def get_save_name_from_kwargs(self, mode:str, file:str,kwargs: Dict):
            fname =""
                
            for k in kwargs:
                if isinstance(kwargs[k], List):
                    fname+=f"{k}_"+"_".join(kwargs[k])+'_'
                else:
                    fname+=f"{k}_{kwargs[k]}_"

            if file == 'statistics':
                fname +=  '_' + file + '.npy'
            else:
                fname += mode + '_' + file + '.npz'
            print(fname)
            return fname
        """

    def get_save_name_from_kwargs(self, mode: str, file: str, kwargs: Dict):
        fname = ""

        if file == "statistics":
            # only cmip 6
            if "climate_model" in kwargs:
                fname += kwargs["climate_model"] + "_"
            if "num_ensembles" in kwargs:
                fname += str(kwargs["num_ensembles"]) + "_"
            # all
            fname += (
                "_".join(kwargs["variables"]) + "_"
            )  # + '_' + kwargs['input_normalization']
            # fname +=  '_' + file + '.npy'
            # print(fname)

        else:
            for k in kwargs:
                if isinstance(kwargs[k], List):
                    fname += f"{k}_" + "_".join(kwargs[k]) + "_"
                else:
                    fname += f"{k}_{kwargs[k]}_"

        if file == "statistics":
            fname += mode + "_" + file + ".npy"
        else:
            fname += mode + "_" + file + ".npz"

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

    def get_dataset_statistics(self, data, mode, type="z-norm", mips="cmip6"):
        if mode == "train" or mode == "train+val":
            if type == "z-norm":
                mean, std = self.get_mean_std(data)
                return mean, std
            elif type == "minmax":
                min_val, max_val = self.get_min_max(data)
                return min_val, max_val
            else:
                print("Normalizing of type {0} has not been implemented!".format(type))
        else:
            print("In testing mode, skipping statistics calculations.")

    def get_mean_std(self, data):
        # shape (examples, seq_len, vars, lon, lat) or DATA shape (examples, seq_len, vars, lon, lat)

        if self.channels_last:
            data = np.moveaxis(data, -1, 0)
        else:
            data = np.moveaxis(
                data, 2, 0
            )  # move vars to first axis, easier to calulate statistics
        vars_mean = np.mean(data, axis=(1, 2, 3, 4))
        vars_std = np.std(data, axis=(1, 2, 3, 4))
        vars_mean = np.expand_dims(
            vars_mean, (1, 2, 3, 4)
        )  # Shape of mean & std (4, 1, 1, 1, 1)
        vars_std = np.expand_dims(vars_std, (1, 2, 3, 4))
        return vars_mean, vars_std

    def get_min_max(self, data):
        if self.channels_last:
            data = np.moveaxis(data, -1, 0)
        else:
            data = np.moveaxis(
                data, 2, 0
            )  # move vars to front, easier to calulate statistics
        vars_max = np.max(data, axis=(1, 2, 3, 4))
        vars_min = np.min(data, axis=(1, 2, 3, 4))
        vars_max = np.expand_dims(
            vars_max, (1, 2, 3, 4)
        )  # Shape of mean & std (vars, 1, 1, 1, 1)
        vars_min = np.expand_dims(vars_min, (1, 2, 3, 4))
        return vars_min, vars_max

    def normalize_data(self, data, stats, type="z-norm"):
        # Only implementing z-norm for now
        # z-norm: (data-mean)/(std + eps); eps=1e-9
        # min-max = (v - v.min()) / (v.max() - v.min())

        print("Normalizing data...")
        if self.channels_last:
            data = np.moveaxis(
                data, -1, 0
            )  # vars from last to 0 (num_vars, years, seq_len, lon, lat)
        else:
            data = np.moveaxis(
                data, 2, 0
            )  # DATA shape (years, seq_len, num_vars, 96, 144) -> (num_vars, years, seq_len, 96, 144)

        norm_data = (data - stats["mean"]) / (stats["std"])

        if self.channels_last:
            norm_data = np.moveaxis(norm_data, 0, -1)
        else:
            norm_data = np.moveaxis(
                norm_data, 0, 2
            )  # Switch back to (years, seq_len, num_vars, 96, 144)

        return norm_data

    def write_dataset_statistics(self, fname, stats):
        np.save(os.path.join(self.output_save_dir, fname), stats, allow_pickle=True)
        return os.path.join(self.output_save_dir, fname)

    def load_dataset_statistics(self, fname, mode, mips):
        if "train_" in fname:
            fname = fname.replace("train", "train+val")
        elif "test" in fname:
            fname = fname.replace("test", "train+val")

        stats_data = np.load(
            os.path.join(self.output_save_dir, fname), allow_pickle=True
        ).item()

        return stats_data

    def increment_cimp6_index(self):
        # if exceeded member reset member index to 0
        if self.cmip6_member_index + 1 >= len(
            self.cmip6_ds_model[self.cmip6_model_index]
        ):
            # resetting member index
            self.cmip6_member_index = 0

            # if exceeded models rerest model index to 0
            if self.cmip6_model_index + 1 >= len(self.cmip6_ds_model):
                self.cmip6_model_index = 0

            else:
                # increment model index
                self.cmip6_model_index += 1
        else:
            # increment member index
            self.cmip6_member_index += 1

    def model_index_to_spec(self, index):
        # get correct openburning spec
        return self.openburning_specs[index]

    def __getitem__(self, index):  # Dict[str, Tensor]):
     
        # check mode and reset the index
        if self.mode=='train':   
            index=self.train_indexes[index]
            # check that everything is reset if index is zero (starting again)
        
            if index == self.reset_train_index:
                
                self.cmip6_model_index = 0
                self.cmip6_member_index = 0
          
        elif self.mode=='val':
            index=self.val_indexes[index]
            # check that everything is reset if index is zero (starting again)
            if index == self.reset_val_index:

                self.cmip6_model_index = 0
                self.cmip6_member_index = 0
        elif self.mode=='test':
            # no need to reset in testing
            index=index
        else:
            print("Unknown Mode. Must be 'train' or 'val'")
            print(self.model)
            raise ValueError

      
        # get climate model index
        # shift indexd (for models and members)
        # deal with multiple models
        # if current iteration longer than the models ds, increment model index
        if (
            index - self.index_shifts[self.cmip6_model_index][self.cmip6_member_index]
        ) > self.cmip6_ds_model[self.cmip6_model_index][
            self.cmip6_member_index
        ].length - 1:
            self.increment_cimp6_index()

        # access data in input4mips and cmip6 datasets
        # get correct input4mips set
        input4mips_spec_index = self.model_index_to_spec(self.cmip6_model_index)

        # for input4mips we need an index shift (dependent on num ensembles and num models)
        input4mips_index = (
            index - self.index_shifts[self.cmip6_model_index][self.cmip6_member_index]
        )

        raw_Xs = self.input4mips_ds[input4mips_spec_index][input4mips_index]
        raw_Ys = self.cmip6_ds_model[self.cmip6_model_index][self.cmip6_member_index][
            input4mips_index
        ]

        if not self.load_data_into_mem:
            X = raw_Xs
            Y = raw_Ys
        else:
            # TO-DO: Need to write Normalizer transform and To-Tensor transform
            # Doing norm and to-tensor per-instance here.
            # X_norm = self.input_transforms(self.X[index])
            # Y_norm = self.output_transforms(self.Y[index])
            X = raw_Xs
            Y = raw_Ys

        # convert cmip model index to overall model num
        model_id = self.climate_models[self.cmip6_model_index]
        # return which climate model index the batch belongs to
        return X, Y, model_id

    def __str__(self):
        s = f" Super Emulator dataset: {len(self.climate_models)} climate models with {self.num_ensembles} ensemble members and {self.n_years} years used, with a total size of {len(self)} examples (in, out)."
        return s

    def get_initial_length(self):
        # len is length of ds times length of each individual
        in_lengths = [
            self.input4mips_ds[spec].length for spec in self.openburning_specs
        ]
        print("Val/Train all_equal test")

        assert all_equal(
            in_lengths
        ), f"input4mip datasets do not have the same length for each openburning spec!"

        for i, m in enumerate(self.cmip6_ds_model):
            # lengths per model member pair
            out_lengths = [ds.length for ds in m]

            # cmip must be num_ensemble members times input4mips

            assert in_lengths[0] * self.num_ensembles[i] == np.sum(
                out_lengths
            ), f"CMIP6 must be num_ensembles times the length of input4mips. Got {np.sum(out_lengths)} and {in_lengths[0]}"

        # total length includes ensemble members so taking output length
        return np.sum(out_lengths) * len(self.cmip6_ds_model)
    
    def __len__(self):
        if self.mode=='train' or self.mode=='test':
            return len(self.train_indexes)
        elif self.mode=='val':
            return len(self.val_indexes)
        else:
            print("Unknown mode.", self.mode)
            raise ValueError



class CMIP6Dataset(SuperClimateDataset):
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

                if os.path.isfile(stats_fname):
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


class Input4MipsDataset(SuperClimateDataset):
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


if __name__ == "__main__":
    ds = SuperClimateDataset(
        seq_to_seq=True,
        in_variables=["BC_sum", "SO2_sum", "CH4_sum"],
        scenarios=["ssp126", "ssp370"],
        climate_models=["MPI-ESM1-2-HR", "GFDL-ESM4", "NorESM2-LM"],
        seq_len=12,
        num_ensembles=1,
        years="2015-2021",
        channels_last=False,
    )
    # for (i,j) in ds:
    # print("i:", i.shape)
    # print("j:", j.shape)
    print(ds)
    print(len(ds))
    # in_len, out_len = len(ds)
    # print(in_len, out_len)

    for i, (x, y, index) in enumerate(ds):
        print("iteration", i)
        #    print("x", x.shape)
        #    print("y", y.shape)
        print("model index", index)
