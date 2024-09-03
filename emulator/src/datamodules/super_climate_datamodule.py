import logging
import copy
from typing import Optional, List, Callable, Union, Dict
import os
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from emulator.src.data.super_climate_dataset import (SuperClimateDataset,CMIP6Dataset,Input4MipsDataset)
import torch
from emulator.src.data.constants import (
    TEMP_RES,
    SEQ_LEN_MAPPING,
    LAT,
    LON,
    NUM_LEVELS,
    DATA_DIR,
    OPENBURNING_MODEL_MAPPING,
    AVAILABLE_MODELS_FIRETYPE
)
from emulator.src.utils.utils import get_logger,all_equal
import numpy as np
#, random_split, random_split_super

log = get_logger()


class StateManager:
    """
    Manages the state of climate models and datasets for training and testing.

    Attributes:
        cmip6_model_index (int): Index of the current CMIP6 model.
        cmip6_member_index (int): Index of the current CMIP6 member.
        climate_models (list): List of climate models.
        num_ensembles (list): Number of ensembles for each climate model.
        ds_kwargs (dict): Keyword arguments for dataset creation.
        dir (str): Directory path for datasets.
        openburning_specs (list): Specifications for open burning for each model.
        cmip6_ds_model (list): CMIP6 datasets.
        input4mips_ds (dict): Input4MIPs datasets.
        index_shifts (ndarray): Shifts in indexes for dataset alignment.
        total_length (int): Total length of the dataset.
        val_indexes (ndarray): Indexes for validation data.
        train_indexes (ndarray): Indexes for training data.
        test_indexes (int): Index for test data.
        reset_val_index (int): Reset index for validation data.
        reset_train_index (int): Reset index for training data.
        reset_test_index (None): Reset index for test data.
    """
    
    def __init__(self, initial_model_index, initial_member_index, climate_models, out_var_ids, in_var_ids, ds_kwargs, dir, ensembles, mode="train", val_split=0.1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.cmip6_model_index = initial_model_index
        self.cmip6_member_index = initial_member_index
        self.climate_models = climate_models
        self.num_ensembles = self.create_num_ensembles(ensembles)
        self.ds_kwargs = ds_kwargs
        self.dir = dir
        
        self.openburning_specs = self.generate_openburning_specs(climate_models)
        self.cmip6_ds_model = self.create_cmip6_ds(out_var_ids, mode)
        self.input4mips_ds = self.create_input4mips_datasets(in_var_ids)
        self.index_shifts = self.calculate_index_shifts()
        
        self.split_datasets(val_split)

    def set_to_test(self):
        """Set the state manager to test mode."""
        self.val_indexes = []
        self.reset_val_index = None
        self.mode = "test"
        self.train_indexes = np.arange(self.total_length)
        self.reset_train_index = 0

    def create_num_ensembles(self, start_val):
        """Create the number of ensembles for each climate model."""
        if isinstance(start_val, int):
            return [start_val] * len(self.climate_models)
        return start_val

    def model_index_to_spec(self, index):
        """Get the openburning spec for a given model index."""
        return self.openburning_specs[index]
    
    def get_raw_xs(self, index):
        """Get raw input data (Xs) for a given index."""
        input4mips_spec_index = self.model_index_to_spec(self.cmip6_model_index)
        input4mips_index = index - self.index_shifts[self.cmip6_model_index][self.cmip6_member_index]
        return self.input4mips_ds[input4mips_spec_index][input4mips_index]

    def get_raw_ys(self, index):
        """Get raw output data (Ys) for a given index."""
        input4mips_index = index - self.index_shifts[self.cmip6_model_index][self.cmip6_member_index]
        return self.cmip6_ds_model[self.cmip6_model_index][self.cmip6_member_index][input4mips_index]

    def create_cmip6_ds(self, out_var_ids, mode):
        """Create CMIP6 datasets for each climate model and ensemble."""
        cmip6_ds_model = []
        for climate_model, num_ensembles, openburning_specs in zip(self.climate_models, self.num_ensembles, self.openburning_specs):
            cmip6_ds_model_member = self.create_cmip6_model_members(climate_model, num_ensembles, openburning_specs, out_var_ids)
            cmip6_ds_model.append(cmip6_ds_model_member)
        return cmip6_ds_model

    def create_cmip6_model_members(self, climate_model, num_ensembles, openburning_specs, out_var_ids):
        """Create CMIP6 model members for a given climate model."""
        cmip6_ds_model_member = []
        root_dir = os.path.join(self.dir, "outputs/CMIP6", climate_model)
        
        ensemble_dirs = self.get_ensemble_dirs(root_dir, num_ensembles)
        for em in ensemble_dirs:
            cmip6_ds_model_member.append(self.create_cmip6_dataset(em, climate_model, openburning_specs, out_var_ids))
        return cmip6_ds_model_member

    def create_cmip6_dataset(self, data_dir, climate_model, openburning_specs, out_var_ids):
        """Create a single CMIP6 dataset."""
        kwargs = self.ds_kwargs
        if("data_dir" in kwargs):
            kwargs.pop("data_dir")
        return CMIP6Dataset(
            data_dir = data_dir,
            climate_model=climate_model,
            openburning_specs=openburning_specs,
            variables=out_var_ids,
            **kwargs,
        )

    def get_ensemble_dirs(self, root_dir, num_ensembles):
        """Get the directories for ensemble members."""
        ensembles = os.listdir(root_dir)
        if num_ensembles == 1:
            return [os.path.join(root_dir, ensembles[0])]
        
        return [os.path.join(root_dir, folder) for i, folder in enumerate(ensembles) if i < num_ensembles]

    def generate_openburning_specs(self, climate_models):
        """Generate openburning specs for each climate model."""
        return [
            OPENBURNING_MODEL_MAPPING[climate_model] if climate_model in AVAILABLE_MODELS_FIRETYPE else OPENBURNING_MODEL_MAPPING["other"]
            for climate_model in climate_models
        ]

    def create_input4mips_datasets(self, in_var_ids):
        """Create Input4MIPs datasets."""
        input4mips_ds = {}
        for spec in set(self.openburning_specs):
            input4mips_ds[spec] = Input4MipsDataset(variables=in_var_ids, openburning_specs=spec, **self.ds_kwargs)
        return input4mips_ds

    def calculate_index_shifts(self):
        """Calculate shifts in indexes for dataset alignment."""
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
        return index
        
    def split_datasets(self, val_split):
        """Split the dataset into training and validation sets."""
        self.total_length = self.get_initial_length()
        self.val_indexes = np.sort(np.random.choice(self.total_length, int(np.round(val_split * self.total_length)), replace=False))
        self.reset_val_index = self.val_indexes[0]
        self.train_indexes = np.delete(np.arange(self.total_length), self.val_indexes)
        self.reset_train_index = self.train_indexes[0]
        self.test_indexes = self.total_length
        self.reset_test_index = None

    def get_initial_length(self):
        """Get the initial length of the dataset."""
        in_lengths = [self.input4mips_ds[spec].length for spec in self.openburning_specs]
        assert all(length == in_lengths[0] for length in in_lengths), "Input4MIPs datasets do not have the same length for each openburning spec!"

        for i, model in enumerate(self.cmip6_ds_model):
            out_lengths = [ds.length for ds in model]
            assert in_lengths[0] * self.num_ensembles[i] == np.sum(out_lengths), f"CMIP6 must be num_ensembles times the length of Input4MIPs. Got {np.sum(out_lengths)} and {in_lengths[0] * self.num_ensembles[i]}"

        return np.sum(out_lengths) * len(self.cmip6_ds_model)

    def find_interval(self, intervals, number):
        """Find the interval in which a number falls."""
        intervals = [interval[0] for interval in intervals]
        for i in range(len(intervals)):
            if i == len(intervals) - 1:
                if number >= intervals[i]:
                    return i
            else:
                if intervals[i] <= number < intervals[i + 1]:
                    return i
        return -1

    def increment_cmip6_index(self, index):
        """Increment the CMIP6 model and member index based on the given index."""
        self.cmip6_model_index = self.find_interval(self.index_shifts, index)
        if self.cmip6_member_index + 1 >= len(self.cmip6_ds_model[self.cmip6_model_index]):
            self.cmip6_member_index = 0
        else:
            self.cmip6_member_index += 1
        return self.cmip6_model_index, self.cmip6_member_index

    def get_indices(self):
        """Get the current indices of CMIP6 model and member."""
        return self.cmip6_model_index, self.cmip6_member

    def reset_index(self):
        """Reset the indices of CMIP6 model and member to 0."""
        self.cmip6_model_index = 0
        self.cmip6_member_index = 0



class SuperClimateDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform, and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        in_var_ids: Union[List[str], str] = ["BC_sum", "CO2_sum", "CH4_sum", "SO2_sum"],
        out_var_ids: Union[List[str], str] = ["pr", "tas"],
        train_years: Union[int, str] = "2000-2090",
        train_historical_years: Union[int, str] = "1950-1955",
        test_years: Union[int, str] = "2090-2100",
        val_split: float = 0.1,
        seq_to_seq: bool = True,
        channels_last: bool = False,
        train_scenarios: List[str] = ["historical", "ssp126"],
        test_scenarios: List[str] = ["ssp370", "ssp126"],
        train_models: List[str] = ["NorESM2-LM"],
        test_models: Union[List[str], None] = None,
        batch_size: int = 16,
        shuffle: bool = False,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        load_train_into_mem: bool = True,
        emissions_tracker: bool = False,
        load_test_into_mem: bool = True,
        verbose: bool = True,
        seed: int = 11,
        seq_len: int = SEQ_LEN_MAPPING[TEMP_RES],
        data_dir: Optional[str] = DATA_DIR,
        output_save_dir: Optional[str] = DATA_DIR,
        num_ensembles: int = 1,
        lat: int = LAT,
        lon: int = LON,
        num_levels: int = NUM_LEVELS,
        name: str = "super_climate"
    ):
        """
        Args:
            batch_size (int): Batch size for the training dataloader.
            eval_batch_size (int): Batch size for the test and validation dataloaders.
            num_workers (int): Dataloader arg for higher efficiency.
            pin_memory (bool): Dataloader arg for higher efficiency.
            seed (int): Used to seed the validation-test set split, such that the split will always be the same.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["input_transform", "normalizer"])
        
        # Setting up model lists
        self.train_models = train_models
        self.test_models = test_models if test_models is not None else train_models
        self.output_save_dir = output_save_dir

        # Internal data variables
        self._data_train_val: Optional[SuperClimateDataset] = None
        self._data_test: Optional[List[SuperClimateDataset]] = None
        self._data_predict: Optional[List[SuperClimateDataset]] = None

        self.test_set_names: Optional[List[str]] = [
            f"{scenario}_{model}"
            for scenario in test_scenarios
            for model in self.test_models
        ]
        
        self.emissions_tracker = self.hparams.emissions_tracker
        self.index_manager = self.create_index_manager()

        self._data_train = SuperClimateDataset(
            index_manager=self.index_manager,
            years=self.hparams.train_years,
            historical_years=self.hparams.train_historical_years,
            mode="train",
            scenarios=self.hparams.train_scenarios,
            climate_models=self.train_models,
            load_data_into_mem=self.hparams.load_train_into_mem,
            val_split=self.hparams.val_split,
            **self.dataset_kwargs(),
        )

        self.log_text = get_logger()

    def create_index_manager(self) -> StateManager:
        """
        Create the index manager used to manage data indices for training and testing.
        """
        years, historical_years = self.years_and_historical_years()
        ds_kwargs = self.dataset_kwargs()
        ds_kwargs.update({
            "scenarios": self.hparams.train_scenarios,
            "years": years,
            "historical_years": historical_years,
            "channels_last": self.hparams.channels_last,
            "mode": "train",
            "seq_to_seq": self.hparams.seq_to_seq,
            "seq_len": self.hparams.seq_len,
        })
    

        return StateManager(
            initial_model_index=0,
            initial_member_index=0,
            climate_models=self.train_models,
            out_var_ids=self.hparams.out_var_ids,
            in_var_ids=self.hparams.in_var_ids,
            ds_kwargs=ds_kwargs,
            dir=self.hparams.data_dir,
            ensembles=self.hparams.num_ensembles,
        )

    def years_and_historical_years(self):
        """
        Parse training and historical years from the provided hyperparameters.
        """
        years = self.parse_years(self.hparams.train_years)
        historical_years = self.parse_years(self.hparams.train_historical_years, default=[])
        return years, historical_years

    def parse_years(self, years: Union[int, str], default=None):
        """
        Helper function to parse years from a string or integer.
        """
        if years is None:
            return default
        if isinstance(years, int):
            return years
        return self.get_years_list(years, give_list=True)

    def get_years_list(self, years: str, give_list: Optional[bool] = False):
        """
        Convert a string representation of a year range into a list or tuple of years.
        """
        if len(years) != 9:
            log.warn(
                "Years string must be in the format xxxx-yyyy eg. 2015-2100 with string length 9. Please check the year string."
            )
            raise ValueError

        min_year, max_year = map(int, years.split("-"))
        return np.arange(min_year, max_year + 1, step=1) if give_list else (min_year, max_year)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data and set internal variables for different stages."""
        if stage in ["fit", "validate", None]:
            self._data_train.set_mode(train=True)

        if stage == "test" or stage is None:
            self.setup_test_data()

        if stage == "predict":
            self._data_predict = self._data_test

    def setup_test_data(self):
        """
        Setup the test datasets for the module.
        """
        self.index_manager.set_to_test()
        kwargs = self.dataset_kwargs()
        if("data_dir" in kwargs):
            kwargs.pop("data_dir")
        self._data_test = [
            SuperClimateDataset(
                index_manager=self.index_manager,
                years=self.hparams.test_years,
                data_dir=self.hparams.data_dir,
                scenarios=test_scenario,
                climate_models=[test_model],
                load_data_into_mem=self.hparams.load_test_into_mem,
                val_split=0,
                **kwargs,
            )
            for test_scenario in self.hparams.test_scenarios
            for test_model in self.test_models
        ]

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """
        Hook to apply any transformations before transferring the batch to the device.
        """
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook to apply any transformations after transferring the batch to the device.
        """
        return batch

    def _shared_dataloader_kwargs(self) -> dict:
        """
        Shared keyword arguments for all dataloaders.
        """
        return {
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "persistent_workers": self.hparams.persistent_workers,
        }

    def _shared_eval_dataloader_kwargs(self) -> dict:
        """
        Shared keyword arguments for evaluation dataloaders.
        """
        return {**self._shared_dataloader_kwargs(), "batch_size": self.hparams.eval_batch_size, "shuffle": False}

    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        self._data_train.set_mode(train=True, indexes=self.index_manager.train_indexes, reset_index=self.index_manager.reset_train_index)
        return DataLoader(
            dataset=copy.deepcopy(self._data_train),
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        self._data_train.set_mode(train=False, indexes=self.index_manager.val_indexes, reset_index=self.index_manager.reset_val_index)
        return DataLoader(
            dataset=copy.deepcopy(self._data_train),
            **self._shared_eval_dataloader_kwargs(),
        )

    def test_dataloader(self) -> List[DataLoader]:
        """
        Returns a list of test dataloaders.
        """
        return [DataLoader(dataset=ds_test, **self._shared_eval_dataloader_kwargs()) for ds_test in self._data_test]

    def predict_dataloader(self) -> List[DataLoader]:
        """
        Returns a list of predict dataloaders.
        """
        return [DataLoader(dataset=ds_predict, **self._shared_eval_dataloader_kwargs()) for ds_predict in self._data_predict] if self._data_predict else []

    def dataset_kwargs(self) -> Dict:
        """
        Shared keyword arguments for dataset initialization.
        """
        return {
            "output_save_dir": self.hparams.output_save_dir,
            "num_ensembles": self.hparams.num_ensembles,
            "out_var_ids": self.hparams.out_var_ids,
            "in_var_ids": self.hparams.in_var_ids,
            "channels_last": self.hparams.channels_last,
            "seq_to_seq": self.hparams.seq_to_seq,
            "seq_len": self.hparams.seq_len,
            "data_dir": self.hparams.data_dir
        }


if __name__ == "__main__":
    dm = SuperClimateDataModule(
        seq_to_seq=True,
        seq_len=12,
        in_var_ids=["BC_sum", "SO2_sum", "CH4_sum"],
        train_years="2015-2020",
        train_scenarios=["historical", "ssp370"],
        test_scenarios=["ssp370"],
        train_models=["MPI-ESM1-2-HR", "GFDL-ESM4", "NorESM2-LM"],
        channels_last=False,
    )
    dm.setup("fit")
    # dm.setup("test")
