import logging
from typing import Optional, List, Callable, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from emulator.src.data.super_climate_dataset import SuperClimateDataset
import torch
from emulator.src.data.constants import (
    TEMP_RES,
    SEQ_LEN_MAPPING,
    LAT,
    LON,
    NUM_LEVELS,
    DATA_DIR,
)
from emulator.src.utils.utils import get_logger
#, random_split, random_split_super

log = get_logger()

class SuperClimateDataModule(LightningDataModule):
    """
    ----------------------------------------------------------------------------------------------------------
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        in_var_ids: Union[List[str], str] = ["BC_sum", "CO2_sum", "CH4_sum", "SO2_sum"],
        out_var_ids: Union[List[str], str] = ["pr", "tas"],
        train_years: Union[int, str] = "2000-2090",
        train_historical_years: Union[int, str] = "1950-1955",
        test_years: Union[
            int, str
        ] = "2090-2100",  # do we want to implement keeping only certain years for testing?
        val_split: float = 0.1,  # fraction of testing to split for valdation
        seq_to_seq: bool = True,  # if true maps from T->T else from T->1
        channels_last: bool = False,  # wheather variables come last our after sequence lenght
        train_scenarios: List[str] = ["historical", "ssp126"],
        test_scenarios: List[str] = ["ssp370", "ssp126"],
        train_models: List[str] = ["NorESM2-LM"],
        test_models: Union[List[str], None] = None,
        batch_size: int = 16,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        load_train_into_mem: bool = True,
        emissions_tracker:bool = False,
        load_test_into_mem: bool = True,
        verbose: bool = True,
        seed: int = 11,
        seq_len: int = SEQ_LEN_MAPPING[TEMP_RES],
        data_dir: Optional[str] = DATA_DIR,
        output_save_dir: Optional[str] = DATA_DIR,
        num_ensembles: int = 1,  # 1 for first ensemble, -1 for all
        lon: int = LON,
        lat: int = LAT,
        num_levels: int = NUM_LEVELS,
        name: str = "super_climate"
        # input_transform: Optional[AbstractTransform] = None,
        # normalizer: Optional[Normalizer] = None,
    ):
        """
        Args:
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloader arg for higher efficiency
            seed (int): Used to seed the validation-test set split, such that the split will always be the same.
        """
        super().__init__()

        self.train_models = train_models
        if test_models is None:
            self.test_models = train_models
        else:
            self.test_models = test_models

        # get unique models to have correct model numbers in all sets (train/val + test)
        # all_models = set(self.train_models + self.test_models)
        self.output_save_dir = output_save_dir
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=["input_transform", "normalizer"])
        # self.input_transform = input_transform  # self.hparams.input_transform
        # self.normalizer = normalizer

        self._data_train_val: Optional[SuperClimateDataset] = None
        # self._data_val: Optional[SuperClimateDataset] = None
        self._data_test: Optional[List[SuperClimateDataset]] = None
        self._data_predict: Optional[List[SuperClimateDataset]] = None
        self.test_set_names: Optional[List[str]] = [
            f"{scenario}_{model}"
            for scenario in test_scenarios
            for model in self.test_models
        ]
        print("Test Set names", self.test_set_names)
        self.emissions_tracker = self.hparams.emissions_tracker
        self._data_train = None
        self._data_val = None
        self._data_test = None
        self._data_predict = None
        self.log_text = get_logger()

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train_val, self._data_test."""
        # shared for all
        dataset_kwargs = dict(
            output_save_dir=self.hparams.output_save_dir,
            num_ensembles=self.hparams.num_ensembles,
            out_variables=self.hparams.out_var_ids,
            in_variables=self.hparams.in_var_ids,
            channels_last=self.hparams.channels_last,
            seq_to_seq=self.hparams.seq_to_seq,
            seq_len=self.hparams.seq_len,
            data_dir=self.hparams.data_dir,
            # input_transform = None, # TODO: implement
            # input_normalization = None, #TODO: implement
            # output_transform = None,
            # output_normalization = None,
        )

        # Train and Validation
        if stage in ["fit", "validate", None]:
            # create one big training dataset with all training scenarios
            #  then split it to assign data train and data val
            full_ds = SuperClimateDataset(
                years=self.hparams.train_years,
                historical_years=self.hparams.train_historical_years,
                mode="train+val",
                scenarios=self.hparams.train_scenarios,
                climate_models=self.train_models,
                load_data_into_mem=self.hparams.load_train_into_mem,
                val_split=self.hparams.val_split,
                **dataset_kwargs,
            )

            # datasets have internal representations of val indexs, set mode when calling dataloaders
            self._data_train = full_ds.set_mode(train=True)
            self._data_val = full_ds.set_mode(train=False)

        # Test sets:
        if stage == "test" or stage is None:
            self._data_test = [
                SuperClimateDataset(
                    years=self.hparams.test_years,
                    mode="test",
                    scenarios=test_scenario,
                    climate_models=[test_model],
                    load_data_into_mem=self.hparams.load_test_into_mem,
                    val_split=0,  # no split in testing
                    **dataset_kwargs,
                )
                for test_scenario in self.hparams.test_scenarios
                for test_model in self.test_models
            ]

        # Prediction set:
        if stage == "predict":
            print("Prediction Set not yet implemented. Using Test Set.")
            self._data_predict = self._data_test

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch

    def _shared_dataloader_kwargs(self) -> dict:
        shared_kwargs = dict(
            num_workers=int(self.hparams.num_workers),
            pin_memory=self.hparams.pin_memory,
        )
        return shared_kwargs

    def _shared_eval_dataloader_kwargs(self) -> dict:
        return dict(
            **self._shared_dataloader_kwargs(),
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
        )

    # x: (batch_size, sequence_length, lon, lat, in_vars) if channels_last else (batch_size, sequence_lenght, in_vars, lon, lat)
    # y: (batch_size, sequence_length, lon, lat, out_vars) if channels_last else (batch_size, sequence_lenght, out_vars, lon, lat)
    def train_dataloader(self):
        # self._data_train_val.set_mode(train=True)
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        # self._data_train_val.set_mode(train=False)
        return (
            DataLoader(dataset=self._data_val, **self._shared_eval_dataloader_kwargs())
            if self._data_val is not None
            else None
        )

    def test_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(dataset=ds_test, **self._shared_eval_dataloader_kwargs())
            for ds_test in self._data_test
        ]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(
                dataset=self._data_predict, **self._shared_eval_dataloader_kwargs()
            )
            if self._data_predict is not None
            else None
        ]


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
