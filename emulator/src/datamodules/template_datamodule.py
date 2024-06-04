import logging
from typing import Optional, List, Callable

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader


import torch

from emulator.src.utils.utils import get_logger, random_split

log = get_logger()


class DummyDataModule(LightningDataModule):
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
        in_var_ids: List[str] = ["BC", "CO2", "CH4", "SO2"],
        out_var_ids: List[str] = ["pr", "tas"],
        train_years: List[str] = ["2020", "2030", "2040"],
        test_years: List[str] = [
            "2040"
        ],  # do we want to implement keeping only certain years for testing?
        seq_len: int = 10,
        seq_to_seq: bool = True,  # if true maps from T->T else from T->1
        lon: int = 32,
        lat: int = 32,
        num_levels: int = 1,
        channels_last: bool = True,  # wheather variables come last our after sequence lenght
        train_scenarios: List[str] = ["historical", "ssp126"],
        test_scenarios: List[str] = ["ssp345"],
        val_scenarios: List[str] = ["ssp119"],
        train_models: List[str] = ["NorESM5"],
        val_models: List[str] = ["NorESM5"],
        test_models: List[str] = ["CanESM5"],
        batch_size: int = 16,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = False,
        persistent_workers:bool = False,
        pin_memory: bool = False,
        load_train_into_mem: bool = False,
        load_test_into_mem: bool = False,
        load_valid_into_mem: bool = True,
        verbose: bool = True,
        seed: int = 11,
        # input_transform: Optional[AbstractTransform] = None,
        # normalizer: Optional[Normalizer] = None,
    ):
        """
        Args:
            in_var_ids List(str): Ids of input variables.
            out_var_ids: Lsit(str): Ids of output variables.
            seq_len (int): Lenght of the input sequence (in time).
            seq_to_seq (bool): If true maps from seq_len to seq_len else from seq_len to- 1.
            lon (int): Longitude of grid.
            lat (int): Latitude of grid.

            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloader arg for higher efficiency
            seed (int): Used to seed the validation-test set split, such that the split will always be the same.
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=["input_transform", "normalizer"])
        # self.input_transform = input_transform  # self.hparams.input_transform
        # self.normalizer = normalizer

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
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""

        train, test, val = None, None, None

        # here we need to figure out how to create the dataset / implement the dataset class
        # probably sequence of scenario-model pairs only load to memory what fits

        # assign to vars

        # Training set:
        if stage == "fit" or stage is None:
            self._data_train = train
        # Validation set
        if stage in ["fit", "validate", None]:
            self._data_val = val
        # Test sets:
        if stage == "test" or stage is None:
            self._data_test = test
        # Prediction set:
        if stage == "predict":
            # just choosing at random here
            self._data_predict = val

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch

    def _shared_dataloader_kwargs(self) -> dict:
        shared_kwargs = dict(
            num_workers=int(self.hparams.num_workers),
            pin_memory=self.hparams.pin_memory,
            persistent_workers= self.hparams.persistent_workers
        )
        return shared_kwargs

    def _shared_eval_dataloader_kwargs(self) -> dict:
        return dict(
            **self._shared_dataloader_kwargs(),
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
        )

    # Probably we also just want a list of Train Dataloaders not just a single one so we can swith sets in our memory
    # resulting tensors sizes:
    # x: (batch_size, sequence_length, lon, lat, in_vars) if channels_last else (batch_size, sequence_lenght, in_vars, lon, lat)
    # y: (batch_size, sequence_length, lon, lat, out_vars) if channels_last else (batch_size, sequence_lenght, out_vars, lon, lat)
    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        return (
            DataLoader(dataset=self._data_val, **self._shared_eval_dataloader_kwargs())
            if self._data_val is not None
            else None
        )

    def test_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(dataset=self._data_test, **self._shared_eval_dataloader_kwargs())
            for _ in self.test_set_names
        ]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(dataset=self._data_val, **self._shared_eval_dataloader_kwargs())
            if self._data_val is not None
            else None
        ]

    dm = DummyDataModule()
    dm.setup("fit")
