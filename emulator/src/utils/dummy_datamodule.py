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
            out_var_ids: List[str] = ['pr', 'tas'],
            seq_len: int = 10,
            lead_time: int = 1,
            lon: int = 32,
            lat: int = 32,
            num_levels: int = 1,
            channels_last: bool = True,
            size: int = 1000,
            test_split: float = 0.2,
            val_split: float = 0.2,
            #input_transform: Optional[AbstractTransform] = None,
            #normalizer: Optional[Normalizer] = None,
            batch_size: int = 16,
            eval_batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            load_train_into_mem: bool = False,
            load_test_into_mem: bool = False,
            load_valid_into_mem: bool = True,
            test_main_dataset: bool = True,
            verbose: bool = True,
            seed: int = 11,
            test_set_names: List[str] = ["main", "second"]
    ):
        """
        Args:
            in_var_ids List(str): Ids of input variables.
            #num_output_vars (int): Number of output variables.
            out_var_ids: Lsit(str): Ids of output variables.
            seq_len (int): Lenght of the input sequence (in time).
            lead_time (int): Number of timesteps to predict. 
            lon (int): Longitude of grid.
            lat (int): Latitude of grid.
            channels_last (int): If true, shape of tensors (batch_size, time, lon, lat, channels) else (batch_size, time, channels, lon, lat). Important for some torch layers.
            size (int): Size (num examples) of the dummy dataset. 
            test_split (float): Fraction of data to use for testing. 
            val_split (float): Fraction of data to use for evaluation. 
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloader arg for higher efficiency
            seed (int): Used to seed the validation-test set split, such that the split will always be the same.
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=["input_transform", "normalizer"])
        #self.input_transform = input_transform  # self.hparams.input_transform
        #self.normalizer = normalizer

        self._data_train = None
        self._data_val = None
        self._data_test = None
        self._data_predict = None
        self.log_text = get_logger()
        self.test_set_names=test_set_names

        if num_levels!=1:
            self.log_text.warn("Multiple pressure levels not yet implemented!")
            raise NotImplementedError

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""


        # create a big input and target tensor
        if self.hparams.channels_last:
            inputs=torch.rand(size=(self.hparams.size, self.hparams.seq_len, self.hparams.lon, self.hparams.lat, len(self.hparams.in_var_ids)))
            targets=torch.ones(size=(self.hparams.size, self.hparams.lead_time, self.hparams.lon, self.hparams.lat, len(self.hparams.out_var_ids)))
            
        else:
            inputs=torch.rand(size=(self.hparams.size, self.hparams.seq_len, len(self.hparams.in_var_ids), self.hparams.lon, self.hparams.lat))
            targets=torch.ones(size=(self.hparams.size, self.hparams.lead_time, len(self.hparams.out_var_ids), self.hparams.lon, self.hparams.lat))

        #targets={}
        #for var in self.hparams.out_var_ids:
        #    targets[var]=torch.rand(size=(self.hparams.size, self.hparams.seq_len, self.hparams.lon, self.hparams.lat)).cuda()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #inputs, targets = inputs.to(device), targets.to(device)
        inputs=inputs.cuda()
        targets=targets.cuda()
        dataset= torch.utils.data.TensorDataset(inputs, targets)
        fractions=[(1-(self.hparams.test_split+self.hparams.val_split)) ,self.hparams.test_split, self.hparams.val_split]    
        ds_list = random_split(dataset, lengths=fractions)
        train, test, val = ds_list

        # assign to vars

        # Training set:
        if stage == "fit" or stage is None:
            self._data_train = train
        # Validation set
        if stage in ['fit', 'validate', None]:
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
        shared_kwargs = dict(num_workers=int(self.hparams.num_workers), pin_memory=self.hparams.pin_memory)
        return shared_kwargs

    def _shared_eval_dataloader_kwargs(self) -> dict:
        return dict(**self._shared_dataloader_kwargs(), batch_size=self.hparams.eval_batch_size, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._data_val,
            **self._shared_eval_dataloader_kwargs()
        ) if self._data_val is not None else None

    def test_dataloader(self) -> List[DataLoader]:
        return [DataLoader(
            dataset=self._data_test,
            **self._shared_eval_dataloader_kwargs()
        ) for _ in self.test_set_names]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self._data_val,
            **self._shared_eval_dataloader_kwargs()
        ) if self._data_val is not None else None]




if __name__=="__main__":

    dm=DummyDataModule()
    dm.setup('fit')