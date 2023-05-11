import hydra

import numpy as np

from omegaconf import DictConfig
from pytorch_lightning import LightningModule

class BaseModel(LightningModule):
    """ Abstract template class for all NN based emulators.
    Each model that inherits from BaseModel must implement the __init__ and
    forward method. Functions provided here can be overriden if needed!
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """
    # TODO data configs
    # TODO normalization / transformation configs and more
    def __init__(self,
                 data_config: DictConfig = None,
                 optimizer: Optional[DictConfig] = None,
                 scheduler: Optional[DictConfig] = None,
                 name: str = "",
                 verbose: bool = True,
                 ):
        super().__init__()
        raise NotImplementedError()
