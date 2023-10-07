import hydra

import torch
import torch.nn as nn
import torch.optim as optim

from emulator.src.utils.utils import to_DictConfig, get_logger
from emulator.src.core.models.basemodel import BaseModel
from emulator.src.core.models.multihead_decoder import MultiHeadDecoder


log_text = get_logger()


class DecoderWrapper(BaseModel):
    def __init__(self, model, multihead_decoder, channels_last=True, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.model = model
        self.multihead_decoder = multihead_decoder
        self.channels_last = channels_last

    def forward(self, x, model_num):
        # model num: (batch_size) (can be different for each batch item)
        out = self.model(x)

        if self.channels_last:
            out = out.permute((0, 1, 4, 2, 3))
       
        out = self.multihead_decoder(out, model_num)

        if self.channels_last:
            out = out.permute((0, 1, 3, 4, 2))
        out = out.nan_to_num()
        
        return out

    '''
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers for training process.
        Calls the BaseModel method, but overwrite with an Adam optimizer which
        filters out parameters with requires_grad=False. Allows for freezing
        and unfreezing of parameters.
        """
        self.log_text.info("in configure otpimizer decoder wrapper")
        opt_lr_dict = super().configure_optimizers()
        self.log_text.info(opt_lr_dict)
        optim_kwargs = {k: v for k, v in self.hparams.optimizer.items() if k not in ['name', '_target_']}
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), **optim_kwargs)
        opt_lr_dict["optimizer"] = optimizer
        return opt_lr_dict
    '''