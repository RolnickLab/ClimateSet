from typing import Any, Dict, List, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from emulator.src.core.models.climax.tokenized_vit_continuous import TokenizedViTContinuous
from emulator.src.core.models.basemodel import BaseModel
#from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
#from src.utils.metrics import (lat_weighted_acc, lat_weighted_mse,
#                               lat_weighted_mse_val, lat_weighted_nrmse,
#                               lat_weighted_rmse, mse)
from emulator.src.utils.pos_embed import (interpolate_channel_embed,
                                 interpolate_pos_embed)
from emulator.src.utils.utils import get_logger

from torchvision.transforms import transforms

from omegaconf import DictConfig

def get_region_info(lat_range, lon_range, lat, lon, patch_size):
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w
    }


class ClimaX(BaseModel):
    def __init__(
        self,
        climate_modeling: bool =True, # always True with us...
        time_history: int =1,
        lon: int = 128,
        lat: int = 256,
        patch_size: int =16,
        drop_path: float =0.1,
        drop_rate: float =0.1,
        learn_pos_emb: bool =False,
        in_vars: List[str] =['CO2',
          'SO2',
          'CH4',
          'BC'],
        out_vars: List[str] =["pr", "tas"], # is default vars for climate modelling
        channel_agg: str ="mean",
        embed_dim: int =1024,
        depth: int =24,
        decoder_depth: int =8,
        num_heads: int =16,
        mlp_ratio: float =4.0,
        init_mode: str ="xavier",
        freeze_encoder: bool =False,
        no_time_aggregation: bool = False, # not(seq_to_seq)
        datamodule_config: DictConfig = None,
        pretrained_path: str = None,
        region_info = None, # TODO: maybe later we could actually include that
        channels_last : bool = False,
        *args, **kwargs,
    ):
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        self.save_hyperparameters()

        self.log_text = get_logger(__name__)


        # get info from datamodule
        if datamodule_config is not None:
            if datamodule_config.get('out_var_ids') is not None:
                out_vars=datamodule_config.get('out_var_ids')
            if datamodule_config.get('in_var_ids') is not None:
                in_vars=datamodule_config.get('in_var_ids')
            if datamodule_config.get('channels_last') is not None:
                self.channels_last = datamodule_config.get('channels_last')
            if datamodule_config.get('lon') is not None:
                self.lon = datamodule_config.get('lon')
            if datamodule_config.get('lat') is not None:
                self.lat = datamodule_config.get('lat')
           
        else:
            self.lon =lon
            self.lat =lat
            self.channels_last=channels_last
            
        if climate_modeling:
            assert out_vars is not None
            self.out_vars = out_vars
        else:
            self.out_vars = in_vars
        
        img_size=[self.lon, self.lat]
        # create class
        self.model = TokenizedViTContinuous( 
            climate_modeling=climate_modeling,
            time_history=time_history,
            img_size=img_size, # grid size
            patch_size=patch_size,
            drop_path=drop_path,
            drop_rate=drop_rate,
            learn_pos_emb=learn_pos_emb,
            in_vars=in_vars,
            out_vars=out_vars, 
            channel_agg=channel_agg,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_mode=init_mode,
            freeze_encoder=freeze_encoder,
            time_aggregation=not(no_time_aggregation),  
            )


        if (pretrained_path is not None):
            if len(pretrained_path) > 0:
                self.load_mae_weights(pretrained_path)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        

    def load_mae_weights(self, pretrained_path):

        self.log_text.info("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))

        
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.model, checkpoint_model, new_size=self.model.img_size)
        # interpolate_channel_embed(checkpoint_model, new_len=self.model.channel_embed.shape[1])

        state_dict = self.state_dict()
        checkpoint_keys = list(checkpoint_model.keys())
        for k in checkpoint_keys:
            if self.model.climate_modeling:
                # replace the pretrained embedding layers and prediction heads, keep attention finetuning
                if 'token_embeds' in k or 'head' in k:
                    self.log_text.info(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
                    continue
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                self.log_text.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        self.log_text.info(msg)

    def forward(self, x):
        if self.channels_last:
            x=x.permute((0,1,4,2,3))

        lead_times = torch.zeros(x.shape[0]).to(x.device) # zero leadtimes for climate modelling task #TODO: remove?
        x = self.model.forward(x, lead_times)
        if self.channels_last:
            x=x.permute((0,1,3,4,2))
        x=x.nan_to_num()
        return x 

    def get_patch_size(self):
        return self.model.patch_size
  
"""
    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch
        pred_steps = 1
        pred_range = self.pred_range

        days = [int(pred_range / 24)]
        steps = [1]

        if self.model.climate_modeling:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse]
        else:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]

        all_loss_dicts, _ = self.model.rollout(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            region_info,
            pred_steps,
            metrics,
            self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
            clim=self.val_clim
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict
"""

if __name__=="__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClimaX(no_time_aggregation=True, channels_last=True, lon=32, lat=32).to(device)
    print(device)
    x, y = torch.randn(8, 5, 32, 32, 4).to(device), torch.randn(8, 5, 32, 32, 2).cuda()
    preds = model.forward(x)
    print(preds.shape)
