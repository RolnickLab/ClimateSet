from typing import Sequence, Optional, Dict, Union, List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

#from esem import gp_model #ClimateBench GP
import segmentation_models_pytorch as smp
from torch.autograd import Variable
import numpy as np
#import gpytorch # gaussion process alternative?

from emulator.src.core.models.basemodel import BaseModel

class GaussianProcess(BaseModel):
    """
    Problem: climatebench baseline very hard to integrat into pytorch lightning - will need to overwrite forward and backward / train_step different + how to do batch learning?
    needs x and y for initialization....
    """
    def _init__(self):

        pass

    def forward(self, x:Tensor)->Tensor:

        return None

 
class RandomForest(BaseModel):
    def _init__(self):

        pass

    
    def forward(self, x:Tensor)->Tensor:
        return None

    

class CNNLSTM_ClimateBench(BaseModel):
    """ As converted from tf to torch, adapted from ClimateBench.
        Predicts single time step only #TODO we wanna change that do we? # tODO: documentation

        Original code below: 

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
        from tensorflow.keras.regularizers import l2
        
        
        cnn_model = Sequential()
        cnn_model.add(Input(shape=(slider, 96, 144, 4)))
        cnn_model.add(TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'), input_shape=(slider, 96, 144, 4)))
        cnn_model.add(TimeDistributed(AveragePooling2D(2)))
        cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))
        cnn_model.add(LSTM(25, activation='relu'))
        cnn_model.add(Dense(1*96*144))
        cnn_model.add(Activation('linear'))
        cnn_model.add(Reshape((1, 96, 144)))"""

    def __init__(self,
        lon,
        lat,
        in_var_ids,
        out_var_ids, 
        num_conv_filters: int= 20,
        lstm_hidden_size: int = 25,
        num_lstm_layers: int = 1,
        seq_to_seq: bool = True,
        seq_len: int = 10,
        dropout: float= 0.0,
        channels_last=True,
        datamodule_config: DictConfig = None,
        *args, **kwargs,


        ):

        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        self.save_hyperparameters()

        
        self.lon=lon
        self.lat=lat
        self.num_input_vars=len(in_var_ids)
        self.num_output_vars=len(out_var_ids)
        self.channels_last=channels_last
        self.seq_len=seq_len
        if seq_to_seq:
            self.out_seq_len=seq_len
        else: self.out_seq_len=1
        if datamodule_config is not None:
            if datamodule_config.get('channels_last') is not None:
                channels_last = datamodule_config.get('channels_last')
        self.channels_last = channels_last

        self.model=torch.nn.Sequential(
            #nn.Input(shape=(slider, width, height, num_input_vars)),
            TimeDistributed(nn.Conv2d(in_channels=self.num_input_vars, out_channels=num_conv_filters, kernel_size=(3, 3), padding='same')), # we might need to permute because not channels last ?
            nn.ReLU(),#, input_shape=(slider, width, height, num_input_vars)),
            TimeDistributed(nn.AvgPool2d(2)),
            #TimeDistributed(nn.AdaptiveAvgPool1d(())), ##nGlobalAvgPool2d(), does not exist in pytorch
            TimeDistributed(nn.AvgPool2d((int(lon/2), int(lat/2)))),
            nn.Flatten(start_dim=2),
            nn.LSTM(input_size=num_conv_filters, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True), # returns tuple and complete sequence
            extract_tensor(seq_to_seq), # ignore hidden and cell state
            nn.ReLU(),
            nn.Linear(in_features=lstm_hidden_size, out_features=self.num_output_vars*lon*lat), 
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, X):
        x=X
        if self.channels_last:
            x=x.permute((0,1,4,2,3)) # torch con2d expects channels before height and witdth
        
        x=self.model(x)
        x=torch.reshape(x, (X.shape[0], self.out_seq_len, self.num_output_vars, self.lon, self.lat))
        if self.channels_last:
            x=x.permute((0,1,3,4,2))

        x=x.nan_to_num()
        return x



class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step" 
    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*args)
        else:
            #only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]   
            out = self.module(*[x.view(bs*seq_len, *x.shape[2:]) for x in args], **kwargs)
            out_shape = out.shape
            return out.view(bs, seq_len,*out_shape[1:])

    def low_mem_forward(self, *args, **kwargs):                                           
        "input x with shape:(bs,seq_len,channels,width,height)"
        #print("args", args[0].size())
        #print("kwargs", kwargs)
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out,dim=self.tdim)
    def __repr__(self):
        return f'TimeDistributed({self.module})'


class extract_tensor(nn.Module):
    def __init__(self, seq_to_seq) -> None:
        super().__init__()
        self.seq_to_seq=seq_to_seq
    """ Helper Module to only extract output of a LSTM (ignore hidden and cell states)"""
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        #print("tensor in extarct", tensor.size(), seq_to_seq)
        if not(self.seq_to_seq):
            tensor = tensor[:, -1, :]
            #print("tensor last step", tensor.size(), seq_to_seq)
        return tensor


class UNet(BaseModel):
    """
    https://github.com/elena-orlova/SSF-project
    """
    def __init__(self,
            in_var_ids : List[str],
            out_var_ids: List[str],
            longitude: int = 32,
            latitude: int = 32,
            activation_function : Union[str, callable, None] = None , # activation after final convolution
            encoder_name = "vgg11", 
            datamodule_config: DictConfig = None,
            channels_last:  bool = True,
            seq_to_seq: bool = True,
            seq_len : int = 1,
            *args, **kwargs):
        
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)

        if datamodule_config is not None:
            if datamodule_config.get('channels_last') is not None:
                self.channels_last = datamodule_config.get('channels_last')
            if datamodule_config.get('lon') is not None:
                self.lon = datamodule_config.get('lon')
            if datamodule_config.get('lat') is not None:
                self.lat = datamodule_config.get('lat')
            if datamodule_config.get('seq_len') is not None:
                self.seq_len = datamodule_config.get('seq_len')
        else:
            self.lon =longitude
            self.lat =latitude
            self.channels_last=channels_last
            self.seq_len=seq_len
        self.save_hyperparameters()
        self.num_output_vars=len(out_var_ids)
        self.num_input_vars=len(in_var_ids)
        
        
        
        # determine padding -> lan and lot must be divisible by 32
        pad_lon=int((np.ceil(self.lon/32)*32)-(self.lon/32)*32)
        pad_lat=int((np.ceil(self.lat/32))*32-(self.lat/32)*32)
        print("lon", self.lon, "lat", self.lat)
        print("Padding", pad_lon, pad_lat)
        self.channels_last = channels_last


        self.model = torch.nn.Sequential(
                torch.nn.ConstantPad2d((pad_lat,0,pad_lon,0), 0), # zero padding along lon and lat
                TimeDistributed(smp.Unet(
                encoder_name=encoder_name,        
                encoder_weights=None,     
                in_channels=self.num_input_vars,           
                classes=self.num_output_vars,                      
                activation=activation_function)),
                torch.nn.Flatten(),
                torch.nn.Linear(in_features=(self.num_output_vars*(self.lon+pad_lon)*(self.lat+pad_lat)*self.seq_len),
                    out_features=(self.num_output_vars*self.lon*self.lat*self.seq_len)) # map back to original size
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    
    def forward(self, x:Tensor)->Tensor:

        # x: (batch_size, sequence_length, lon, lat, in_vars) if channels_last else (batch_size, sequence_lenght, in_vars, lon, lat)
        if self.channels_last:
            x=x.permute((0,1,4,2,3)) # torch con2d expects channels before height and witdth
        # if images width not divisible by 
        x=self.model(x)
        x=x.reshape((-1, self.seq_len, self.num_output_vars, self.lon, self.lat))
        if self.channels_last:
            x=x.permute((0,1,3,4,2))
        x=x.nan_to_num()

        if not(self.hparams.seq_to_seq):
            x=x[:,-1,:]
            x=torch.unsqueeze(x,1)

        # returns (batch_size, sequence_length/1, lon, lat, out_vars) if channels_last else (batch_size, sequence_lenght, out_vars, lon, lat)
        return x
    
    

if __name__=="__main__":
    
    in_vars=["CO2", "BC", "CH4", "SO2"]
    out_vars=['pr', 'tas']
    in_time=2
    #lead_time=3
    lon=96
    lat=144
    batch_size=2
    num_con_layers=20
    num_lstm_layers=25
    channels_last=False
    seq_to_seq=False
    num_in_vars=len(in_vars)
    num_out_vars=len(out_vars)

    if channels_last:
        dummy=torch.rand(size=(batch_size, in_time, lon, lat, num_in_vars)).cuda()
    else:
        dummy=torch.rand(size=(batch_size, in_time, num_in_vars, lon, lat)).cuda()
   
    unet = UNet(in_vars, out_vars, seq_to_seq=seq_to_seq, seq_len=in_time, longitude=lon, latitude=lat,channels_last=channels_last)
    #lstm = CNNLSTM_ClimateBench(in_var_ids=in_vars, out_var_ids=out_vars, seq_len=in_time, seq_to_seq=seq_to_seq, channels_last=channels_last, lon=lon, lat=lat)

 
    out=unet(dummy)
    print("Out", out.size())
    #out=lstm(dummy)
    #print("Out", out.size())