from typing import Sequence, Optional, Dict, Union, List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

#from esem import gp_model #ClimateBench GP
import segmentation_models_pytorch as smp
from torch.autograd import Variable

#import gpytorch # gaussion process alternative?




from emulator.src.core.models.basemodel import BaseModel

class GaussianProcess(BaseModel):
    """
    Problem: climatebench baseline very hard to integrat into pytorch lightning - will need to overwrite forward and backward / train_step different + how to do batch learning?
    needs x and y for initialization....
    """
    def _init__(self):

        pass

    def forward(self, X:Tensor)->Tensor:

        return None

 
class RandomForest(BaseModel):
    def _init__(self):

        pass

    
    def forward(self, X:Tensor)->Tensor:
        return None

    

class ConvLSTMCell(nn.Module):
    """
    adapted from: https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
    belonging to the paper: " Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" by Shi et al.
    """
    def __init__(self, 
        hidden_channels,
        kernel_size,
        input_channels=None,
        output_channels=None 
        ):

        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0
        if input_channels is None:
            input_channels=hidden_channels
        if output_channels is None:
            output_channels=hidden_channels

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        print(f"x size {x.size()}, h size {h.size()}, c size {c.size()}")
        print(self.Wxi(x).size())
        print(self.Whi(h).size())
        print((c*self.Wci).size())
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        print(f"ci {ci.size} cf {cf.size}")
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, shape, layer_type=None):
        if layer_type=='first':
            hidden=self.input_channels
        elif layer_type=='last':
            hidden=self.output_channels
        else:
            hidden=self.hidden_channels

        print("init hidden:", hidden)

        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, self.input_channels, shape[0], shape[1]))#.cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, self.input_channels, shape[0], shape[1])),#.cuda()
                Variable(torch.zeros(batch_size, self.hidden_channels, shape[0], shape[1])))#.cuda())


class ConvLSTM(BaseModel):
    """
    adapted from: https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
    belonging to the paper: " Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" by Shi et al.
    
    in the original implementation, c and h were allowed to have different channel dimensions in a layer, do not see how this works as they have to be added at some point...
    implementing with same channels per layer now
    (overall a bit goofy)

    #TODO: fix channel adaptation
    """

    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, 
        input_channels, # num features in
        hidden_channels,
        output_channels, # num features out
        kernel_size, 
        num_hidden_layers=2,
        step=1, # predictive steps
        effective_step=[1]):

        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels#[input_channels] + hidden_channels
        self.hidden_channels = hidden_channels#hidden_channels + [output_channels]
        self.output_channels = output_channels#output_channels
        self.kernel_size = kernel_size
        self.num_layers = num_hidden_layers+1
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i==0:
                cell=ConvLSTMCell(hidden_channels=self.hidden_channels, input_channels=self.input_channels, kernel_size=self.kernel_size)
            elif i==(self.num_layers-1):
                 cell=ConvLSTMCell(hidden_channels=self.hidden_channels, output_channels=self.output_channels, kernel_size=self.kernel_size)
            else:
                cell = ConvLSTMCell(self.hidden_channels, self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, X: Tensor): # X of size (batch_size, time, lon, lat, num_vars)

        # first -> pass input time step per time step through the convnet
        internal_state = []
        bsize, in_time, height, width, num_feats = X.size()
 
        
        for i,step in enumerate(range(in_time)):
            # get new input per time step
            x = X[:,i,:,:,:]
            # layers expect channels to come second
            x = torch.permute(x, (0,3,1,2))
            print(f"Feeding step {step}, size {x.size()}")
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    #bsize, _, height, width, num_feats = x.size()
                    if i==0:
                        layer_type="first"
                    elif i==(self.num_layers-1):
                        layer_type="last"
                    else:
                        layer_type=None
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, layer_type=layer_type,
                                                             shape=(height, width)) # TODO: check output dimensionality!
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                print("h", h.size())
                print("c", c.size())
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            #if step in self.effective_step:
            #    outputs.append(x)

        # second -> use last state to predict desired time steps into the future

        internal_state_pred = []
        outputs = []
        for step in range(self.step):
            #x = X
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)

                if step==0:
                    # use latest internal state
                    internal_state_pred.append(internal_state[-1])

                # do forward
                (h, c) = internal_state_pred[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state_pred[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs  # only outputting states -> list with len number effective steps


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
        lead_time: int = 1, #so far must be same as input seq lenght
        dropout: float= 0.0,
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
        self.lead_time=lead_time
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
            extract_tensor(), # ignore hidden and cell state
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
        x=torch.reshape(x, (X.shape[0], self.lead_time, self.num_output_vars, self.lon, self.lat))
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
    """ Helper Module to only extract output of a LSTM (ignore hidden and cell states)"""
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        return tensor


class UNet(BaseModel):
    """
    https://github.com/elena-orlova/SSF-project
    """
    def __init__(self,
            in_var_ids : List[str],
            out_var_ids: List[str],
            activation_function : Union[str, callable, None] = None , # activation after final convolution
            encoder_name = "vgg11", 
            datamodule_config: DictConfig = None,
            channels_last:  bool = True,
            *args, **kwargs):
        
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        self.save_hyperparameters()
        num_output_vars=len(out_var_ids)
        num_input_vars=len(in_var_ids)

        self.model = TimeDistributed(smp.Unet(
            encoder_name=encoder_name,        
            encoder_weights=None,     
            in_channels=num_input_vars,           
            classes=num_output_vars,                      
            activation=activation_function))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)



    
    def forward(self, x:Tensor)->Tensor:
        if self.hparams.channels_last:
            x=x.permute((0,1,4,2,3)) # torch con2d expects channels before height and witdth
        x=self.model(x)
        if self.hparams.channels_last:
            x=x.permute((0,1,3,4,2))
        x=x.nan_to_num()
        return x
    
    

if __name__=="__main__":
    
    in_vars=4
    out_vars=['pr', 'tas']
    in_time=10
    lead_time=3
    lon=32
    lat=32
    batch_size=16
    num_con_layers=20
    num_lstm_layers=25
    channels_last=False

    if channels_last:
        dummy=torch.rand(size=(batch_size, in_time, lon, lat, in_vars)).cuda()
    else:
        dummy=torch.rand(size=(batch_size, in_time, in_vars, lon, lat)).cuda()
    #dummy=torch.rand(size=(batch_size,in_time,in_vars,lon,lat))
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dummy.to(device)
    targets=torch.rand(size=(batch_size, lead_time, lon, lat, len(out_vars)))#.cuda()

    #lstm=ConvLSTM(input_channels=in_vars, hidden_channels=4, output_channels=out_vars, kernel_size=3, step=out_time)
    #print(lstm)

    #lstm(dummy)

    #lstm =CNNLSTM_ClimateBench(num_input_vars=in_vars, out_var_ids['pa','ts'], lon=lon, lat=lat, lead_time=in_time)
    unet = UNet(in_vars, out_vars)
    dummy=torch.rand(size=(batch_size, in_time, in_vars, lon, lat)).cuda()
    out=unet(dummy)
    print("Out", out.size())