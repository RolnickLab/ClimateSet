import torch
import torch.nn as nn
import logging 
import gpytorch

# somehow cannot import that..
#from emulator.src.utils.utils import get_logger, diff_max_min
from pytorch_lightning.utilities import rank_zero_only

# import problems from utils
def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def diff_max_min(x,dim):
    return torch.max(x,dim=dim)-torch.min(x,dim=dim)


log = get_logger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLL(nn.Module):
    """
    Marginal log likelihood: loss used for the Variational Gaussian Process
    """
    def __init__(self, gp_model, train_y):
        self.mll = gpytorch.mlls.VariationalELBO(gp_model.likelihood, gp_model.model, num_data=train_y.size(0))

    def forward(self, pred, y):
        return -self.mll(pred, y)


class RMSELoss(nn.Module):
    def __init__(self, reduction: str = 'none', mask=None):
        super().__init__()
        self.mask = mask
        
        if reduction=='none':
            self.reduction_fn = None
        elif reduction=='mean':
            self.reduction_fn = torch.mean
        elif reduction=='sum':
            self.reduction_fn = torch.sum()
        else:
            log.warn(f"Reduction type {reduction} not supported.")
            raise NotImplementedError
        
        self.mse = nn.MSELoss(reduction='none') # mean over all dimensions
        
    def forward(self,pred,y):
        
        error = torch.sqrt(self.mse(pred,y))
        if self.mask is not None:
            error = (error.mean(dim=1) * self.mask).sum() / self.mask.sum() #TODO: check
        
        if self.reduction_fn is not None:
            error = self.reduction_fn(error)
    
        return error 


class NRMSELoss_s_ClimateBench(nn.Module):
    """
    Spatial normalized weighted RMSE taken from Climate Bench. 
    Weigting to account for decreasing grid size towards the pole.
    """

    def __init__(self,  deg2rad: bool = True):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
         
        self.deg2rad=deg2rad

    def forward(self,pred,y): 
        
        # weighting to account for decreasing grid-cell area towards pole
        # lattitude weights
        if self.deg2rad:
            weights=torch.cos((torch.pi * torch.arange(y.shape[-1]))/180)
        else:
            weights=torch.cos(torch.arange(y.shape[-1]))
        weights=weights.to(device)
       
        # nrmses = sqrt((weights * (x_mean_t -y_mean_n_t)**2))_mean_s / ((weights*y)_mean_s)_mean_t_n
        # TODO: clarify with duncan why not mean over n with x..
        nrmse_s = torch.sqrt( self.weighted_global_mean( (pred.mean(dim=(0,1))-y.mean(dim=(0,1)) )**2,weights)) / self.weighted_global_mean(y,weights).mean(dim=(0,1))

        return nrmse_s

    def weighted_global_mean(self, x, weights):
        # weitghs * x summed over lon lat / lon+lat
       
        return torch.mean(x*weights, dim=(-1,-2))


class NRMSELoss_g_ClimateBench(nn.Module):
    """
    Spatial normalized weighted RMSE taken from Climate Bench. 
    Weigting to account for decreasing grid size towards the pole.
    """

    def __init__(self,  deg2rad: bool = True):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
         
        self.deg2rad=deg2rad 
        

    def forward(self,pred,y): 
        
        # weighting to account for decreasing grid-cell area towards pole
        # lattitude weights
        if self.deg2rad:
            weights=torch.cos((torch.pi * torch.arange(y.shape[-1]))/180)
        else:
            weights=torch.cos(torch.arange(y.shape[-1]))

        weights=weights.to(device)

        # nrmseg = sqrt(((x - ( (weights * y_mean_t)_mean_s)**2)_mean_t )  ) / ((weights*y)_mean_s)_mean_t_n
        denom = self.weighted_global_mean(y,weights).mean(dim=(0,1))
        
        # TODO: clarify with duncan when to mean over samples for predictions? before or after sqrt?
        nrmse_g = torch.sqrt( (self.weighted_global_mean(pred.mean(dim=0), weights) - self.weighted_global_mean(y.mean(dim=0), weights)**2).mean(dim=(0)) ) / denom
        
        return nrmse_g

    def weighted_global_mean(self, x, weights):
        # weitghs * x summed over lon lat / lon+lat
        return torch.mean(x*weights, dim=(-1,-2))

class NRMSELoss_ClimateBench(nn.Module):
    """
    Combination of global weighted and spatially weighted nrmse.

    """
    def __init__(self,  deg2rad:bool = True, alpha: int = 5):
        super().__init__()

        self.nrmse_g = NRMSELoss_g_ClimateBench(deg2rad)
        self.nrmse_s = NRMSELoss_s_ClimateBench(deg2rad)
        self.alpha=alpha

    def forward(self, pred, y):

        nrmseg=self.nrmse_g(pred, y)
        nrmses=self.nrmse_s(pred,y)
        nrmse = nrmses + self.alpha* nrmseg
        return nrmse



class LLWeighted_RMSELoss_WheatherBench(nn.Module):
  
    """
    Weigthed RMSE taken from Wheather Bench. 
    Weighting to account for decreasing grid sizes towards the pole.

    rmse = mean over forecasts and time of torch.sqrt( mean over lon lat L(lat_j)*)MSE(pred, y)
    weights = cos(latitude)/cos(latitude).mean()
    """
    def __init__(self):
        super().__init__()

        self.mse=nn.MSELoss(reduction='none')

    def forward(self,pred,y):

        weights = (torch.cos(torch.arange(y.shape[-1]))/torch.cos(torch.arange(y.shape[-1]))).mean()
        weights=weights.to(device)
        rmse = torch.sqrt(torch.mean(weights*self.mse(pred,y),dim=(-1,-2))).mean()

        return rmse

class LLweighted_MSELoss_Climax(nn.Module):
    """
    Latitude weighted mean squared error taken from ClimaX.
    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable. 
    If given a mask, normalized by sum of that.

    """
    def __init__(self,  deg2rad: bool = True, mask=None):
        super().__init__()

        self.mse=nn.MSELoss(reduction='none')
        self.deg2rad = deg2rad
        self.mask=mask

    def forward(self, pred, y):
    
        mse = self.mse(pred, y)
        
        # lattitude weights
        if self.deg2rad:
            weights=torch.cos((torch.pi * torch.arange(y.shape[-1]))/180)
        else:
            weights=torch.cos(torch.arange(y.shape[-1]))

        # they normalize the weights first
        weights = weights / weights.mean()
        weights=weights.to(device)
        if self.mask is not None:
            error= (mse * weights * self.mask).sum() / self.mask.sum()
        else:
            error= (mse*weights).mean()
    
        return error

class LLweighted_RMSELoss_Climax(nn.Module):
    """
    Latitude weighted root mean squared error taken from ClimaX.
    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable. 
    If given a mask, normalized by sum of that.
    """
    def __init__(self,  deg2rad: bool = True, mask=None):
        super().__init__()

        self.mse=nn.MSELoss(reduction='none')
        self.deg2rad = deg2rad
        
        self.mask=mask

    def forward(self, pred, y):
    
        mse = self.mse(pred, y)
        
        # lattitude weights
        if self.deg2rad:
            weights=torch.cos((torch.pi * torch.arange(y.shape[-1]))/180)
        else:
            weights=torch.cos(torch.arange(y.shape[-1]))

        # they normalize the weights first
        weights = weights / weights.mean()
        weights=weights.to(device)
        if self.mask is not None:
            error= (mse * weights * self.mask).sum() / self.mask.sum()
        else:
            error= (mse*weights).mean()

        error = torch.sqrt(error)

        return error

''' deprected 
class NRMSE_Loss(nn.Module):
    """ deprecated
    NRMSE: normalized RMSE
    Different options for normalization: mean, std, max-min, quantiles...
    Different options for aspect to normalize over: features, grid... 

    """

    def __init__(self, aspect: str = "grid", noramlization: str = "mean",  ):
        super().__init__()
        self.rmse = RMSELoss(reduction='none')
         
        self.aspect = aspect
        self.normalization = noramlization
         # if channels last shape (batch_size, seq_len, lon, lat, features) else (batch_size, seq_len, features, lon, lat)
       
        self.lat_idx=-1
        self.feat_idx=-3
        if aspect=='grid':
            self.dim=(self.lat_idx, self.lat_idx-1)
        elif aspect=='feature':
            self.dim=(self.feat_idx)
        else:
            log.warn(f"Aspect {aspect} not supported.")
            raise ValueError 

        if noramlization=="mean":
            self.normalization_fn=torch.mean
        elif noramlization=="std":
            self.normalization_fn=torch.std
        elif noramlization=="min_max":
            self.normalization_fn=diff_max_min
        else:
            log.warn(f"Normalization method {noramlization} not supported")
            raise ValueError



    def forward(self,pred,y):
       
        normalization_factor = self.normalization_fn(y, self.dim)
        nrmse = (self.rmse(pred, y).mean(dim=self.dim) / normalization_factor).mean()

        return nrmse
'''

if __name__=="__main__":

    batch_size=16
    out_time=10
    lon=32
    lat=64
    dummy=torch.rand(size=(batch_size, out_time, lon, lat)).cuda()
  
    targets=torch.rand(size=(batch_size, out_time, lon, lat)).cuda()

    reduction='mean'
    mse=torch.nn.MSELoss(reduction='mean')
    #rmse=RMSELoss(reduction=reduction)
    
    nrmse_g = NRMSELoss_g_ClimateBench()
    nrmse_s = NRMSELoss_s_ClimateBench()
    nrmse = NRMSELoss_ClimateBench()

    llrmse_wb = LLWeighted_RMSELoss_WheatherBench()

    llmse_cx = LLweighted_MSELoss_Climax()
    llrmse_cx = LLweighted_RMSELoss_Climax()

    loss=nrmse_g(dummy, targets)
    print("CB nrmse g loss", loss, loss.size())

    loss=nrmse_s(dummy, targets)
    print("CB nrmse s loss", loss, loss.size())

    loss=nrmse(dummy, targets)
    print("CB nrmseloss", loss, loss.size())
    

    loss=llrmse_wb(dummy, targets)
    print("WB rmse loss", loss, loss.size())

    loss=llmse_cx(dummy, targets)
    print("CX mse loss", loss, loss.size())


    loss=llrmse_cx(dummy, targets)
    print("CX nmse loss", loss, loss.size())

    


