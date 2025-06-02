import abc
import torch
import torch.nn as nn
import logging
import gpytorch

from pytorch_lightning.utilities import rank_zero_only

# import problems from utils
def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def diff_max_min(x, dim):
    return torch.max(x, dim=dim) - torch.min(x, dim=dim)


log = get_logger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLL(nn.Module):
    """
    Marginal log likelihood: loss used for the Variational Gaussian Process
    """

    def __init__(self, gp_model, train_y):
        self.mll = gpytorch.mlls.VariationalELBO(
            gp_model.likelihood, gp_model.model, num_data=train_y.size(0)
        )

    def forward(self, pred, y):
        return -self.mll(pred, y)

class MSELoss(nn.Module): 
    def __init__(self, reduction: str = "none", mask=None):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, y):
        return self.mse(pred, y)
    
class RMSELoss(nn.Module):
    def __init__(self, reduction: str = "none", mask=None):
        super().__init__()
        self.mask = mask

        if reduction == "none":
            self.reduction_fn = None
        elif reduction == "mean":
            self.reduction_fn = torch.mean
        elif reduction == "sum":
            self.reduction_fn = torch.sum()
        else:
            raise ValueError(f"Reduction type {reduction} not supported.")

        self.mse = nn.MSELoss(reduction="none")  # mean over all dimensions

    def forward(self, pred, y):
        error = self.mse(pred, y)

        # TODO this one is not tested
        if self.mask is not None:
            error = (
                error.mean(dim=1) * self.mask
            ).sum() / self.mask.sum()

        if self.reduction_fn is not None:
            error = self.reduction_fn(error)

        # apply root on overall error
        error = torch.sqrt(error)

        return error

# Parent class with latitude weights and some other functions
class ClimateSetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    # weighting to account for decreasing grid-cell area towards poles
    def get_latitude_weights(self, lat_size: int) -> torch.Tensor:
        """ Returns latitude weights for a given number of data points along the latitude axis (y.shape[-2]).
        The weights are 1 at the equator and decrease towards the poles. 

        Parameters:
            lat_size (int): How many latitude data points should be considered
        Returns:
            torch.Tensor: Latitude weights
        """
        # we are not using -90, 90, but a small offset to make sure none of the datapoints get weights of 0
        lats = torch.linspace(-89.75, 89.75, lat_size)
        weights = torch.cos((torch.pi * lats) / 180)
        weights = weights.unsqueeze(-1)
        return weights
    
    def weighted_global_mean(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """ Get a weighted global mean of a tensor.
        Parameters:
            x (torch.Tensor): Values that are to be averaged
            weights (torch.Tensor): Latitude weights for x
        Returns:
            torch.Tensor: Single number representing the global mean
        """
        # sum_lat(sum_lon(x * weights)) / N_lat * N_lon
        # i.e.: sum(sum(x * weights)) / (96 * 144)
        return torch.mean(x * weights, dim=(-2, -1)) # dims order does not matter
    
    def check_lat_lon(self, pred: torch.Tensor, y: torch.Tensor):
        """ Functions that checks if latitude and longitude is behaving as expected.
        Parameters:
            pred (torch.Tensor): Predictions
            y (torch.Tensor): Targets
        """
        # Expected shape: [4, 12, 96, 144] -> [batch, time, latitude, longitude]
        if (pred.shape[-1] == 1) or (y[-1].shape == 1):
            raise ValueError("Loss function: Last dimension (values/channels) must be squeezed away")
        
        if (pred.shape[-1] < pred.shape[-2]) or (y.shape[-1] < y.shape[-2]):
            raise ValueError("There are more latitude than longitude grid cells. Check if you swapped longitude and latitude.")


class NRMSELoss_s_ClimateBench(ClimateSetLoss):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weighting to account for decreasing grid size towards the poles.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_lat_lon(pred, y)
        weights = self.get_latitude_weights(y.shape[-2])
        weights = weights.to(device)

        # nrmses = sqrt((weights * (pred_mean_t - y_mean_n_t)**2)_mean_s) / ((weights*y)_mean_s)_mean_t_n
        # n is for the different ensemble members in the target t. We don't have that here.
        # we need to calculate the mean over the time dimension (1), but also over the batch dimension (0) [climatebench doesn't need to do that]
        # so our adapted nrmse is :
        # nrmse = sqrt((weights * (pred_mean_b_t - y_mean_b_t)**2)_mean_s) / ((weights*y)_mean_s)_mean_b_t
        # with mean_s being the weighted global mean
        nrmse_s = torch.sqrt(
            self.weighted_global_mean(
                (pred.mean(dim=(0, 1)) - y.mean(dim=(0, 1))) ** 2, weights
            )
        ) / self.weighted_global_mean(y, weights).mean(dim=(0, 1))

        return nrmse_s

class NRMSELoss_g_ClimateBench(ClimateSetLoss):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weighting to account for decreasing grid size towards the pole.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_lat_lon(pred, y)
        weights = self.get_latitude_weights(y.shape[-2])
        weights = weights.to(device)

        # nrmseg = sqrt(
        #   (((weights * x_mean) - (weights * y_mean))**2)_mean_t_b 
        # ) / (weights*y)_mean_t_b
        # we are meaning over the batches at the same time when averaging over the temporal scale
        nrmse_g = (
            torch.sqrt(
                (
                    (self.weighted_global_mean(pred, weights)
                    - self.weighted_global_mean(y, weights)) ** 2
                ).mean(dim=(0, 1))
            )
            / self.weighted_global_mean(y, weights).mean(dim=(0, 1))
        )
        return nrmse_g

class NRMSELoss_ClimateBench(ClimateSetLoss):
    """
    Combination of global weighted and spatially weighted nrmse.
    """
    def __init__(self, alpha: int = 5):
        super().__init__()

        self.nrmse_g = NRMSELoss_g_ClimateBench()
        self.nrmse_s = NRMSELoss_s_ClimateBench()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_lat_lon(pred, y)
        nrmseg = self.nrmse_g(pred, y)
        nrmses = self.nrmse_s(pred, y)
        nrmse = nrmses + self.alpha * nrmseg
        return nrmse

class LLWeighted_RMSELoss_WeatherBench(ClimateSetLoss):
    """
    Weigthed RMSE taken from Weather Bench.
    Weighting to account for decreasing grid sizes towards the pole.

    rmse = mean over forecasts and time of torch.sqrt( mean over lon lat L(lat_j)*)MSE(pred, y)
    weights = cos(latitude)/cos(latitude).mean()
    """
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_lat_lon(pred, y)
        weights = self.get_latitude_weights(y.shape[-2])
        weights = weights.to(device)
        weights = weights.to(device)

        #rmse_before = torch.sqrt(torch.mean(weights * self.mse(pred, y), dim=(-2, -1))).mean()
        rmse = torch.mean(torch.sqrt(torch.mean(weights * ((pred - y)**2), dim=([-2, -1]))))

        return rmse

class LLweighted_MSELoss_Climax(ClimateSetLoss):
    """
    Latitude weighted mean squared error taken from ClimaX.
    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.
    """
    def __init__(self, mask=None):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.mask = mask

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_lat_lon(pred, y)
        mse = self.mse(pred, y)
        weights = self.get_latitude_weights(y.shape[-2])
        weights = weights.to(device)

        # how they create the weights (does not work for us, results make no sense)
        # if self.deg2rad:
        #     weights02 = torch.cos((torch.pi * torch.arange(y.shape[-2])) / 180)
        # else:
        #     weights02 = torch.cos(torch.arange(y.shape[-2]))

        # # ClimaX creates weird weights by dividing them by the mean 
        # #this leads to the climax rmse and mse to be the exact same like the unweighted mse / rmse
        #mean_weights = weights02 / weights02.mean()

        if self.mask is not None:
            error = (mse * weights * self.mask).sum() / self.mask.sum()
        else:
            error = (mse * weights).mean()

        return error


class LLweighted_RMSELoss_Climax(ClimateSetLoss):
    """
    Latitude weighted root mean squared error taken from ClimaX.
    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.
    """
    def __init__(self, mask=None):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.mask = mask

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Latitude is expected to be on position -2
        """
        self.check_lat_lon(pred, y)
        mse = self.mse(pred, y) # [batch, time, lat, lon]
        
        weights = self.get_latitude_weights(y.shape[-2])
        weights = weights.to(device)

        # ClimaX creates weird weights by dividing them by the mean 
        # this leads to the climax rmse and mse to be the exact same like the unweighted mse / rmse
        #weights = weights / weights.mean() # ignored in this code

        if self.mask is not None:
            raise NotImplementedError("Masking is not supported in the loss functions anymore.")
        
        # rmse for each month, and each batch
        error = torch.sqrt(torch.mean(mse * weights, dim=(-1, -2)))
        # mean over all months and batch samples
        error = error.mean()

        return error


if __name__ == "__main__":
    batch_size = 16
    out_time = 12
    lat = 96
    lon = 144
    dummy = torch.rand(size=(batch_size, out_time, lat, lon))#.cuda()
    targets = torch.rand(size=(batch_size, out_time, lat, lon))#.cuda()

    # targets = torch.ones(size=(batch_size, out_time, lat, lon))
    # dummy = targets + 0.1

    reduction = "mean"
    mse = MSELoss(reduction=reduction)
    rmse = RMSELoss(reduction=reduction)

    nrmse_g = NRMSELoss_g_ClimateBench()
    nrmse_s = NRMSELoss_s_ClimateBench()
    nrmse = NRMSELoss_ClimateBench()

    llrmse_wb = LLWeighted_RMSELoss_WeatherBench()

    llmse_cx = LLweighted_MSELoss_Climax()
    llrmse_cx = LLweighted_RMSELoss_Climax()
    
    loss = mse(dummy, targets)
    print("MSE loss", loss, loss.size())
    # np_dummy = dummy.cpu().detach().numpy()
    # np_targets = targets.cpu().detach().numpy()

    loss = rmse(dummy, targets)
    print("RMSE loss", loss, loss.size())

    loss = nrmse_s(dummy, targets)
    print("CB nrmse s loss", loss, loss.size())

    loss = nrmse_g(dummy, targets)
    print("CB nrmse g loss", loss, loss.size())

    loss = nrmse(dummy, targets)
    print("CB nrmseloss", loss, loss.size())

    loss = llrmse_wb(dummy, targets)
    print("WB rmse loss", loss, loss.size())

    loss = llmse_cx(dummy, targets)
    print("CX mse loss", loss, loss.size())

    loss = llrmse_cx(dummy, targets)
    print("CX rmse loss", loss, loss.size())


# REFACTOR
# - weight function should be one function

# Same tests needed for metrics

# Shape tests at the end of the whole pipeline??


# How weights were created before:

        # weights = (
        #     torch.cos(torch.arange(y.shape[-2])) / torch.cos(torch.arange(y.shape[-2]))
        # ).mean()
        # weights = weights.to(device)

##### OLD CODE #####
        # mse = self.mse(pred, y)
        # # lattitude weights
        # if self.deg2rad:
        #     weights = torch.cos((torch.pi * torch.arange(y.shape[-1])) / 180)
        # else:
        #     weights = torch.cos(torch.arange(y.shape[-1]))
        # # they normalize the weights first
        # weights = weights / weights.mean()
        # weights = weights.to(device)
        # if self.mask is not None:
        #     error = (mse * weights * self.mask).sum() / self.mask.sum()
        # else:
        #     error = (mse * weights).mean()
        # error = torch.sqrt(error)
        # return error
        ##### END OF OLD CODE #####