import torch 
import pytest
import numpy as np

from emulator.src.core.metrics import *
from emulator.src.core.losses import *
from emulator.src.utils.metric_utils import get_latitude_weights_np

batch_size = 16
out_time = 12
lat = 96
lon = 144

# Define the seed value
seed = 42

# Set seed for PyTorch
torch.manual_seed(seed)
# Set seed for CUDA (if using GPUs)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
# Ensure deterministic behavior for PyTorch operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PRECISION_VALUE = 0.0001

@pytest.fixture 
def rand_targets():
    return torch.rand(size=(batch_size, out_time, lat, lon))

@pytest.fixture 
def rand_predics():
    return torch.rand(size=(batch_size, out_time, lat, lon))

@pytest.fixture
def ones_targets():
    return torch.ones(size=(batch_size, out_time, lat, lon))

@pytest.fixture
def ones_predics():
    return (torch.ones(size=(batch_size, out_time, lat, lon)) + 0.1)

################################################################
### compare numpy vs torch weights                           ###
################################################################
def test_weights_comparison(ones_predics):
    lat_size = int(ones_predics.shape[-2])
    parent_loss = ClimateSetLoss()
    torch_weights = parent_loss.get_latitude_weights(lat_size)
    numpy_weights = get_latitude_weights_np(lat_size)

    assert np.allclose(torch_weights.numpy(), numpy_weights, atol=PRECISION_VALUE)


################################################################
### compare if all metrics are the same like the loss values ###
################################################################
def test_mse_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = MSELoss(reduction="mean")
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = MSE(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = MSE(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)

def test_rmse_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = RMSELoss(reduction="mean")
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = RMSE(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = RMSE(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)

def test_nrmse_s_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = NRMSELoss_s_ClimateBench()
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = NRMSE_s_ClimateBench(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = NRMSE_s_ClimateBench(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)

def test_nrmse_g_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = NRMSELoss_g_ClimateBench()
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = NRMSE_g_ClimateBench(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = NRMSE_g_ClimateBench(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)

def test_nrmse_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = NRMSELoss_ClimateBench()
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = NRMSE_ClimateBench(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = NRMSE_ClimateBench(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)

def test_wb_rmse_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = LLWeighted_RMSELoss_WeatherBench()
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = LLWeighted_RMSE_WeatherBench(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = LLWeighted_RMSE_WeatherBench(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)

def test_cx_mse_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = LLweighted_MSELoss_Climax()
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = LLweighted_MSE_Climax(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = LLweighted_MSE_Climax(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)

def test_cx_rmse_metric(rand_predics, rand_targets, ones_predics, ones_targets):
    loss_error = LLweighted_RMSELoss_Climax()
    loss_ones = loss_error(ones_predics, ones_targets) 
    loss_rand = loss_error(rand_predics, rand_targets)

    metric_ones = LLweighted_RMSE_Climax(ones_predics.numpy(), ones_targets.numpy())
    metric_rand = LLweighted_RMSE_Climax(rand_predics.numpy(), rand_targets.numpy())

    assert metric_ones == pytest.approx(loss_ones, abs=PRECISION_VALUE)
    assert metric_rand == pytest.approx(loss_rand, abs=PRECISION_VALUE)