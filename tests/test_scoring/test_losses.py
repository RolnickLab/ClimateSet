import torch
import pytest

from emulator.src.core.losses import *

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

PRECISION_VALUE = 0.0005

# TODO test channel issue

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
    return torch.ones(size=(batch_size, out_time, lat, lon)) + 0.1

def test_weights(ones_targets):
    lat_size = int(ones_targets.shape[-2])
    parent_loss = ClimateSetLoss()
    torch_weights = parent_loss.get_latitude_weights(lat_size)
    assert torch_weights[0] == pytest.approx(0.0044, abs=0.0001)
    assert torch_weights[-1] == pytest.approx(0.0044, abs=0.0001)
    assert torch_weights[0] == torch_weights[-1]
    assert torch_weights[int(lat_size/2)] == pytest.approx(1, abs=0.001)

def expected_loss(loss_obj, expected_loss_value, precision_threshold):
    assert loss_obj.item() == pytest.approx(expected_loss_value, abs=precision_threshold)
    assert loss_obj.shape == ()

def test_testing_data(rand_predics, rand_targets, ones_predics, ones_targets):
    """ Make sure the testing data has actually some error
    """
    # make sure it's not the same
    assert not torch.equal(rand_predics, rand_targets)
    assert not torch.equal(ones_predics, ones_targets)

def test_mse(rand_predics, rand_targets, ones_predics, ones_targets):
    reduction = "mean"
    error = MSELoss(reduction=reduction)
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.0100, PRECISION_VALUE)
    expected_loss(loss_rand, 0.1668, PRECISION_VALUE)

def test_rmse(rand_predics, rand_targets, ones_predics, ones_targets):
    reduction = "mean"
    error = RMSELoss(reduction=reduction)
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.1000, PRECISION_VALUE)
    expected_loss(loss_rand, 0.4084, PRECISION_VALUE)

def test_nrmse_s(rand_predics, rand_targets, ones_predics, ones_targets):
    error = NRMSELoss_s_ClimateBench()
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.1258, PRECISION_VALUE)
    expected_loss(loss_rand, 0.0739, PRECISION_VALUE)

def test_nrmse_g(rand_predics, rand_targets, ones_predics, ones_targets):
    error = NRMSELoss_g_ClimateBench()
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.1000, PRECISION_VALUE)
    expected_loss(loss_rand, 0.0081, PRECISION_VALUE)

def test_nrmse(rand_predics, rand_targets, ones_predics, ones_targets):
    error = NRMSELoss_ClimateBench()
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.6258, PRECISION_VALUE)
    expected_loss(loss_rand, 0.1142, 0.005)

def test_wb_rmse(rand_predics, rand_targets, ones_predics, ones_targets):
    error = LLWeighted_RMSELoss_WeatherBench()
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.0795, PRECISION_VALUE)
    expected_loss(loss_rand, 0.3244, PRECISION_VALUE)

def test_cx_mse(rand_predics, rand_targets, ones_predics, ones_targets):
    error = LLweighted_MSELoss_Climax()
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.0063, PRECISION_VALUE)
    expected_loss(loss_rand, 0.1053, PRECISION_VALUE)

def test_cx_rmse(rand_predics, rand_targets, ones_predics, ones_targets):
    error = LLweighted_RMSELoss_Climax()
    loss_ones = error(ones_predics, ones_targets) 
    loss_rand = error(rand_predics, rand_targets)
    expected_loss(loss_ones, 0.0795, PRECISION_VALUE)
    expected_loss(loss_rand, 0.3244, PRECISION_VALUE)

def test_equality_ones(ones_predics, ones_targets):
    rmse = RMSELoss(reduction="mean")
    nrmse_s = NRMSELoss_s_ClimateBench()
    nrmse_g = NRMSELoss_g_ClimateBench()
    nrmse = NRMSELoss_ClimateBench()
    wb_rmse = LLWeighted_RMSELoss_WeatherBench()
    cx_rmse = LLweighted_RMSELoss_Climax()

    rmse_loss_ones = rmse(ones_predics, ones_targets)
    nrmse_s_loss_ones = nrmse_s(ones_predics, ones_targets) 
    nrmse_g_loss_ones = nrmse_g(ones_predics, ones_targets) 
    nrmse_loss_ones = nrmse(ones_predics, ones_targets) 
    wb_rmse_loss_ones = wb_rmse(ones_predics, ones_targets)
    cx_rmse_loss_ones = cx_rmse(ones_predics, ones_targets)

    assert rmse_loss_ones.item() == pytest.approx(nrmse_g_loss_ones.item(), abs=PRECISION_VALUE)
    assert wb_rmse_loss_ones.item() == pytest.approx(cx_rmse_loss_ones.item(), abs=PRECISION_VALUE)
    assert (nrmse_s_loss_ones.item() + 5 * nrmse_g_loss_ones.item()) == pytest.approx(nrmse_loss_ones.item(), abs=PRECISION_VALUE)

def test_equality_rand(rand_predics, rand_targets):
    wb_rmse = LLWeighted_RMSELoss_WeatherBench()
    cx_rmse = LLweighted_RMSELoss_Climax()
    nrmse_s = NRMSELoss_s_ClimateBench()
    nrmse_g = NRMSELoss_g_ClimateBench()
    nrmse = NRMSELoss_ClimateBench()

    nrmse_s_loss_rand = nrmse_s(rand_predics, rand_targets) 
    nrmse_g_loss_rand = nrmse_g(rand_predics, rand_targets) 
    nrmse_loss_rand = nrmse(rand_predics, rand_targets) 
    wb_rmse_loss_rand = wb_rmse(rand_predics, rand_targets)
    cx_rmse_loss_rand = cx_rmse(rand_predics, rand_targets)

    assert wb_rmse_loss_rand.item() == pytest.approx(cx_rmse_loss_rand.item(), abs=PRECISION_VALUE)
    assert (nrmse_s_loss_rand.item() + 5 * nrmse_g_loss_rand.item()) == pytest.approx(nrmse_loss_rand.item(), abs=PRECISION_VALUE)


