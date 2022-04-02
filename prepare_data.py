# Read out data used from ClimateBench
import xarray as xr
from pathlib import Path

TRAIN_DIR = Path("~/Documents/Master/CausalSuperEmulator/Data/train_val/")
TEST_DIR = Path("~/Documents/Master/CausalSuperEmulator/Data/test/")


def main():
    """ Read the ClimateBench data and print some information about it.
    """
    # netcdf data
    # Input: year, longitude, latitude, CO2, CH4, BC, SO2
    # Output: longitude, latitude, time, diurnal_temperature_range, tas, pr, pr90
    output = xr.open_dataset(Path.joinpath(TEST_DIR, "outputs_ssp245.nc")).sel(time=slice(2050, 2100))
    print(output)


if __name__ == '__main__':
    main()
