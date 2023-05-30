import numpy as np
import xarray as xr

np.random.seed(0)

temperature = 15 + 8 * np.random.randn(2, 2, 3)

precipitation = 10 * np.random.rand(2, 2, 3)

lon = [[-99.83, -99.32], [-99.79, -99.23]]

lat = [[42.25, 42.21], [42.63, 42.59]]

time = pd.date_range("2014-09-06", periods=3)

#reference_time = pd.Timestamp("2014-09-05")

ds = xr.Dataset(

    data_vars=dict(

        temperature=(["x", "y", "time"], temperature),

        precipitation=(["x", "y", "time"], precipitation),

    ),

    coords=dict(

        lon=(["x", "y"], lon),

        lat=(["x", "y"], lat),

        time=time,


    ),

    attrs=dict(description="Weather related data."),

)

ds.to_netcdf("test.nc")
