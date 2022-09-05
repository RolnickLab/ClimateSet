import xarray as xr

print("Start testing downlaoder ...")
dataset = xr.open_dataset("http://basin.ceoe.udel.edu/thredds/dodsC/DEOSAG.nc")

print("Dataset properties:")
print(dataset)

print("... end of testing downloader.")
