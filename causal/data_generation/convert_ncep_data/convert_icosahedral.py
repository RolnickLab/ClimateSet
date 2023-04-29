import xarray as xr
import glob
import tables
import pandas as pd
import numpy as np
from pathlib import Path
import os


class SeasonRemover:
    """ Remove the seasonal cycle. Basically remove the mean for each day/week of
    the year. Update keeps track of the mean and std in an online fashion """
    def __init__(self, t_per_year, d_x):
        self.count = 0
        self.mean = np.zeros((t_per_year, d_x))
        self.std = np.zeros((t_per_year, d_x))
        self.m2 = np.zeros((t_per_year, d_x))  # this is \sum(x_i - mean) ** 2

    def update(self, data):
        self.count += 1
        delta = data - self.mean
        self.mean += delta / self.count
        delta2 = data - self.mean
        self.m2 += delta * delta2


def load(file_path: Path) -> xr.Dataset:
	""" Load a GRIB2 file and return an xarray object.
	Args:
		file_path (Path): Path to the file that should be loaded.
	Returns:
		xr.Dataset: The loaded dataset
	"""
	return xr.load_dataset(file_path, engine="cfgrib")


def find_all_files(directory: str, extension: str = "grib") -> list:
    """
    Find all NetCDF or grib files in 'directory'
    Returns: a list of the files name
    """
    if directory[-1] == "/":
        pattern = f"{directory}*.{extension}"
    else:
        pattern = f"{directory}/*.{extension}"
    filenames = sorted([x for x in glob.glob(pattern)])
    return filenames


def convert_to_h5(file_path, extension, output_path="./", verbose=True):
    filenames = find_all_files(file_path, extension)
    data_path = os.path.join(output_path, "data.h5")
    total_rows = 0
    sections = []

    for i, filename in enumerate(filenames):
        if verbose:
            print(f"opening file: {filename}")
        ds = load(filename)
        df = ds.to_dataframe()
        # ds.close()
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df['valid_time']).astype(int) / 10**9

        df['location'] = df.groupby(df.step).cumcount()

        df = df[['timestamp', 'sp', 'location']]
        df = df.pivot_table(index="timestamp", columns="location", values="sp")
        np_array = df.values
        np_array = np_array.reshape(1, np_array.shape[0], 1, -1)

        if i == 0:
            # create the file for the first step
            f = tables.open_file(data_path, mode='w')
            atom = tables.Float64Atom()
            array = f.create_earray(f.root, 'data', atom, (1, 0, 1, np_array.shape[-1]))
            array.append(np_array)
            f.close()
            total_rows += np_array.shape[1]
            sections.append(np_array.shape[1])

            season_remover = SeasonRemover(df.shape[0], df.shape[1])
            season_remover.update(df.values)
        else:
            # append data to the existing hdf5 file
            f = tables.open_file(data_path, mode='a')
            f.root.data.append(np_array)
            f.close()
            total_rows += np_array.shape[1]
            sections.append(np_array.shape[1])
            season_remover.update(df.values)
        print(total_rows)

    # repass through the data to remove the seasonal effect
    idx = 0
    f = tables.open_file(data_path, mode='r+')
    for section in sections:
        data = f.root.data[:, idx:idx + section]
        m2 = season_remover.m2.reshape(data.shape)
        mean = season_remover.mean.reshape(data.shape)

        std = np.sqrt(m2 / season_remover.count)
        f.root.data[:, idx:idx + section] = (data - mean) / std
        idx += section
    f.close()

    # quick reading test
    f = tables.open_file(data_path, mode='r')
    data = f.root.data[0, 5:10]
    if verbose:
        print("Test the hdf5 file, here are the row 5 to 10:")
        print(data)
    f.close()

def main():
	# file_path = Path("./icosahedral_weekly/slp.1948.grib")
    convert_to_h5("./data/icosahedral_weekly", "grib")

	# ds = load(file_path)
	# print(has_nans(ds))
	# print_infos(ds)

if __name__ == "__main__":
	main()
