import glob
import os
import json
import tables
import xarray as xr
import pandas as pd
import numpy as np
from os import path
from pathlib import Path
from plot_map import plot_at_origin, plot_contour, plot_gif

def convert_netcdf_to_pandas(filename: str, features_name: list,
                             columns_to_drop: list, frequency: str = "day"):
    # open dataset with xarray and convert it to a pandas DataFrame
    ds = xr.open_dataset(filename)
    df = ds.to_dataframe()
    ds.close()
    df = df.reset_index()

    # if no features_name is specified, take all columns that are not latitude,
    # longitude and time
    if not features_name:
        features_name = list(set(df.columns) - {"lat", "lon", "time"})
    columns_to_keep = ["lat", "lon"] + features_name

    # average the data over week or month
    if frequency == "day":
        pass
    elif frequency == "week":
        df = df.groupby([pd.Grouper(key='time', freq="W"), "lat", "lon"])[features_name].mean().reset_index()
    elif frequency == "month":
        df = df.groupby([pd.Grouper(key='time', freq="M"), "lat", "lon"])[features_name].mean().reset_index()
    else:
        raise ValueError(f"This value for frequency ({frequency}) is not yet implemented")

    # convert time to timestamp (in seconds)
    df["timestamp"] = pd.to_datetime(df['time']).astype(int)/ 10**9

    # keep only lat, lon, timestamp and the feature in 'features_name'
    columns_to_keep = ["timestamp"] + columns_to_keep
    df = df[columns_to_keep]
    # df = df.drop(columns_to_drop, axis=1)

    return df, ds.attrs, features_name


def find_all_nc_files(directory: str):
    if directory[-1] == "/":
        pattern = f"{directory}*.nc"
    else:
        pattern = f"{directory}/*.nc"
    filenames = sorted([x for x in glob.glob(pattern)])
    return filenames


def main(netcdf_directory: str, output_path: str, features_name: list, frequency: str, verbose: bool):
    """
    Convert netCDF4 files from the NCEP-NCAR Reanalysis project to a numpy file.
    All the files are expected to be in the directory `netcdf_directory`
    Args:
        netcdf_directory: x
        output_path: x
        features_name: x
        frequency: x (day, month, week)
        verbose: if True, print messages at each step
    Returns:
        df, nparray: the dataframe and numpy array of the concatenated data
    """
    # TODO: could add year anomalizing, detrending?
    df = None

    # find all the netCDF4 in the directory `netcdf_directory`
    filenames = find_all_nc_files(netcdf_directory)
    if verbose:
        print(f"NetCDF Files found: {filenames}")

    # convert all netCDF4 files in a directory to a single pandas dataframe
    for filename in filenames:
        if verbose:
            print(f"opening file: {filename}")
        df_temp, metadata, features_name = convert_netcdf_to_pandas(filename, features_name, ["time"], frequency)
        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])
        if verbose:
            print(df.columns)
            print(df.shape)

    # convert the dataframe to numpy, create the path if necessary and save it
    data_path = os.path.join(output_path, "data.npy")
    if verbose:
        print(f"All files opened, converting to numpy and saving to {data_path}.")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    np_array = df.values
    np.save(data_path, np_array)

    # save a copy of one metadata file
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return df, np_array, features_name


def main_inplace(netcdf_directory: str, output_path: str, features_name: str, verbose: bool):
    """
    Convert netCDF4 files from the NCEP-NCAR Reanalysis project to a hdf5 file.
    Only open one files at a time and append the data to the hdf5 file
    All the files are expected to be in the directory `netcdf_directory`
    Args:
        netcdf_directory: x
        output_path: x
        features_name: x
        verbose: if True, print messages at each step
    """
    df = None

    # find all the netCDF4 in the directory `netcdf_directory`
    filenames = find_all_nc_files(netcdf_directory)
    if verbose:
        print(f"NetCDF Files found: {filenames}")

    # convert all netCDF4 files in a directory to pandas dataframe and append
    # it to the hdf5 file
    for i, filename in enumerate(filenames):
        if verbose:
            print(f"opening file: {filename}")
        df, metadata = convert_netcdf_to_pandas(filename, features_name, ["time"])
        print(df.shape)

        # convert the dataframe to numpy, create the path if necessary and save it
        data_path = os.path.join(output_path, "data.h5")
        if verbose:
            print(f"Converting to numpy and saving to {data_path}.")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        if i == 0:
            # create the file for the first step
            f = tables.open_file(data_path, mode='w')
            atom = tables.Float64Atom()
            array = f.create_earray(f.root, 'data', atom, (0, df.shape[1]))
            array.append(df.values)
            f.close()

            # save a copy of the first metadata file
            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        else:
            # append data to the existing hdf5 file
            f = tables.open_file(data_path, mode='a')
            f.root.data.append(df.values)
            f.close()

    # quick reading test
    f = tables.open_file(data_path, mode='r')
    data = f.root.data[5:10]
    if verbose:
        print("Test the hdf5 file, here are the row 5 to 10:")
        print(data)
    f.close()


if __name__ == "__main__":
    netcdf_directory = "data/specific_humidity"
    output_path = "specific_humidity_results"
    features_name = []
    df, _, features_name = main(netcdf_directory, output_path, features_name, "week", verbose=True)
    # main_inplace(netcdf_directory, output_path, features_name, verbose=True)

    # plot data and save file
    plot_origin_path = os.path.join(output_path, "at_origin.png")
    plot_average_path = os.path.join(output_path, "average.png")

    plot_at_origin(df, features_name, plot_origin_path)
    plot_contour(df, features_name, plot_average_path)
    plot_gif(df, features_name, output_path, 100)
