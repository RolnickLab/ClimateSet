import glob
import os
import json
import tables
import argparse
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from plot_map import plot_timeserie, plot_average, plot_gif


def convert_netcdf_to_pandas(filename: str, features_name: list, frequency: str = "day"):
    """
    Convert a NetCDF file to a pandas dataframe using xarray.
    Args:
        filename: name of a NetCDF file
        features_name: name of feature to consider, if empty consider all
        frequency: frequency to aggregate the data (day|week|month)
    Returns:
        df, coordinates, ds.attrs, features_name: the dataframe, the latitude x
        longitude, the NetCDF metadata and the features_name used
    """
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
        raise NotImplementedError(f"This value for frequency ({frequency}) is not yet implemented")

    # convert time to timestamp (in seconds)
    df["timestamp"] = pd.to_datetime(df['time']).astype(int) / 10**9

    # keep only lat, lon, timestamp and the feature in 'features_name'
    columns_to_keep = ["timestamp"] + columns_to_keep
    df = df[columns_to_keep]

    # pivot the table so that there is only one row per time
    df_pivoted = df.pivot_table(index="timestamp", columns=["lat", "lon"], values=features_name[0])
    coordinates = np.zeros((df_pivoted.shape[1], 2))
    for i, col in enumerate(df_pivoted.columns):
        coordinates[i, 0] = col[0]
        coordinates[i, 1] = col[1]

    return df_pivoted, coordinates, ds.attrs, features_name


def find_all_nc_files(directory: str) -> list:
    """
    Find all NetCDF files in 'directory'
    Returns: a list of the files name
    """
    if directory[-1] == "/":
        pattern = f"{directory}*.nc"
    else:
        pattern = f"{directory}/*.nc"
    filenames = sorted([x for x in glob.glob(pattern)])
    return filenames


def main_numpy(netcdf_directory: str, output_path: str, features_name: list, frequency: str, verbose: bool):
    """
    Convert netCDF4 files from the NCEP-NCAR Reanalysis project to a numpy file.
    All the files are expected to be in the directory `netcdf_directory`
    Returns:
        df, n, coordinates, features_name: the complete dataframe, the number of
        samples, an array of the coordinates and the features_name
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
        df_temp, coordinates, metadata, features_name = convert_netcdf_to_pandas(filename, features_name, frequency)
        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])
        if verbose:
            print(df.shape)

    # convert the dataframe to numpy, create the path if necessary and save it
    data_path = os.path.join(output_path, "data.npy")
    if verbose:
        print(f"All files opened, converting to numpy and saving to {data_path}.")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    np_array = df.values
    # expand to have axis for n and d, respectively the number of timeseries
    # and of features
    np_array = np.expand_dims(np_array, axis=0)
    np_array = np.expand_dims(np_array, axis=2)
    np.save(data_path, np_array)

    # save a copy of one metadata file
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return df, df.shape[0], coordinates, features_name


def main_hdf5(netcdf_directory: str, output_path: str, features_name: list, frequency: str, verbose: bool):
    """
    Convert netCDF4 files from the NCEP-NCAR Reanalysis project to a hdf5 file.
    Only open one files at a time and append the data to the hdf5 file
    All the files are expected to be in the directory `netcdf_directory`
    Returns:
        df, n, coordinates, features_name: the first dataframe, the number of
        samples, an array of the coordinates and the features_name of the first file
    """
    df = None
    n = 0

    # find all the netCDF4 in the directory `netcdf_directory`
    filenames = find_all_nc_files(netcdf_directory)
    if verbose:
        print(f"NetCDF Files found: {filenames}")

    # convert all netCDF4 files in a directory to pandas dataframe and append
    # it to the hdf5 file
    for i, filename in enumerate(filenames):
        if verbose:
            print(f"opening file: {filename}")
        df, coordinates, metadata, features_name = convert_netcdf_to_pandas(filename, features_name, frequency)

        # convert the dataframe to numpy, create the path if necessary and save it
        data_path = os.path.join(output_path, "data.h5")
        if verbose:
            print(df.shape)
            print(f"Converting to numpy and saving to {data_path}.")
        Path(output_path).mkdir(parents=True, exist_ok=True)


        # expand to have axis for n and d, respectively the number of timeseries
        # and of features
        np_array = df.values
        np_array = np.expand_dims(np_array, axis=0)
        np_array = np.expand_dims(np_array, axis=2)
        n += np_array.shape[1]

        if i == 0:
            first_df = df
            first_features_name = features_name

            # create the file for the first step
            f = tables.open_file(data_path, mode='w')
            atom = tables.Float64Atom()
            array = f.create_earray(f.root, 'data', atom, (np_array.shape[0], 0, np_array.shape[2], np_array.shape[3]))
            array.append(np_array)
            f.close()

            # save a copy of the first metadata file
            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        else:
            # append data to the existing hdf5 file
            f = tables.open_file(data_path, mode='a')
            f.root.data.append(np_array)
            f.close()

    # quick reading test
    f = tables.open_file(data_path, mode='r')
    data = f.root.data[0, 5:10]
    if verbose:
        print("Test the hdf5 file, here are the row 5 to 10:")
        print(data)
    f.close()

    return first_df, n, coordinates, first_features_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NetCDF files to numpy or hdf5. Can also plot visualizations.")
    parser.add_argument("--data-path", type=str, default="data/specific_humidity",
                        help="Path to the directory containing the NetCDF files")
    parser.add_argument("--output-path", type=str, default="specific_humidity_results",
                        help="Path where to save the results.")
    parser.add_argument("--features-name", nargs="+",
                        help="Name of the feature to use, if not specified use all")
    parser.add_argument("--frequency", type=str, default="day",
                        help="Frequency to which we parse the data (day|week|month)")
    parser.add_argument("--verbose", action="store_true",
                        help="If True, print useful messages")
    parser.add_argument("--hdf5", action="store_true",
                        help="If True, save result as an hdf5 file")
    parser.add_argument("--gif-max-step", type=int, default=50,
                        help="Maximal number of step to consider to generate the gif")
    args = parser.parse_args()

    # TODO: if necessary, adapt to multiple feature. Now, probably only works
    # for one feature

    if args.hdf5:
        df, n, coordinates, features_name = main_hdf5(args.data_path,
                                                      args.output_path,
                                                      args.features_name,
                                                      args.frequency,
                                                      args.verbose)
    else:
        df, n, coordinates, features_name = main_numpy(args.data_path,
                                                       args.output_path,
                                                       args.features_name,
                                                       args.frequency,
                                                       args.verbose)

    # save a json containing some parameters of the dataset
    params = {"n": 1,
              "num_timesteps": n,
              "num_features": 1,
              "d_x": df.shape[1]
             }
    json_path = os.path.join(args.output_path, "data_params.json")
    with open(json_path, "w") as file:
        json.dump(params, file, indent=4)

    # save the coordinates
    coord_path = os.path.join(args.output_path, "coordinates.npy")
    np.save(coord_path, coordinates)

    # plot data
    timeserie_path = os.path.join(args.output_path, "timeserie.png")
    average_path = os.path.join(args.output_path, "average.png")
    plot_timeserie(df, coordinates, args.frequency, features_name[0], timeserie_path)
    plot_average(df, coordinates, average_path)
    plot_gif(df, coordinates, args.output_path, args.gif_max_step)
