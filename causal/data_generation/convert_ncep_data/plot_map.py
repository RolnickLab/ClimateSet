import imageio
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pathlib import Path


def plot_timeserie(df: pd.DataFrame, coordinates: np.ndarray, frequency: str,
                   feature_name: str, path: str, lat: float = None, lon: float = None):
    """
    Plot the value through time at lat, lon
    Args:
        df: a dataframe containing all the data
        coordinates: array of latitude and longitude
        frequency: frequency of the data (day|week|month), just used for the
        x-label
        feature_name: name of the feature, used for y-label
        path: path where to save the figure
        lat: latitude
        lon: longitude
    """
    if lat is None and lon is None:
        idx = np.random.choice(coordinates.shape[0], size=1)[0]
        lat_lon = (coordinates[idx, 0], coordinates[idx, 1])
    else:
        lat_lon = (lat, lon)

    plt.plot(df[lat_lon].values)
    plt.xlabel(frequency)
    plt.ylabel(feature_name)
    plt.savefig(path, format="png")


def plot_average(df: pd.DataFrame, coordinates: np.ndarray, path: str):
    """
    Plot an average of the data on a map.
    Args:
        df: a dataframe containing all the data
        coordinates: array of latitude and longitude
        path: path where to save the figure
    """
    plt.close()
    z = df.mean().values

    map = Basemap(projection='mill')  # , lat_0=-90, lon_0=0)
    map.drawcoastlines()
    map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
    map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])

    lat = np.unique(coordinates[:, 0])
    lon = np.unique(coordinates[:, 1])
    X, Y = np.meshgrid(lon, lat)
    Z = z.reshape(X.shape[0], X.shape[1])

    map.contourf(X, Y, Z, latlon=True)
    # map.scatter([120.], [60.], s=100, c="red", latlon=True)
    plt.colorbar()
    plt.savefig(path, format="png")


def plot_gif(df: pd.DataFrame, coordinates: np.ndarray, output_path: str,
             max_steps: int = None):
    """
    Make a gif of the timeserie on a map
    Only plot the first feature in 'features_name'
    Args:
        df: a dataframe containing all the data
        coordinates: array of latitude and longitude
        output_path: path where to save the gifs and the animation
        max_steps: maximal number of step to consider to generate the
        animation, if None, process the whole time serie
    """
    filenames = []

    output_path = os.path.join(output_path, "gif")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    lat = np.unique(coordinates[:, 0])
    lon = np.unique(coordinates[:, 1])
    X, Y = np.meshgrid(lon, lat)

    for i in range(df.shape[0]):
        plt.close()
        map = Basemap(projection='mill')  # , lat_0=-90, lon_0=0)
        map.drawcoastlines()
        map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
        map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])

        Z = df.iloc[i].values.reshape(X.shape[0], X.shape[1])
        map.contourf(X, Y, Z, latlon=True)
        filename = f"gif{i}.png"
        path = os.path.join(output_path, filename)
        filenames.append(filename)
        plt.savefig(path, format="png")
        if max_steps is not None and i >= max_steps:
            break

    # build gif from images
    filepath = os.path.join(output_path, "final_gif.gif")
    with imageio.get_writer(filepath, mode='I') as writer:
        for filename in filenames:
            filepath = os.path.join(output_path, filename)
            image = imageio.imread(filepath)
            writer.append_data(image)
