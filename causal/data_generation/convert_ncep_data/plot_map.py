import imageio
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pathlib import Path


def plot_at_origin(df: pd.DataFrame, features_name: list, path: str, lat: float
                   = None, lon: float = None):
    """
    Plot the value through time at lat, lon
    Only plot the first feature in 'features_name'
    """
    if lat is None and lon is None:
        lat = np.random.choice(df["lat"].unique(), size=1)[0]
        lon = np.random.choice(df["lon"].unique(), size=1)[0]
    x = df[((df["lat"] == lat) & (df["lon"] == lon))][features_name[0]].values
    plt.plot(x)
    plt.savefig(path, format="png")


def plot_contour(df: pd.DataFrame, features_name: list, path: str):
    """
    Plot an average of the data on a map.
    Only plot the first feature in 'features_name'
    """
    plt.close()
    df = df.groupby(["lat", "lon"])[features_name[0]].mean()
    df = df.reset_index()

    map = Basemap(projection='mill')  # , lat_0=-90, lon_0=0)
    map.drawcoastlines()
    map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
    map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])

    # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    # map.drawmapboundary(fill_color='aqua')
    # map.fillcontinents(color='coral',lake_color='aqua')
    X, Y = np.meshgrid(df["lon"].unique(), df["lat"].unique())
    Z = df[features_name[0]].values.reshape(X.shape[0], X.shape[1])
    map.contourf(X, Y, Z, latlon=True)
    # map.scatter([120.], [60.], s=100, c="red", latlon=True)
    plt.savefig(path, format="png")


def plot_gif(df: pd.DataFrame, features_name: list, output_path: str,
             max_steps: int = None):
    """
    Make a gif of the timeserie on a map
    Only plot the first feature in 'features_name'
    """
    timestamps = df["timestamp"].unique()
    timestamps = sorted(timestamps)
    filenames = []

    output_path = os.path.join(output_path, "gif")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i, timestamp in enumerate(timestamps):
        plt.close()
        map = Basemap(projection='mill')  # , lat_0=-90, lon_0=0)
        map.drawcoastlines()
        map.drawparallels(np.arange(-90, 90, 30),labels=[1, 0, 0, 0])
        map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])
        df_ = df[df["timestamp"] == timestamp]

        X, Y = np.meshgrid(df_["lon"].unique(), df_["lat"].unique())
        Z = df_[features_name[0]].values.reshape(X.shape[0], X.shape[1])
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
