"""
Helper script to bring files downloaded via a esgf wget script rather than the causalpaca downloader into a desired form so it can be processed further with the causalpaca pipeline.
"""
import os
import numpy as np
import xarray as xr

from data_building.parameters.esgf_server_constants import RES_TO_CHUNKSIZE


# path to folder where data downloaded with a wget script is located
esgf_path = "/home/charlie/Documents/MILA/causalpaca/data/mother_data/all_input4mips/"

# path to where sorted raw data should be stored -> seperation for years and brought into desired hierachy for further preprocessing with our pipeline
data_dir_parent = "/home/charlie/Documents/MILA/causalpaca/data/data/input4mips_all/"

# flag if existent data should be overwritten
overwrite = False


def extract_target_mip_exp_name(filename: str, target_mip: str):
    """Helper function extracting the target experiment name from a given file name and the target's umbrella MIP.
    supported target mips: "CMIP" "ScenarioMIP", "DAMIP", "AerChemMIP"

    params:
        filename (str): name of the download url to extract the information from
        target_mip (str): name of the umbreall MIP

    """
    print("file_name", f)
    try:
        year_end = filename.split("_")[-1].split("-")[1].split(".")[0][:4]
    # print(f'years from {year_from} to {year_end}')
    except IndexError:
        print("sth went wrong with year extraction")
        return None

    if (target_mip == "ScenarioMIP") or (target_mip == "DAMIP"):
        # extract ssp experiment from file name
        experiment = "ssp" + filename.split("ssp")[-1][:3]
        if "covid" in filename:
            experiment = experiment + "_covid"
    elif (target_mip == "CMIP") & (int(year_end) < 2016):
        experiment = "historical"

    elif target_mip == "AerChemMIP":
        experiment = "ssp" + filename.split("ssp")[-1][:3]
        if "lowNTCF" in filename:
            experiment = experiment + "_lowNTTCF"

    elif int(year_end) < 2016:
        experiment = "historical"
    else:
        print("WARNING: unknown target mip", target_mip)
        experiment = "None"

    return experiment


# for file in esgf data

if __name__ == "__main__":
    for f in os.listdir(esgf_path):
        print(f)
        if f.split(".")[-1] == "csv":
            continue  # skip unwanted csv files
        elif f.split(".")[-1] == "sh":
            continue  # skip unwanted bash scripts
        elif f.split(".")[-1] == "status":
            continue  # skip unwanted bash scripts
        # extract inoframiton from file_name (var, experiment)
        fl = f.split("_")
        print(fl)
        variable = fl[0]
        variable = variable.replace(" ", "_").replace("-", "_")
        project = "input4mips"

        # co2mass handling (duncan data) TODO: handle (?)
        if variable == "co2mass":
            # save to cmip file structure
            project = "CMIP6"
            ex = "co2mass_Amon_NorESM2-LM_abrupt-4xCO2_r1i1p1f1_gm_009101-010012.nc"
            model = fl[2]
            experiment = fl[3]
            ensemble_member = fl[4]
            grid_label = fl[5]

        else:
            target_mip = fl[3]
            experiment = extract_target_mip_exp_name(f, target_mip)
            if experiment is None:
                print("Skipping")
                continue
            print("Target MIP: ", target_mip)
        print("Variable: ", variable)
        print("Experiment: ", experiment)

        try:
            ds = xr.open_dataset(esgf_path + f, engine="netcdf4")
        except OSError:
            print("Something is wrong with the file. Skipping.")
            continue

        # TODO: extract frequency and grid label and nom_res
        frequency = ds.attrs["frequency"]
        grid_label = ds.attrs["grid_label"]
        nominal_resolution = ds.attrs["nominal_resolution"].replace(" ", "_")
        print("Nominal resolution: ", nominal_resolution)
        print("Frequency: ", frequency)
        print("Grid Label: ", grid_label)

        chunksize = RES_TO_CHUNKSIZE[frequency]

        years = np.unique(ds.time.dt.year.to_numpy())
        print(f"Data covering years: {years[0]} to {years[-1]}")

        for y in years:
            y = str(y)

            if project == "CMIP6":
                out_dir = f"{project}/{model}/{ensemble_member}/{experiment}/{variable}/{nominal_resolution}/{frequency}/{y}/"

            else:
                out_dir = f"{project}/{experiment}/{variable}/{nominal_resolution}/{frequency}/{y}/"

            # Check whether the specified path exists or not
            path = data_dir_parent + out_dir
            isExist = os.path.exists(path)

            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(path)
                print("The new directory is created!")

            out_name = f"{project}_{experiment}_{variable}_{nominal_resolution}_{frequency}_{grid_label}_{y}.nc"
            outfile = path + out_name
            print(ds.data_vars)
            if variable == "co2mass":
                pass
            else:
                var = list(ds.keys())[0]
                print(var)
                ds.rename({var: variable})

            if (not overwrite) and os.path.isfile(outfile):
                print(f"File {outfile} already exists, skipping.")
            else:
                print("Selecting specific year ", y)
                try:
                    ds_y = ds.sel(time=y)
                except ValueError:
                    continue  # some very strange data

                print("Writing file")
                print(outfile)
                ds_y.to_netcdf(outfile)
