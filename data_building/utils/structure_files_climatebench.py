# source of duncan input4mips data: https://gws-access.jasmin.ac.uk/public/impala/dwatsonparris/ClimateBench/climate_bench_inputs.tar.gz
import os
import argparse
import numpy as np
import xarray as xr

from data_building.parameters.esgf_server_constants import RES_TO_CHUNKSIZE

parser = argparse.ArgumentParser(description="Restructure Input4MIPs data from ClimateBench for our file structure.")
parser.add_argument("-i", "--source", type=str, help="Where the downloaded Input4MIPs data can be found.", required=True)
parser.add_argument("-s", "--store", type=str, help="Where to store the newly structured files (in causalpaca).", required=True)
parser.add_argument("-o", "--overwrite", action="store_true", help="If data storage path should be overwritten.")
# duncan_path="/home/charlie/Documents/MILA/causalpaca/data/climate_bench_inputs/" # source of duncan data
# data_dir_parent="/home/charlie/Documents/MILA/causalpaca/data/data/" # to store
# duncan_path="/home/julia/LargeFiles/climate_bench_inputs/" # source of duncan data
# data_dir_parent="/home/julia/Documents/Master/CausalSuperEmulator/Code/causalpaca/data/data/" # to store
# overwrite=True


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
    elif (target_mip == "CMIP") & (int(year_end) < 2015):
        experiment = "historical"

    elif target_mip == "AerChemMIP":
        experiment = "ssp" + filename.split("ssp")[-1][:3]
        if "lowNTCF" in filename:
            experiment = experiment + "_lowNTTCF"

    else:
        print("WARNING: unknown target mip", target_mip)
        experiment = "None"

    return experiment


# for file in duncan data

if __name__ == '__main__':
    # handle argparser
    # print("hello")
    # print(len(parser[0]))
    # if (len(parser[0]) < 1) or (len(parser[1]) < 1) :
    #     raise ValueError("Please provide the path where the Input4MIPs was downloaded and where the newly structured files should be stored.")
    args = parser.parse_args()
    duncan_path = args.source
    data_dir_parent = args.store
    overwrite = args.overwrite

    for f in os.listdir(duncan_path):
        print(f)
        if f.split('.')[-1]=='csv':

            continue # unwanted csv files
        # extract inoframiton from file_name (var, experiment)
        fl=f.split("_")
        print(fl)
        variable=fl[0]
        variable=variable.replace(" ", "_").replace("-", "_")
        project="input4mips"
        if variable=="co2mass":
            # save to cmip file structure
            project="CMIP6"
            ex="co2mass_Amon_NorESM2-LM_abrupt-4xCO2_r1i1p1f1_gm_009101-010012.nc"
            model=fl[2]
            experiment=fl[3]
            ensemble_member=fl[4]
            grid_label=fl[5]


        else:
            target_mip = fl[3]
            experiment = extract_target_mip_exp_name(f, target_mip)
            if experiment is None:
                print("Skipping")
                continue
            print("target_mips", target_mip)
        print("var", variable)
        print("experiment", experiment)

        frequency = "mon"
        grid_label = "gn"
        nominal_resolution = "50_km"

        chunksize = RES_TO_CHUNKSIZE[frequency]

        try:
            ds = xr.open_dataset(
                duncan_path + f, chunks={"time": chunksize}, engine="netcdf4"
            )
        except OSError:
            print("Something is wrong with the file. Skipping.")
            continue


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
                    continue # some very strange data

                print(ds_y)

                print("Writing file")
                print(outfile)
                ds_y.to_netcdf(outfile)
