from parameters.constants import RES_TO_CHUNKSIZE
from pyesgf.search import SearchConnection
from parameters.mother_params import VARS, SCENARIOS

# from pyesgf.logon import LogonManager

from parameters.constants import (
    MODEL_SOURCES,
    VAR_SOURCE_LOOKUP,
    OPENID,
    PASSWORD,
    SUPPORTED_EXPERIMENTS,
)
from utils.helper_funcs import get_keys_from_value, get_MIP
import netCDF4

# import h5py
import pandas as pd
import xarray as xr
import os
import os.path
import numpy as np
from typing import List

overwrite = False  # flag if files should be overwritten


class Downloader:
    """
    Class handling the downloading of the data. It communicates with the esgf nodes to search and download the specified data.
    """

    def __init__(
        self,
        model: str = "NorESM2-LM",  # defaul as in ClimateBench
        experiments: List[str] = [
            "historical",
            "ssp370",
            "hist-GHG",
            "piControl",
            "ssp434",
            "ssp126",
        ],  # sub-selection of ClimateBench defaul
        vars: List[str] = ["tas", "pr", "SO2", "BC"],
        data_dir: str = "data/data/",
        max_ensemble_members: int = 10, #max ensemble members
        ensemlble_members: List[str] = None #preferred ensemble members used, if None not considered
    ):
        """Init method for the Downloader
        params:
            model (str): Id of the model from which output should be downloaded. A list of all supported model ids can be find in parameters.constants.MODEL_SOURCES. Model data only.
            experiments ([str]):  List of simulations from which data should be downloaded. Model data only.
            experiments ([str]): List of variables for which data should be downloaded. Both model and raw data.
            data_dir: (str): Relative or absolute path to the directory where data should be stored. Will be created if not yet existent.
        """

        self.model = model
        self.experiments = experiments  # TODO: have a list of supported experiments before trying to look for them on the node to reduce computation cost
        # assign vars to either target or raw source
        self.raw_vars = []
        self.model_vars = []
        self.max_ensemble_members = max_ensemble_members
        self.ensemble_members = ensemlble_members

        # take care of var mistype (node takes no spaces or '-' only '_')
        vars = [v.replace(" ", "_").replace("-", "_") for v in vars]
        print("Cleaned vars", vars)
        for v in vars:
            t = get_keys_from_value(VAR_SOURCE_LOOKUP, v)
            if t == "model":
                self.model_vars.append(v)
            elif t == "raw":
                self.raw_vars.append(v)

            else:
                print(
                    f"WARNING: unknown source type for var {v}. Not supported. Skipping."
                )

        print(f"Raw variables to download: {self.raw_vars}")
        print(f"Model prediced vars to download: {self.model_vars}")

        try:

            self.model_node_link = MODEL_SOURCES[self.model]["node_link"]
            self.model_source_center = MODEL_SOURCES[self.model]["center"]
        except KeyError:
            print(f"WARNING: Model {self.model} unknown. Using default instead.")
            self.model = next(iter(MODEL_SOURCES))
            self.model_node_link = MODEL_SOURCES[self.model]["node_link"]
            self.model_source_center = MODEL_SOURCES[self.model]["center"]
            print("Using:", self.model)
        print("model node link:", self.model_node_link)

        # log on Manager
        # self.lm = LogonManager()
        # self.lm.logoff()
        # self.lm.logon_with_openid(openid=OPENID, password=PASSWORD, bootstrap=True)
        # print("Log In to Node:", self.lm.is_logged_on())

        self.data_dir_parent = data_dir
        self.overwrite = False

        # TODO: create folder hierachy / check if existent make new if not

        # TODO: more checkups?

    def download_from_model_single_var(
        self,
        variable: str,
        experiment: str,
        project: str = "CMIP6",
        default_frequency: str = "mon",
        default_version: str = "latest",
        default_grid_label: str = "gn",
    ):
        """Function handling the download of a single variable-experiment pair that is associated wtih a model's output (CMIP data).
        params:
            variable (str): variable Id
            experiment (str): experiment Id
            project (str): umbrella project id e.g. CMIPx
            default_frequency (str): default frequency to download
            default_version (str): data upload version, if 'latest', the newest version will get selected always
            defaul_grid_label (str): default gridding method in which the data is provided

        """
        conn = SearchConnection(self.model_node_link, distrib=False)

        facets = "project,experiment_id,source_id,variable,frequency,variant_label,variable, nominal_resolution, version, grid_label, experiment_id"

        """"

        # extracting available facets
        ctx = conn.new_context(project=project, source_id=self.model)
        available_facets=ctx.facet_counts
        for k in available_facets.keys():
            print(f"\n facet {k}")
            vs=[str(v) for v in available_facets[k].keys()]
            print(vs)
        raise RuntimeError

        """

        ctx = conn.new_context(
            project=project,
            experiment_id=experiment,
            source_id=self.model,
            variable=variable,
            facets=facets,
        )

        # dealing with grid labels
        grid_labels = list(ctx.facet_counts["grid_label"].keys())
        print("Available grid labels:", grid_labels)
        if default_grid_label in grid_labels:
            print("Choosing grid:", default_grid_label)
            grid_label = default_grid_label
        else:
            print("Default grid label not available.")
            grid_label = grid_labels[0]
            print(f"Choosing grid {grid_label} instead.")
        ctx = ctx.constrain(grid_label=grid_label)

        try:
            nominal_resolutions = list(ctx.facet_counts["nominal_resolution"].keys())
            print("Available nominal resolution:", nominal_resolutions)
            # deal with multipl nom resolutions availabe
            if len(nominal_resolutions) > 1:
                print(
                    "Multiple nominal resolutions exist, choosing smallest_nominal resolution (trying), please do a check up"
                )

            nominal_resolution = nominal_resolutions[-1]
            print("Choosing nominal resolution", nominal_resolution)
        except IndexError:
            print("No nominal resolution")

        # dealing with frequencies
        print("Available frequencies: ", ctx.facet_counts["frequency"].keys())
        frequency = "mon"  # list(ctx.facet_counts['frequency'].keys())[-1]
        print("choosing frequency: ", frequency)

        ctx_origin = ctx.constrain(
            frequency=frequency, nominal_resolution=nominal_resolution
        )

        variants = list(ctx.facet_counts["variant_label"].keys())

        download_files = {}

        print("Available variants:", variants, "\n")

        if self.ensemble_members is None:
            if self.max_ensemble_members>len(variants):
                print("Less ensemble members available than maximum number desired. Including all variants.")
                ensemble_member_final_list=variants
            else:
                print(f"{len(variants)} ensemble members available than desired (max {self.max_ensemble_members}. Choosing only the first {self.max_ensemble_members}.).")
                ensemble_member_final_list=variants[:self.max_ensemble_members]
        else:
            print(f"Desired list of ensemble members given: {self.ensemble_members}")
            ensemble_member_final_list = list(set(variants) & set(self.ensemble_members))
            if len(ensemble_member_final_list)==0:
                print("WARNING: no overlap between available and desired ensemble members!")
                print("Skipping.")
                return None



        for i, ensemble_member in enumerate(ensemble_member_final_list):

            print(f"Ensembles member: {ensemble_member}")
            ctx = ctx_origin.constrain(variant_label=ensemble_member)

            # pick a version
            versions = list(ctx.facet_counts["version"].keys())
            print("Available versions:", versions)

            if default_version == "latest":
                version = versions[0]
                print("Chooosing latetst version:", version)
            else:
                try:
                    version = versions[default_version]
                except KeyError:
                    print(f"Preferred version {default_version} does not exist.")
                    version = versions[0]
                    print(f"Resuming with latest verison:", version)

            ctx = ctx.constrain(version=version)

            result = ctx.search()

            print(f"Result len: {len(result)}")

            files_list = [r.file_context().search() for r in result]

            for i, files in enumerate(files_list):

                file_names = [files[i].opendap_url for i in range(len(files))]
                print(f"File {i} names: ", file_names)

                num_files = len(file_names)

                chunksize = RES_TO_CHUNKSIZE[frequency]
                print("Chunksize", chunksize)

                nominal_resolution = nominal_resolution.replace(" ", "_")

                for f in file_names:
                    # try to opend datset
                    try:
                        ds = xr.open_dataset(
                            f, chunks={"time": chunksize}, engine="netcdf4"
                        )

                    except OSError:
                        print(
                            "Having problems downloading th edateset. The server might be down. Skipping"
                        )
                        continue

                    years = np.unique(ds.time.dt.year.to_numpy())
                    print(f"Data covering years: {years[0]} to {years[-1]}")

                    for y in years:
                        y = str(y)
                        out_dir = f"{project}/{self.model}/{ensemble_member}/{experiment}/{variable}/{nominal_resolution}/{frequency}/{y}/"

                        # check if path is existent
                        path = self.data_dir_parent + out_dir
                        isExist = os.path.exists(path)

                        if not isExist:

                            # Create a new directory because it does not exist
                            os.makedirs(path)
                            print("The new directory is created!")

                        out_name = f"{project}_{self.model}_{ensemble_member}_{experiment}_{variable}_{nominal_resolution}_{frequency}_{grid_label}_{y}.nc"
                        outfile = path + out_name

                        if (not overwrite) and os.path.isfile(outfile):
                            print(f"File {outfile} already exists, skipping.")
                        else:

                            print("Selecting specific year", y)
                            ds_y = ds.sel(time=y)
                            print(ds_y)
                            print("writing file")
                            print(outfile)
                            ds_y.to_netcdf(outfile)


    def download_raw_input_single_var(
        self,
        variable,
        project="input4mips",
        institution_id="PNNL-JGCRI",  # make sure that we have the correct data
        default_frequency="mon",
        default_version="latest",
        default_grid_label="gn",
    ):
        """Function handling the download of a all input4mips data associated with a single variable. A
        params:
            variable (str): variable Id
            project (str): umbrella project, here "input4mips"
            institution_id (str): id of the institution that provides the data
            default_frequency (str): default frequency to download
            default_version (str): data upload version, if 'latest', the newest version will get selected always
            defaul_grid_label (str): default gridding method in which the data is provided

        """
        conn = SearchConnection(self.model_node_link, distrib=False)

        facets = "project,frequency,variable,nominal_resolution,version,target_mip,grid_label"

        # basic constraining (projec, var, institution)
        ctx = conn.new_context(
            project=project,
            variable=variable,
            institution_id=institution_id,
            facets=facets,
        )

        # dealing with grid labels
        grid_labels = list(ctx.facet_counts["grid_label"].keys())
        print("Available grid labels:", grid_labels)
        if default_grid_label in grid_labels:
            print("Choosing grid:", default_grid_label)
            grid_label = default_grid_label
        else:
            print("Default grid label not available.")
            grid_label = grid_labels[0]
            print(f"Choosing grid {grid_label} instead.")
        ctx = ctx.constrain(grid_label=grid_label)

        # choose nominal resolution if existent
        try:
            nominal_resolutions = list(ctx.facet_counts["nominal_resolution"].keys())
            print("Available nominal resolution:", nominal_resolutions)

            # deal with mulitple nominal resoulitions, taking smalles one as default
            if len(nominal_resolutions) > 1:
                print(
                    "Multiple nominal resolutions exist, choosing smallest_nominal resolution (trying), please do a check up"
                )
            nominal_resolution = nominal_resolutions[0]
            print("Choosing nominal resolution", nominal_resolution)
            ctx = ctx.constrain(nominal_resolution=nominal_resolution)

        except IndexError:
            print("No nominal resolution")
            nominal_resolution = "none"

        # choose default frequency if wanted
        frequencies = list(ctx.facet_counts["frequency"].keys())
        print("Available frequencies: ", frequencies)

        if default_frequency in frequencies:
            frequency = default_frequency
            print("Choosing default frequency", frequency)
        else:
            frequency = frequencies[0]
            print(
                "Default frequency not available, choosing first available one instead: ",
                frequency,
            )
        ctx = ctx.constrain(frequency=frequency)

        # target mip group
        target_mips = list(ctx.facet_counts["target_mip"].keys())
        print(f"Available target mips: {target_mips}")
        ctx_origin = ctx

        print("\n")
        for t in target_mips:
            print(f"Target mip: {t}")
            ctx = ctx_origin.constrain(target_mip=t)

            versions = list(ctx.facet_counts["version"].keys())
            print("Available versions", versions)
            ctx_origin_v = ctx

            # deal with different versions
            if default_version == "latest":
                version = versions[0]
                print("Chooosing latetst version:", version)
            else:
                try:
                    version = versions[default_version]
                except KeyError:
                    print(f"Preferred version {default_version} does not exist.")
                    version = versions[0]
                    print(f"Resuming with latest verison:", version)

            ctx = ctx_origin_v.constrain(version=version)

            result = ctx.search()

            print(f"Result len: {len(result)}")

            files_list = [r.file_context().search() for r in result]

            for i, files in enumerate(files_list):
                file_names = [files[i].opendap_url for i in range(len(files))]
                print(f"File {i} names: ", file_names)
                num_files = len(file_names)

                # find out chunking dependent on resolution
                chunksize = RES_TO_CHUNKSIZE[frequency]
                print("Chunksize", chunksize)

                # replacing spaces for file naming
                nominal_resolution = nominal_resolution.replace(" ", "_")

                for f in file_names:

                    experiment = self.extract_target_mip_exp_name(f, t)

                    # make sure to only download data for wanted scenarios
                    if experiment in self.experiments:

                        print("Downloading data for experiment:", experiment)
                    else:

                        print(
                            f"Experiment {experiment} not in wanted experiments ({self.experiments}). Skipping"
                        )
                        continue

                    try:
                        ds = xr.open_dataset(f, chunks={"time": chunksize})
                    except OSError:
                        print(
                            "Having problems downloading th edateset. The server might be dwon. Skipping"
                        )
                        continue

                    years = np.unique(ds.time.dt.year.to_numpy())
                    print(f"Data covering years: {years[0]} to {years[-1]}")

                    for y in years:
                        y = str(y)
                        out_dir = f"{project}/{experiment}/{variable}/{nominal_resolution}/{frequency}/{y}/"

                        # Check whether the specified path exists or not
                        path = self.data_dir_parent + out_dir
                        isExist = os.path.exists(path)

                        if not isExist:

                            # Create a new directory because it does not exist
                            os.makedirs(path)
                            print("The new directory is created!")

                        out_name = f"{project}_{experiment}_{variable}_{nominal_resolution}_{frequency}_{grid_label}_{y}.nc"
                        outfile = path + out_name

                        if (not overwrite) and os.path.isfile(outfile):
                            print(f"File {outfile} already exists, skipping.")
                        else:

                            print("Selecting specific year ", y)
                            ds_y = ds.sel(time=y)
                            print(ds_y)

                            print("Writing file")
                            print(outfile)
                            ds_y.to_netcdf(outfile)

    def extract_target_mip_exp_name(self, filename: str, target_mip: str):

        """Helper function extracting the target experiment name from a given file name and the target's umbrella MIP.
        supported target mips: "CMIP" "ScenarioMIP", "DAMIP", "AerChemMIP"

        params:
            filename (str): name of the download url to extract the information from
            target_mip (str): name of the umbreall MIP

        """
        year_end = filename.split("_")[-1].split("-")[1].split(".")[0][:4]
        # print(f'years from {year_from} to {year_end}')

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

    def download_from_model(
        self,
        project: str = "CMIP6",
        default_frequency: str = "mon",
        default_version: str = "latest",
        default_grid_label: str = "gn",
    ):
        """
        Function handling the download of all variables that are associated wtih a model's output
        Searches for all filles associated with the respected variables and experiment that the downloader wsa initialized with.

        A search connection is established and the search is iterativeley constraint to meet all specifications.
        Data is downloaded and stored in a seperate file for each year. The default format is netCDF4.
        Resulting hierachy:
            CMIPx
                model_id
                    ensemble_member
                        experiment
                            variable
                                nominal_resolution
                                    frequency
                                        year.nc
        If the constraints cannot be met, per default behaviour for the downloader to selecf first other available value.


        params:
            project (str): umbrella project id e.g. CMIPx
            default_frequency (str): default frequency to download
            default_version (str): data upload version, if 'latest', the newest version will get selected always
            defaul_grid_label (str): default gridding method in which the data is provided

        """

        # iterate over respective vars
        for v in self.model_vars:
            print(f"Downloading data for variable: {v} \n \n ")
            # iterate over experiments
            for e in self.experiments:
                #check if experiment is availabe
                if e in SUPPORTED_EXPERIMENTS:
                    print(f"Downloading data for experiment: {e}\n")
                    self.download_from_model_single_var(v, e)
                else:
                    print(
                        f"Chosen experiment {e} not supported. All supported experiments: {SUPPORTED_EXPERIMENTS}. \n Skipping. \n"
                    )

    def download_raw_input(
        self,
        project="input4mips",
        institution_id="PNNL-JGCRI",  # make sure that we have the correct data
        default_frequency="mon",
        default_version="latest",
        default_grid_label="gn",
    ):

        """
        Function handling the download of all variables that are associated wtih a model's input (input4mips).
        Searches for all filles associated with the respected variables that the downloader was initialized with.
        A search connection is established and the search is iterativeley constraint to meet all specifications.
        Data is downloaded and stored in a seperate file for each year. The default format is netCDF4.
        Resulting hierachy:
            input4mips
                experiment
                    variable
                        nominal_resolution
                            frequency
                                year.nc
        If the constraints cannot be met, the default behaviour for the downloader is to select first other available value.

        params:
            project (str): umbrella project, in this case "input4mips"
            institution_id (str): institution that provided the data
            default_frequency (str): default frequency to download
            default_version (str): data upload version, if 'latest', the newest version will get selected always
            defaul_grid_label (str): default gridding method in which the data is provided

        """
        for v in self.raw_vars:
            print(f"Downloading data for variable: {v} \n \n ")
            self.download_raw_input_single_var(v)


if __name__ == "__main__":

    vars=VARS
    experiments=SCENARIOS
    model="CanESM5"
    vars=["pr", "tas"]
    #experiments=["ssp126", "ssp245", "ssp370", ""]
    #vars=["BC_em_anthro", "BC_em_openburning"]
    #experiments=["ssp126", "historical"]
    #model="NorESM2-LM"
    max_ensemble_members=1
    ensemble_members=["r1i1p1f1"]
    data_dir=f"{os.environ['SLURM_TMPDIR']}/causalpaca/data/"

    downloader = Downloader(experiments=experiments, vars=vars, model=model, data_dir=data_dir, ensemlble_members=ensemble_members)
    downloader.download_from_model()
    #downloader.download_raw_input()
