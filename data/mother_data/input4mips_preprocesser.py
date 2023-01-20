# Input4MIPs data is preprocessed here after downloading them
import os
import re
import cftime
import warnings
import numpy as np
import xesmf as xe
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from pathlib import Path
from cftime import num2date, date2num

# to delete (only for testing)
import pandas as pd

# EXTERNAL TODOS
    # rename "mon" in input4mips to 5y
    # check with charlie: abbreviation for yearly
    # we want to have two data dirs: raw (current data, untouched, but checked for corruption) / preprocessed (first: raw-processed; second: res-preprocessed)
    # test_mother in testing?
    # cmip6 assuming several vars: check if the requested vars are available in the same resolutions
    # Charlie: BC_em_biomassburning from Duncan: nominal_resolution is actually 25km - can we store that accordingly?
# LATER TODOS
    # how to make the path handeling if this is done on an server?
    # for the moment we interpolate and aggregate only input4mips. we need to be able to do the same for cmip6
# INTERNAL TODOS
    # Structure: raw vs res & input4mips vs cmip6.
        # I propose: A dir "preprocessers". All preprocessing files live there. (duncan,)
        # And on the mother_data level: raw_preprocess.py (is run by us, only one time, processes both input4mips and cmip6 data)
        # also on the mother_data level: res_preprocess.py (is run with params and should be callable from user_data)
    # CO2 baseline model thingy (is it raw or res preprocessing?)
    # --> check climatebench for that one (https://github.com/duncanwp/ClimateBench/blob/main/prep_input_data.ipynb)
    # --> 1) Substract CO2 baseline from CO2 values in CMIP6 data, to make models more comparable: Different models might have different baselines
             # Question: Why not for all long-living GHG?
    # --> 2) Use the cumulative CO2 mass in atmosphere (after interpolating to annual data) [so, only CO2 is cumulative??]
             # Question: Why not for all long-living GHG. Answer: Because only CO2 is a cumulative emissor.
    # Make a list in params with long-living GHGs (so we can check if the gas is in that list)
    # special case historical data (different res etc)
# Questions:
    # should the variable check made be one time (mother) or during usage (user)


# TODO import this from data_paths.py
DATA_PATH = "/home/julia/Documents/Master/CausalSuperEmulator/Code/causalpaca/data/data/raw/input4mips/"
PROCESSED_PATH = "/home/julia/Documents/Master/CausalSuperEmulator/Code/causalpaca/data/data/processed/input4mips/"
# TODO import this from mother_params
VARS = ["BC", "CH4", "CO2", "SO2"]

class Input4mipsRawPreprocesser:
    """ Responsible for the raw preprocessing of input4mips data.
    This class is used only once within the mother and should not be called by
    the user.
    """
    def __init__(
        self,
        raw_path: Path,
        processed_path: Path,
        test_scenario: bool = False,
        ghg_vars: List[str] = [],
    ):
        """ Initialize raw preprocesser
        Args:
            raw_path (Path): where the downloaded input4mips data lives
            test_scenario (bool): For internal use - indicates testing scenario,
                where only a subset of the files are tested. Default False.
            ghg_vars (List<str>): Greenhouse gases considered from the
                Input4MIPs data. E.g. ["CO2", "CH4", "SO2", "BC"]
        """
        self.raw_path = raw_path
        # ATTENTION: this path is not used during res processing right now
        self.processed_path = processed_path
        self.test_scenario = test_scenario
        self.vars = ghg_vars

        # the usual unit for input4mips data. this is a flux rate.
        # for conversion to kg: multiply with area of emission and timespan of emission
        # historical openburning fire dataset by Van Marle et al. 2017 uses two emissions:
            # 1) fire carbon emissions (g C m-2 month-1)
            # 2) dry matter emissions (kg DM m-2 month-1)
        # the units are not stored in the meta data
        self.ghg_unit = "kg m-2 s-1"
        self.ghg_str_unit = "Mass flux"

        # store which scenarios do have all the vars
        self.full_scenarios = self._get_full_scenarios()

        # size of directories
        self.raw_len = self._count_files(self.raw_path)
        self.processed_len = None

        # afterwards:
        # 0. summing over sectors (and future other processes applied to all)
        # 1. nominal resolution processing (different class)
        # 2. temporal resolution processing (diff class)
        # 3. emission processing - summing up
        # 4. compare with CMIP6 data -> process and put into "loader"
            # attention: CMIP6 might need a separate "checker"

        # TODO: create one input4mips processer with different subclasses?

    def _count_files(self, path: Path) -> int:
        """ Counts number of files in a directory recursively.
        Args:
            path (Path): Pathlib Path of a directory whose files should be counted.
        Returns:
            int: number of files
        """
        return sum([len(files) for r, d, files in os.walk(path)])

    # TODO (later) make this TWO running through file loops:
        # one loop (on raw data): sanity checks + copy
        # second loop (on processed data): sum over sectors + future functions
    def run(
        self,
        sanity_checking: bool = True,
        create_processed_dir: bool = True,
        sum_over_sectors: bool = True
    ):
        """ Runs through all relevant raw preprocessing steps.

        Args:
            sanity_checking (bool): Indicates if sanity checks should be performed.
                Needs only to be performed once.
            create_processed_dir (bool): Indicates if the raw data should be
                copied to the preprocessing directory. Needs only to be performed
                once.
            sum_over_sectors (bool): Indicates if sectors should be summed up
                to one (within preprocessed dir).
        """
        if sanity_checking:
            print("Starting sanity checks ...")
            self.sanity_checks()
            if sanity: print("... checks ended successfully!")

        if create_processed_dir:
            print("Starting to copy raw data to processed data directory ...")
            self.copy_raw_to_processed()
            print("...finished copying all raw files to the processed directory.")

        self.processed_len = self._count_files(self.processed_path)

        if sum_over_sectors:
            print("Starting to sum over sectors ...")
            self.sum_up_sectors()
            print("...finished summing over sectors.")


    def _get_full_scenarios(self) -> List[str]:
        """ Create a list of those scenarios that have all desired vars.
        """
        # list of dirs that each scenario should contain
        desired_ghg_vars = []
        for var in self.vars:
            desired_ghg_vars.append(var + "_em_anthro")
            if not self.test_scenario:
                desired_ghg_vars.append(var + "_em_biomassburning")
                desired_ghg_vars.append(var + "_em_AIR_anthro")

        # go into the scenario dirs
        # & check if all vars + extensions are available
        full_scenarios = []
        os_walker = os.walk(self.raw_path)
        scenarios = next(os_walker)[1]
        for scenario in scenarios:
            # check if desired scenarios are available
            if set(desired_ghg_vars) <= set(next(os.walk(self.raw_path / scenario))[1]):
                full_scenarios.append(scenario)
            else:
                print(("{} scenario does not contain all needed variables. This scenario will be skipped.").format(scenario))

        return full_scenarios

    # TODO make this compatible for different OS (pathlib!)
    def sanity_check_file(self, file: str, root: str) -> bool:
        """ Makes sanity checks for a single file. Returns true if the checks
        were passed. Checks if the file uses the expected unit (fluxes in
        kg m-2 s-1). Checks if the temporal and nominal resolution in the file
        name are also the resolutions used within the file.

        Args:
            file (str): File name
            root (str): Root of the file

        Returns:
            True if all sanity checks were alright.
        Raises:
            AssertionErrors if there is a mismatch between expected and found
                resolutions and units
        """
        # TODO optional check that could be added: does the root name (50_km/mon/) match with file names?
        spat_res = file.split('_')[-5] + ' ' + file.split('_')[-4]
        temp_res = file.split('_')[-3]
        # read the file and check resolutions
        with xr.open_dataset (root + '/' + file) as ds:
            if spat_res != ds.nominal_resolution:
                raise AssertionError("Nominal resolution is not as expected. File: {}".format(file))
            if temp_res != ds.frequency:
                raise AssertionError("Temporal resolution is not as expected. File: {}".format(file))
            # is the variable's unit as expected?
            try:
                if self.ghg_unit != ds.variable_units:
                    raise AssertionError("Unit is not as expected. File: {}".format(file))
            except AttributeError:
                try:
                    if not (self.ghg_str_unit in ds.reporting_unit):
                        raise AssertionError("Unit is not as expected. File: {}".format(file))
                except AttributeError:
                    # e.g. historical openburning dataset by Van Merle et al. 2017
                    print("Unit of the file could not be accessed. File: {}".format(file))
        return True

    # this is part of the raw preprocesser
    # TODO calendar checks (type + number of days per year)
    def sanity_checks(self) -> bool:
        """ Checks if all Input4MIPs files use the expected unit (fluxes in
        kg m-2 s-1). Checks if the temporal and nominal resolution in the file
        name are also the resolutions used within the file.

        Returns:
            True if all sanity checks were alright.
        Raises:
            AssertionErrors if there is a mismatch between expected and found
                resolutions and units
        """
        # list all files in one scenario
        for scenario in self.full_scenarios:
            for root, dirs, files in os.walk(self.raw_path / scenario):
                if len(files) > 0:
                    for file in files:
                        self.sanity_check_file(file, root)
        return True

    # move to utils
    def copy_file(self, src: Path, dst: Path, overwrite: bool = True):
        """ Copies a single file. Might not work for all Operating Systems.
        Args:
            src (Path): Source of the file
            dst (Path): Destination of the file
            overwrite (bool): Whether an existing file should be overwritten.
                Default is True.
        """
        if overwrite:
            os.system("cp -p %s %s" % (src, dst))
        else:
            os.system("cp -p -n %s %s" % (src, dst))

    def copy_raw_to_processed(self, overwrite: bool = True):
        """ Copies the raw Input4Mips data to the processed path. This way the
        processers can operate savely on the data without modifying the original
        data.

        Args:
            overwrite (bool): If existing files should be overwritten. Default
                is True.
        """
        # we assume that the scenarios are stored in the first level!
        scenario_level = True

        # copy all dirs and files
        for root, dirs, files in tqdm(os.walk(self.raw_path), total=self.raw_len):
            # use only eligible scenarios
            if scenario_level:
                dirs = self.full_scenarios
                scenario_level = False

            # determine eligible scenarios
            eligible_scenarios = [bool(re.search(r'{0}$|{0}/|{0}\\'.format(scenario), root)) for scenario in self.full_scenarios]
            if any(eligible_scenarios):
                # create dirs
                # TODO do the same thing in resolution processing
                output_root = root.replace(str(self.raw_path), str(self.processed_path))
                self.create_output_dirs(output_root, dirs)

                output_root = Path(output_root)
                root = Path(root)

                # copy files from raw to processed directory
                for file in files:
                    self.copy_file(root/file, output_root/file, overwrite=overwrite)

    def sum_up_sectors_ds(self, ds: xr.Dataset, log_warnings: bool = True) -> bool:
        """ Summarizes all emissions that exist across different sectors for
        a single  dataset. Function changes the xarray dataset in place.

        Args:
            ds (xarray.Dataset): Dataset with emissions across different sectors.
            log_warnings (bool): If warnings should be printed. Default is True
                (i.e. warnings will be printed).
        Returns:
            bool: True if sectors has been updated, False if nothing was changed
        """
        # check if sectors exist
        if not self.sectors_exist(ds):
            if log_warnings: warnings.warn("...Warning: No sectors exist that could be summed up.")
            return False
        else:
            ds_ghg = ds.attrs["variable_id"]
            ds[ds_ghg] = ds[ds_ghg].sum("sector", skipna=True, min_count=1, keep_attrs=True)
            return True

    # TODO add a function that applies this to all files
    # Later: user can decide if only certain sectors should be dropped?
    def sum_up_sectors(self):
        """ Summarizes all emissions that exist across different sectors for
        a directory.
        """
        # run through all files
        for root, dirs, files in tqdm(os.walk(self.processed_path), total=self.processed_len):
            for file in files:
                file_path = Path(root) / Path(file)
                # open, sum-up, save new
                with xr.open_dataset(file_path) as ds:
                    sectors_were_updated = self.sum_up_sectors_ds(ds, log_warnings=False)
                    ds.load() # we must load the ds to be able to save it after the with open statement

                if sectors_were_updated:
                    ds.to_netcdf(file_path) # can only be done outside of with open!!

    # this is raw preprocessing (and needs to be done only one time)
    def aggregate_emissions(self):
        """ Summarizes all the emissions that are available within one scenario.
        E.g. BC_em_anthro, BC_em_biomassburning and BC_em_air are summarized to
        BC_em_total. This needs only to be done once.
        """
        # this has to happen AFTER the resolutions are matching each other
        pass

    # TODO move this to RES preprocessing
    def temp_interpolate_future_scenarios(self):
        """ Converting future scenarios with a 5y frequency to annual scenarios.
        """
        pass

    # TODO move to utils
    def space_name(self, name: str) -> str:
        """ Replaces '_' chars and replaces with space.
        Args:
            name (str): string or name that should be spaced
        Returns:
            str: new string with spaces instead of underscores
        """
        return name.replace('_', ' ')

    # TODO move to utils
    def underscore_name(self, name: str) -> str:
        """ Replaces spaces with underscores.
        Args:
            name (str): string or name that should be underscored
        Returns:
            str: new string with underscores instead of spaces
        """
        return name.replace(' ', '_')

    # TODO move to utils
    def create_output_dirs(
        self,
        root: str,
        dirs: List,
    ):
        """ Create directories from a list and a given (shared) root.
        Args:
            root (str): Root of the directories that should be created.
            dirs (list<str>): List of directories that should be created.
        """
        # makes only sense for non-empty list
        if len(dirs) > 0:
            # iterate over dir list
            for dir in dirs:
                # create path of output directory
                output_dir = os.path.join(root, dir)
                # create dir if it does not exist yet
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)

    def sectors_exist(self, ds: xr.Dataset) -> bool:
        """ Checks if sectors are still used as coordinate for
        the GHG variable.

        Args:
            ds (xarray.Dataset): xarray dataset that should be checked for
                sector coordinates
        Returns:
            bool: True if sectors are used as coordinates, False if not.
        """
        coords = list(ds[ds.attrs["variable_id"]].coords)
        return True if "sector" in coords else False

    # TODO move this to res_processer
    def spat_aggregate_dir(
        self,
        directory: Path,
        role_model_file: Path,
        store_dir_word: str = "processed",
        regridder_type: str = "bilinear",
        overwrite: bool = False,
    ):
        """ Spatially aggregates all files given in one directory according to a
        role-model-file (which has the right resolution, etc.). Stores the Resulting
        files in a given storage path. The function assumes that the aggregation
        happens along the dimension of the variable (inferred from directory Path).
        Must be nc files!

        Args:
            directory (Path): Directory which should be spatially aggregated.
            role_model_file (Path): Full path to an example file, the 'role model'. The aggregation
                happens such that the spatial resolution of this role model is matched.
            store_dir_word (str): In which subdir the processed data should be stored.
                Default is ``processed``. (Automatically stored on the same level like ``raw``).
            regridder_type (str): Which kind of regridder should be used
            overwrite (bool): If the files in the storage dir should be overwritten
                in case they already exist. Default: False
        """
        # checks
        if not os.path.isfile(role_model_file):
            raise ValueError("``role_model_file`` must be a file.")
        if Path(role_model_file).suffix != ".nc":
            raise ValueError("``role_model_file`` must be an .nc file (netCDF4).")
        if not os.path.isdir(directory):
            raise ValueError("``directory`` must be a directory.")

        # get output directory (should exist after the copy action raw2processed)
        raw_path_str = str(self.raw_path)
        processed_path_str = str(self.processed_path)
        output_directory = str(directory).replace(raw_path_str, processed_path_str)
        output_dir_len = self._count_files(Path(output_directory))

        # load role model & get its resolution
        role_model_ds = xr.open_dataset(role_model_file)
        new_res = self.underscore_name(role_model_ds.attrs["nominal_resolution"])
        # # sum over role model sectors
        # # TODO do this beforehand! (in raw preprocessing!)
        # if self.sectors_exist(role_model_ds):
        #     self.sum_up_sectors_ds(role_model_ds)

        # run through directory
        for root, dirs, files in tqdm(os.walk(directory), total=output_dir_len):
            # get output root
            output_root = root.replace(raw_path_str, processed_path_str)

            # skip all of this in case the new res already exists
            if not new_res in root:
                for file in files:
                    # load dataset
                    in_path = Path(root) / file
                    in_ds = xr.open_dataset(in_path)

                    # extract relevant information
                    old_res = self.underscore_name(in_ds.attrs["nominal_resolution"])

                    # create regridder file path & dirs
                    out_path = Path(output_root) / file
                    regridded_file_path = Path(str(out_path).replace(old_res, new_res))
                    exit(0)
                    # CONTINUE HERE
                        # rewrite this complete function
                        # remove 50km folder and check if it is still created
                        # remove sector exist part
                        # spat_aggregat should return a file
                        # save this file here

                    # create resolution dirs & subdirs if they dont exist yet
                    os.makedirs(regridded_file_path.parent, exist_ok=True)

                    # sum over sectors
                    # TODO do this beforehand! (raw preprocessing!)
                    if self.sectors_exist(in_ds):
                        self.sum_up_sectors_ds(in_ds)

                    self.spat_aggregate(
                        in_ds = in_ds,
                        role_model_ds = role_model_ds,
                        regridded_file_path = regridded_file_path,
                        regridder_type = regridder_type,
                        overwrite = overwrite,
                    )
                    in_ds.close()
                    # TODO remove this print statement later, find different way to track progress
                    print("Spatially aggregated the file and stored it!")

        role_model_ds.close()

    # QUESTION: How to handle the nans??
        # ? make nans to zeros beforehand (when processing single files)
        #in_ds = in_ds.fillna(value={in_var: 0})
        # -> if we remove nans, no landscape. For emissions it might make sense to keep nans??
    # TODO move this to "internal testing"
    def test_spat_aggregate(
        self,
        scenario: str = "historical",
        in_var_file_naming = "BC_em_biomassburning",
        in_var: str = "BC", # "BC_em_biomassburning" is only the file name, not how it is named in the dataset
        role_model_var: str = "BC_em_anthro",
        out_var_name: str = "", # if empty: same as in_var_file_naming
    ):
        # load files
        role_model_path = self.raw_path / scenario / role_model_var / "50_km" / "mon" / "1750"
        role_model_name = "input4mips_historical_BC_em_anthro_50_km_mon_gn_1750.nc"
        role_model_file = role_model_path / role_model_name
        in_path = self.raw_path / scenario / in_var_file_naming / "25_km" / "mon" / "1750"
        in_name = "input4mips_historical_BC_em_biomassburning_25_km_mon_gn_1750.nc"
        in_file = in_path / in_name
        in_ds = xr.open_dataset(in_file)
        role_model_ds = xr.open_dataset(role_model_file) # define a role model for the latitude and longitude values

        # get information from role_model and input file
        old_res = self.underscore_name(in_ds.attrs["nominal_resolution"])
        new_res = self.underscore_name(role_model_ds.attrs["nominal_resolution"])
        in_var_name = in_ds.attrs["variable_id"]

        # example, we could set the name of output variable
        #out_var_name = "BC_em_biomassburning"

        # make regridder file
        regridded_file_root = self.processed_path / scenario / out_var_name / new_res / "mon" / "1750"
        os.makedirs(regridded_file_root, exist_ok=True) # create dirs if they do not exist yet
        regridded_name = str(in_name).replace(old_res, new_res)
        regridded_file_path = regridded_file_root / regridded_name

        # aggregate
        self.spat_aggregate(in_ds, role_model_ds, regridded_file_path, out_var_name)

        # close files
        in_ds.close()
        role_model.close()

    # TODO Documentation
    # TODO can this only aggregate? or does it regrid in both directions?
    # TODO return a dataset - the file handling should be done elsewhere
    def spat_aggregate(
        self,
        in_ds: xr.Dataset,
        role_model_ds: xr.Dataset,
        regridded_file_path: Path,
        out_var_name: str = "", # if empty: same as variable naming from input file
        regridder_type: str = "bilinear",
        overwrite: bool = False,
    ):
        """ Spatially aggregates a single file according to the nominal resolution
        given by a role model dataset (``role_model_ds``). The input dataset
        (``in_ds``) is aggregated in-place, i.e. no dataset will be returned.
        Note: This function cannot regrid files with sectors.

        Args:
            in_ds (xarray.Dataset): The input dataset that should be regridded.
            role_model_ds (xarray.Dataset): The role model - the output file
                should look like the role model resolution-wise in the end.
                The longitude-latitude structure from the role model is used to
                build the new grid.
            regridded_file_path (Path): Path where the regridded file should be
                stored.
            out_var_name (str): Can be changed in case the variable in the
                regridded name should be different from the variable naming
                of the input file. With default (empty string) the file will have
                the same variable naming as listed as attribute ``variable_id``
                in the input file.
            regridder_type (str): See xesmf for that. All regridder types listed
                there can be used here. Default is ``bilinear``.
            overwrite (bool): Indicates if the regridded file should be overwritten
                in case the file (from ``regridded_file_path``) already exists.
        """
        # check if sectors exist -> raise error if role_model or in_ds have sectors
        if self.sectors_exist(in_ds):
            raise ValueError("Argument ``in_ds`` is not allowed to contain sectors. Sum over sectors before.")
        if self.sectors_exist(role_model_ds):
            raise ValueError("Argument ``role_model_ds`` is not allowed to contain sectors. Sum over sectors before.")

        # get relevant information
        in_var = in_ds.attrs["variable_id"]
        new_res = role_model_ds.attrs["nominal_resolution"]

        # how the output variable should be named
        if len(out_var_name) < 1:
            out_var_name = in_var

        # create the output dataset in the right shape
        out_ds = xr.Dataset({"lat": (["lat"], role_model_ds.lat.values, {"units": "degrees_north"}),
                             "lon": (["lon"], role_model_ds.lon.values, {"units": "degrees_east"}),
                            })

        # regrid
        regridder = xe.Regridder(in_ds, out_ds, regridder_type)
        regridded_ds = regridder(in_ds, keep_attrs=True)

        # rename GHG vars & nominal resolution
        regridded_ds = regridded_ds.rename_vars({in_var: out_var_name})
        regridded_ds.attrs["nominal_resolution"] = new_res
        regridded_ds.attrs["variable_id"] = out_var_name

        # safe regridded data to file
        if (not os.path.isfile(regridded_file_path)) or (overwrite):
            regridded_ds.to_netcdf(regridded_file_path, mode="w", format="NETCDF4")
            print("Saved the regridded file {}".format(regridded_file_path))
        else:
            print("Skipping file {}".format(regridded_file_path))


    # TODO clean this up
    def spat_aggregate_plot_example(
        self,
        old_res: str = "25_km",
        new_res: str = "50_km",
        scenario: str = "historical",
        var: str = "BC_em_biomassburning", # TODO make this a list option
        overwrite: bool = False,
    ):
        """ Testing the xesmf regridder
        """
        # testing here how to aggregate a normal file, this is a 50km one
        scenario = "historical"
        a_var = "BC_em_anthro"
        abbr_var = "BC" # used instead of "BC_em_biomassburning" in the files!
        a_path = self.raw_path / scenario / a_var / "50_km" / "mon" / "1750"
        a_file = a_path / "input4mips_historical_BC_em_anthro_50_km_mon_gn_1750.nc"
        b_path = self.raw_path / scenario / var / "25_km" / "mon" / "1750"
        b_file = b_path / "input4mips_historical_BC_em_biomassburning_25_km_mon_gn_1750.nc"
        new_file_path = self.raw_path / "None" / "test_file.nc"

        # resolution ratio (old / new res)
        res_ratio = int(old_res.split('_')[0]) / int(new_res.split('_')[0])
        # TODO!!! Desired degree resolution
        res_degree = 0.5
        # aggregation size
        aggr_size = res_ratio**(-1)

        # open both files
        in_ds = xr.open_dataset(b_file)
        role_model_ds = xr.open_dataset(a_file)

        ##### MOVE THIS OUTSIDE ###############
        # sectors
        old_sectors = role_model_ds.sizes["sector"]
        # TODO make this a user param
        aggr_sectors = True # True (default): summarize all sectors to one; False: leave them as it is
        new_sectors = 1 if aggr_sectors else old_sectors # we only have 1 sector for biomassburning

        if (not aggr_sectors) and (not "sector" in in_ds):
            raise ValueError("If you do not want to aggregate sectors, sectors must exists in the high res file! Consider setting aggr_sectors=True.")

        ### create new nc file with lower res for lon and lat ###
        # copy the original dataset (desired dimensions etc)
        if new_sectors < old_sectors: # change the sector dimension if necessary
            out_ds = role_model_ds.where(role_model_ds.sector < new_sectors).dropna(dim="sector")
        elif new_sectors > old_sectors:
            raise ValueError("Trying to create more sectors than available in original file. We are not able to do this.")
        else:
            out_ds = role_model_ds

        # replace with nans
        out_ds[a_var][:, :, :, :] = np.nan

        # rename GHG variable (target var that changes resolution!) if needed
        if var != a_var:
            out_ds[var] = out_ds[a_var]
            out_ds = out_ds.drop(a_var)
        ##############################################

        # CONTINUE HERE: what is the state of the sectors? sectors are missing in the new file

        # print(in_ds)
        # print(out_ds)
        # exit(0)

        # data array that we would like to regrid:
        # in_array = in_ds["BC"]
        # structure we would like to have for the output
        out_ds_test = xr.Dataset({"lat": (["lat"], role_model_ds.lat.values, {"units": "degrees_north"}),
                                  "lon": (["lon"], role_model_ds.lon.values, {"units": "degrees_east"}),
                                 })
        # regridder = xe.Regridder(in_ds, out_ds_test, "bilinear")
        # regridded_array = regridder(in_array, keep_attrs=True)
        # regridded_file = in_ds
        # regridded_file["BC"] = regridded_array
        # print(regridded_file)
        # exit(0)
        print("Role Model")
        print(role_model_ds)
        role_model_ds.sel(time="1750-01-16 00:00:00", sector=4)["BC_em_anthro"].squeeze().plot.pcolormesh(vmin=0, vmax=5e-11)
        plt.show()
        print("Input File")
        print(in_ds)
        print(in_ds["BC"][0:10, 0:10, 0:10])
        in_ds.sel(time="1750-01-16 12:00:00")["BC"].squeeze().plot.pcolormesh(vmin=0, vmax=5e-11)
        plt.show()
        print("Regridder")
        regridder = xe.Regridder(in_ds, out_ds_test, "bilinear")
        print("Output File")
        regridded_file = regridder(in_ds, keep_attrs=True)
        print(regridded_file["BC"][0:10, 0:10, 0:10])
        regridded_file.sel(time="1750-01-16 12:00:00")["BC"].squeeze().plot.pcolormesh(vmin=0, vmax=5e-11)
        plt.show()
        # TODO update nominal resolution or add attribute: current_resolution:...

        exit(0)
        # Bug analysis:
            # time is missing / not as expected
            # sectors are missing
        # hence: cannot be plotted
        regridded_file.sel(time="1750-01-16 00:00:00")["BC"].squeeze().plot.pcolormesh(vmin=0, vmax=5e-11)
        plt.show()
        # BUG BC has only nans. Answer: hm, input has nans, output has nans, makes sense to me
        # TODO adapt nans to 0 beforehand
        exit(0)

        # close xarray files
        in_ds.close()
        role_model.close()


class Input4mipsResPreprocesser:
    """ Responsible for all Input4mips data peprocessing connected to resolutions.
    This class might be called by the user and might be called several times.
    """
    def __init__(
        self,
        raw_path: Path,
        processed_path: Path,
        test_scenario: bool,
    ):
        """ Initialize this class
        Args:
            raw_path (Path): where the downloaded input4mips data lives
            processed_path (Path): where the preprocessed data should be stored
            test_scenario (bool): if you are in a testing scenario, only one file is considered
        """
        # name the vars you want to work with!
            # prints which scenarios are not having all the vars (skips them)
        # makes a list which folders are traversed (other ones are ignored)

        # create a preprocessed dir on the level above (default) path, or where the users want
        pass

    def print_res_inconsistencies():
        """ function that lists all spatial and temporal resolution inconsistencies
        """
        pass

    # we do not have an interpolation function, because input4mips usually has
    # always higher res than CMIP6
    def aggregate_spatial_dim(
        self,
        target_res: int = 250,
        source_res: int = 50,
        overwrite: bool = False,
    ):
        """ Aggregates along the spatial dimension. Target resolution and source
        resolution must be given.
        Args:
            target_res (int): in kms. Default value is 250 (typical CMIP6 res)
            source_res (int): in kms. The resolution that we are aggregating from.
            overwrite (bool): If the target_res already exists, should the value be overwritten?
        """
        # check if source_res is available

        # check if target res already exists (exit function if overwrite true and print a message)

        # create new folder for the new res

        # use an external file-traversing function (in utils maybe?)

            # call an aggregation functions from utils or here (we need to divide between math functions, spatial_wrapper and temporal_wrapper)

            # store the summarized values in the new folder
        pass

    # uhm, actually this should be part of raw-preprocessing since this is about the frequency - the temporal resolution within that year is still monthly
    def interpolate_temporal_dim(
        self,
        target_res: str = "annual",
        source_res: str = "5y",
        interpolation_func: str = "linear",
        overwrite: bool = False,
    ):
        """ Since Inpu4MIPs data is only available with 5y gaps, this can be
        interpolated to an annual frequency. This is added

        Args:
            target_res (str): The temporal resolution you would like to have.
            source_res (str): The temporal resolution currently available.
            interpolation_func (str): Which kind of interpolation func you would like to use.
                Must be available in mother_data/utils/interpolations.py.
            overwrite (bool): In case the target_res already exists, should the files be overwritten?
        """
        pass

# Big processer class:
# - alls subclasses: same attributes & some functions they share
# - everyone has to implement "run" with boolean guided instructions
if __name__ == '__main__':
    # create & run raw preprocesser
    raw_preprocesser = Input4mipsRawPreprocesser(
        raw_path=Path(DATA_PATH),
        processed_path=Path(PROCESSED_PATH),
        test_scenario=True,
        ghg_vars=VARS)

    # test something
    # test_file = Path(raw_preprocesser.processed_path / "historical"/ "BC_em_biomassburning" / "50_km" / "mon" / "1750" / "input4mips_historical_BC_em_biomassburning_50_km_mon_gn_1750.nc")
    # ds = xr.open_dataset(test_file)
    # print(ds.attrs["variable_id"])
    # print(ds)
    # print(raw_preprocesser.sectors_exist(ds))
    # print("finished")
    # exit(0)

    # raw_preprocesser.run(
    #     sanity_checking = False,
    #     create_processed_dir = False,
    #     sum_over_sectors = True,
    # )
    # exit(0)

    # TODO Plotting must be moved outside! (different responsibility)
    #raw_preprocesser.spat_aggregate_plot_example()\

    # TODO move this to a "resolution" preprocesser
    # aggregate all the historical openburnings (they have 25km res instead of 50)
    #raw_preprocesser.test_spat_aggregate()
    # define role model and which dir should be processed
    role_model_file = Path(raw_preprocesser.raw_path / "historical"/ "BC_em_anthro" / "50_km" / "mon" / "1750" / "input4mips_historical_BC_em_anthro_50_km_mon_gn_1750.nc")
    directory = Path(raw_preprocesser.raw_path / "historical")
    raw_preprocesser.spat_aggregate_dir(directory, role_model_file, overwrite=True)
