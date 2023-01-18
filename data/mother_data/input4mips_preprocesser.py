# Input4MIPs data is preprocessed here after downloading them
import os
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

ACCELERATE = True

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

        if not ACCELERATE: # TODO delete
            self.sanity_checks()

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
                        # optional check that could be added: does the root name (50_km/mon/) match with file names?
                        spat_res = file.split('_')[-5] + ' ' + file.split('_')[-4]
                        temp_res = file.split('_')[-3]
                        # read the file and check resolutions
                        ds = xr.open_dataset(root + '/' + file)
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
        print("Checks ended successfully!")
        return True

    # this is raw preprocessing (and needs to be done only one time)
    def aggregate_emissions(self):
        """ Summarizes all the emissions that are available within one scenario.
        E.g. BC_em_anthro, BC_em_biomassburning and BC_em_air are summarized to
        BC_em_total. This needs only to be done once.
        """
        pass

    # # TODO move to utils
    # def sum_xarray(ds, dim):
    #     """ Sums over a given dimension
    #     Args:
    #         ds (xarray.DataSet): xarray dataset
    #         dim (int): Dimension over which should be summed
    #     Returns:
    #
    #     """



    def sum_up_sectors(self, ds):
        """ Summarizes all emissions that exist across different sectors.
        Function changes the xarray dataset in place.

        Args:
            ds (xarray.DataSet): Dataset with emissions across different sectors.
        """
        # check if sectors exist
        if "sector" not in ds.dims:
            warnings.warn("...Warning: No sectors exist that could be summed up.")

        ds_ghg = ds.attrs["variable_id"]
        ds[ds_ghg] = ds.SO2_em_anthro.sum("sector", skipna=True, min_count=1, keep_attrs=True)


    # TODO move this to RES preprocessing
    def temp_interpolate_future_scenarios(self):
        """ Converting future scenarios with a 5y frequency to annual scenarios.
        """
        pass

    # TODO move to utils
    def space_name(
        self,
        name
    ):
        """ Replaces '_' chars and replaces with space.
        Returns:
            str: new string with spaces instead of underscores
        """
        return name.replace('_', ' ')

    # TODO move to utils
    def underscore_name(
        self,
        name
    ):
        """ Replaces spaces with underscores.
        Returns:
            str: new string with underscores instead of spaces
        """
        return name.replace(' ', '_')

    def spat_aggregate_dir(
        self,
        #dir: Path,
        #role_model_file: Path,
        #store_path: Path,
        regridder_type: str = "bilinear",
        overwrite: bool = False,
    ):
        """ Spatially aggregates all files given in one directory according to a
        role-model-file (which has the right resolution, etc.). Stores the Resulting
        files in a given storage path. The function assumes that the aggregation
        happens along the dimension of the variable (inferred from directory Path).
        Must be nc files!

        Args:
            dir (Path): Directory which should be spatially aggregated.
            role_model_file (Path): Full path to an example file, the 'role model'. The aggregation
                happens such that the spatial resolution of this role model is matched.
            store_path (Path): Where the new aggregated files should be stored
            regridder_type (str): Which kind of regridder should be used
            overwrite (bool): If the files in the storage dir should be overwritten
                in case they already exist. Default: False
        """
        # test path
        scenario = "historical"
        role_model_var = "BC_em_anthro"
        role_model_path = self.raw_path / scenario / role_model_var / "50_km" / "mon" / "1750"
        role_model_name = "input4mips_historical_BC_em_anthro_50_km_mon_gn_1750.nc"
        role_model_file = role_model_path / role_model_name
        directory = Path(self.raw_path / "historical")

        # checks
        if not os.path.isfile(role_model_file):
            raise ValueError("``role_model_file`` must be a file.")
        if Path(role_model_file).suffix != ".nc":
            raise ValueError("``role_model_file`` must be an .nc file (netCDF4).")
        if not os.path.isdir(directory):
            raise ValueError("``directory`` must be a directory.")

        # load role model
        role_model_ds = xr.open_dataset(role_model_file)
        # extract relevant information from role_model
        new_res = self.underscore_name(role_model_ds.attrs["nominal_resolution"])

        # run through directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # load dataset
                in_ds = xr.open_dataset(os.path.join(root, file))
                # extract relevant information
                old_res = self.underscore_name(in_ds.attrs["nominal_resolution"])
                in_var = in_ds.attrs["variable_id"]
                in_var_file_naming = '_'.join(file.split('_')[2:5])
                sectors_exist = True if "sector" in in_ds.dims else False

                # sum over sectors (should I do that here??)
                if sectors_exist:
                    self.sum_up_sectors(in_ds)

                exit(0)
                # aggregate
                # old_res CHECK
                # new_res CHECK
                # regridder type CHECK
                # overwrite CHECK
                # sectors_exist = False

                # new ones: in_file, rolde_model_file, regridded_file (Path)

                self.spat_aggregate(
                    old_res = old_res, # TODO can be done within spat_aggregate
                    new_res = new_res, # TODO can be done within spat_aggregate
                    in_ds = in_ds, # TODO two cases: 
                    regridder_type = regridder_type,
                    overwrite = overwrite,
                    sectors_exist = False,
                )
                exit(0)
                # TODO change in spat_aggregate:
                # divide between the "Path" and "xarray" case


            # # extract relevant information from single file
            # # example: input4mips_historical_BC_em_biomassburning_25_km_mon_gn_1750.nc
            # old_res # from file name or from file??
            # in_var # from file itself
            # sectors_exist # from file itself (in or role model??)
            #
            # new_store_path # create that one
            #
            # # call spat_aggregate

    # TODO Documentation
    # TODO clean up / add simple file name args
    # TODO sectors_exist??
    # TODO include call the aggregate sectors
    # TODO overwrite -> implement before safing!
    def spat_aggregate(
        self,
        old_res: str = "25_km",
        new_res: str = "50_km",
        scenario: str = "historical",
        in_var_file_naming = "BC_em_biomassburning",
        in_var: str = "BC", # "BC_em_biomassburning" is only the file name, not how it is named in the dataset
        role_model_var: str = "BC_em_anthro",
        out_var_name: str = "", # if empty: same as in_var_file_naming
        regridder_type: str = "bilinear",
        overwrite: bool = False,
        sectors_exist: bool = False,
    ):
        """ xesmf regridder large scale.

        Notes: This function cannot regrid files with sectors. It includes
        a call to aggregate sectors - please put [sectors_exist = False] if you
        already aggregated the files and no sectors exists
        """
        # stuff that needs to be set / calculated in the beginning
        if len(out_var_name) < 1:
            out_var_name = in_var_file_naming

        # Here: apply this to all files (create a large scale and a generic regridder function (put that in utils!!))
        # right now: regridd a single file

        # load files
        # TODO adapt for multiple files
        # TODO move this all out of this function!!
        role_model_path = self.raw_path / scenario / role_model_var / "50_km" / "mon" / "1750"
        role_model_name = "input4mips_historical_BC_em_anthro_50_km_mon_gn_1750.nc"
        role_model_file = role_model_path / role_model_name
        in_path = self.raw_path / scenario / in_var_file_naming / "25_km" / "mon" / "1750"
        in_name = "input4mips_historical_BC_em_biomassburning_25_km_mon_gn_1750.nc"
        in_file = in_path / in_name
        regridded_file_path = self.processed_path / scenario / out_var_name / new_res / "mon" / "1750"
        os.makedirs(regridded_file_path, exist_ok=True) # create dirs if they do not exist yet
        regridded_name = str(in_name).replace(old_res, new_res)
        regridded_file = regridded_file_path / regridded_name

        in_ds = xr.open_dataset(in_file)
        role_model_ds = xr.open_dataset(role_model_file) # define a role model for the latitude and longitude values

        # QUESTION: How to handle the nans??
            # ? make nans to zeros beforehand (when processing single files)
            #in_ds = in_ds.fillna(value={in_var: 0})
            # -> if we remove nans, no landscape. For emissions it might make sense to keep nans??

        # create the output dataset in the right shape
        out_ds = xr.Dataset({"lat": (["lat"], role_model_ds.lat.values, {"units": "degrees_north"}),
                             "lon": (["lon"], role_model_ds.lon.values, {"units": "degrees_east"}),
                            })

        # regrid
        regridder = xe.Regridder(in_ds, out_ds, regridder_type)
        regridded_ds = regridder(in_ds, keep_attrs=True)

        # rename GHG vars & nominal resolution
        regridded_ds = regridded_ds.rename_vars({in_var: out_var_name})
        regridded_ds.attrs["nominal_resolution"] = self.space_name(new_res)

        # safe regridded data to file
        regridded_ds.to_netcdf(regridded_file, mode="w", format="NETCDF4")
        print("Saved the regridded file!")


# TODO sector aggregation functionalities:
# TWO functions: one for single aggregation (utils), one applying it to all
# user can decide if certain sectors should be dropped
# simplest version: all sectors are aggregated
# finds automatically what the sectors are (different variable names?)
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

if __name__ == '__main__':
    # create raw preprocesser
    raw_preprocesser = Input4mipsRawPreprocesser(
        raw_path=Path(DATA_PATH),
        processed_path=Path(PROCESSED_PATH),
        test_scenario=True,
        ghg_vars=VARS)

    #raw_preprocesser.spat_aggregate_plot_example()\

    # aggregate all the historical openburnings (they have 25km res instead of 50)
    # raw_preprocesser.spat_aggregate(
    #     old_res="25_km",
    #     new_res="50_km",
    #     scenario="historical",
    #     in_var_file_naming="BC_em_biomassburning",  # TODO make this a list option
    #     in_var="BC", # TODO this is a problem - should be aligned with in_var_file_naming...
    #     overwrite=False)
    raw_preprocesser.spat_aggregate_dir()
    print("hello end")
