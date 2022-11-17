# Input4MIPs data is preprocessed here after downloading them
import os
import cftime
import numpy as np
import xesmf as xe
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from pathlib import Path
from cftime import num2date, date2num


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
DATA_PATH = "/home/julia/Documents/Master/CausalSuperEmulator/Code/causalpaca/data/data/input4mips/"
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
    # TODO calendar checks
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

    def temp_interpolate_future_scenarios(self):
        """ Converting future scenarios with a 5y frequency to annual scenarios.
        """
        pass

    def spat_aggregate(
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
        aggr_sectors = True # True (default): summarize all sectors to one False: leave them as it is
        new_sectors = 1 if aggr_sectors else old_sectors # we only have 1 sector for biomassburning

        if (not aggr_sectors) and (not "sector" in high_res_file):
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

        # CONTINUE HERE: try to create a new file with just long and lat?
        out_ds_test = xr.Dataset({"lat": (["lat"], role_model_ds.lat.values),
                                  "lon": (["lon"], role_model_ds.lon.values),
                                 })
        print("Role Model")
        role_model_ds.sel(time="1750-01-16 00:00:00", sector=4)["BC_em_anthro"].squeeze().plot.pcolormesh(vmin=0, vmax=5e-11)
        plt.show()
        print("Input File")
        print(in_ds)
        regridder = xe.Regridder(in_ds, out_ds_test, "bilinear")
        print("Regridder")
        print(regridder)
        regridded_file = regridder(in_ds)
        print("Output File")
        print(regridded_file)
        # Bug analysis:
            # time is missing / not as expected
            # sectors are missing
        # hence: cannot be plotted
        regridded_file.sel(time="1750-01-16 00:00:00")["BC"].squeeze().plot.pcolormesh(vmin=0, vmax=5e-11)
        plt.show()
        # BUG BC has only nans
        exit(0)

    # TODO add option to use different kinds of aggregations
    # TODO this function can be used in the res preprocesser as well
    # -> find an elegant, generalized solution for that
    # ATTENTION right now this only works for mapping from X km to Y km,
         # where Y is 2-fold number of X! (2-times, 4-times, etc.)
    def spat_aggregate_single_var(
        self,
        old_res: str = "25_km",
        new_res: str = "50_km",
        scenario: str = "historical",
        var: str = "BC_em_biomassburning", # TODO make this a list option
        overwrite: bool = False,
    ):
        """ Spatial aggregation of higher res openburning files to the standard
        nominal resolution used for the other files. This function can be used
        multiple times for different desired resolutions. This function is
        intended to use to spatiall aggregate historical openburning files.
        The function can be used equally for other scenarios if necessary
        (see Args). The desired nominal resolution must be given as argument
        (Number_unit: [50_km]).

        Args:
            old_res (str): Resolution the files currently have. E.g. "25 km".
            new_res (str): Resolution the files should have. E.g. "50 km"
            scenario (str): A subdir such as "historical" that describes a
                scenario run by the climate models.
            var (str): Which variable is considered within the scenario. E.g.
                "BC_em_biomassburning".
            overwrite (bool): Indicating if the files should be overwritten if
                they already exist for this resolution.
        """
        # TODO rename longitude and latitude -> must be consistent for all files

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
        high_res_file = xr.open_dataset(b_file)
        full_original = xr.open_dataset(a_file)

        # sectors
        old_sectors = full_original.sizes["sector"]
        # TODO make this a user param
        aggr_sectors = True # True (default): summarize all sectors to one False: leave them as it is
        new_sectors = 1 if aggr_sectors else old_sectors # we only have 1 sector for biomassburning

        if (not aggr_sectors) and (not "sector" in high_res_file):
            raise ValueError("If you do not want to aggregate sectors, sectors must exists in the high res file! Consider setting aggr_sectors=True.")

        ### create new nc file with lower res for lon and lat ###
        # copy the original dataset (desired dimensions etc)
        if new_sectors < old_sectors: # change the sector dimension if necessary
            copy_original = full_original.where(full_original.sector < new_sectors).dropna(dim="sector")
        elif new_sectors > old_sectors:
            raise ValueError("Trying to create more sectors than available in original file. We are not able to do this.")
        else:
            copy_original = full_original

        # replace with nans
        copy_original[a_var][:, :, :, :] = np.nan

        # rename GHG variable (target var that changes resolution!) if needed
        if var != a_var:
            copy_original[var] = copy_original[a_var]
            copy_original = copy_original.drop(a_var)

        #############################################

        # open both nc files
        a = full_original # TO DEL
        b = copy_original # TO DEL
        #xr.open_dataset(b_file) # TODO: grap first available file here
        #print(a["BC_em_anthro"][11, :, 359, 719]) # month (12), sector (8), lat (360), lon (720)
        #print(b["BC"][11, 719, 1439]) # time (12), lon (720), lat (1440)
        #print(b["BC"][6, 200:500, 1000:1200].values) # time (12), lon (720), lat (1440)

        aggr_file = copy_original

        # replace nans with zeros
        high_res_file = high_res_file.where(~np.isnan(high_res_file[abbr_var][:, :, :]), 0) # later: could be accelerated

        # TODO make this is a looot faster ...
            # calculate lon lat block once separately
            # find a way to use an "apply" method that is faster


        # move over high res file, aggregate and fill the new low res file
        for i_lon, lon in tqdm(enumerate(aggr_file.lon.values), total=aggr_file.lon.size):
            mid_lon = (aggr_size * i_lon) + (0.5 * aggr_size)
            str_lon = int(mid_lon - (0.5 * aggr_size))
            end_lon = int(mid_lon + (0.5 * aggr_size))
            for i_lat, lat in enumerate(aggr_file.lat.values):
                mid_lat = (aggr_size * i_lat) + (0.5 * aggr_size)
                str_lat = int(mid_lat - (0.5 * aggr_size))
                end_lat = int(mid_lat + (0.5 * aggr_size))
                for i_t, t in enumerate(aggr_file.time.values): # maybe use without values here
                    if not aggr_sectors:
                        # another for loop over sectors
                        for i_s, s in enumerate(aggr_file.sector.values):
                            high_res_indices = dict(time=i_t,
                                                    sector=i_s,
                                                    latitude=slice(str_lat, end_lat),
                                                    longitude=slice(str_lon, end_lon))
                            high_res_values = high_res_file[abbr_var][high_res_indices]
                            aggr_value = high_res_values.sum().values
                            low_res_coord_labels = dict(time=t, lat=lat, lon=lon, sector=s)
                            aggr_file[var].loc[low_res_coord_labels] = aggr_value
                    else:
                        # Attention: problems can arise when names are not "longitude" & "latitude"
                        high_res_indices = dict(time=i_t,
                                                latitude=slice(str_lat, end_lat),
                                                longitude=slice(str_lon, end_lon))
                        high_res_values = high_res_file[abbr_var][high_res_indices]
                        aggr_value = high_res_values.sum().values
                        # check if coordinate sector exists
                        if "sector" in aggr_file.coords:
                            low_res_coord_labels = dict(time=t, lat=lat, lon=lon, sector=0)
                        else:
                            low_res_coord_labels = dict(time=t, lat=lat, lon=lon)
                        aggr_file[var].loc[low_res_coord_labels] = aggr_value
                        #print(aggr_file[var].loc[low_res_coord_labels])

        print("Finished!")
        print(aggr_file)

        # save as new nc file
        aggr_file.to_netcdf(new_file_path)
        exit(0)
        curr_path = self.raw_path / scenario / var

        # check if old res exists
        print(curr_path / old_res)

        # check if new res exists (if overwrite false, exit the function)

        # create folder if new res does not exist yet

        # iterate through subdirs and files

            # create the same subdirs (if they dont exist yet)

            # aggregate the single file (from old dir) to desired res
                # ... do stuff ...

            # save as new file in the new dir

        # finished
        pass

    # this is raw preprocessing (and needs to be done only one time)
    def aggregate_emissions(self):
        """ Summarizes all the emissions that are available within one scenario.
        E.g. BC_em_anthro, BC_em_biomassburning and BC_em_air are summarized to
        BC_em_total. This needs only to be done once.
        """
        pass

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
        test_scenario=True,
        ghg_vars=VARS)

    # aggregate all the historical openburnings (they have 25km res instead of 50)
    raw_preprocesser.spat_aggregate(
        old_res="25_km",
        new_res="50_km",
        scenario="historical",
        var="BC_em_biomassburning", # TODO make this a list option
        overwrite=False)
