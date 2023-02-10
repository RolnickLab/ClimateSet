import os
import pint
import h5netcdf
import numpy as np
import xarray as xr
import netCDF4 as nc

from pathlib import Path
from email.policy import default

from data_generation.parameters.constants import RES_TO_CHUNKSIZE
from data_generation.parameters.data_paths import RAW_DATA, PROCESSED_DATA, LOAD_DATA


overwrite = False

ureg = pint.UnitRegistry()

# mother preprocesser: only used once (by us), never used by user, basic preprocessing steps

# preprocesses high-resolution data

# three cases:
# input external var
# input model internal var
# output model

# holt sich daten von RAW und steckt sie in PREPROCESSED

# TODO: unit check

# class
# params:
# data_source: which data should be raw_processed
# data_store: where the cleaned up and structured data is put (PROCESSED)
# LATER: add further raw_processing params from mother_params if that should become necessary

# functions:

# clean_up() data such that data is in the finest resolution and raedy for res_preprocesser

# structure() data in PREPROCESSED (clear names, structure, etc)


class RawProcesser:
    def __init__(self, source: Path, store: Path):
        """
        Init method for the RawProcesser
        @params:
            source (Path): Which directory should be processed. Freshly downloaded data was stored there.
            store (Path): Where to store the raw-processed files
        """
        self.source = source
        self.store = store

        print(f"Source data: {self.source}")
        print(f"Store data: {self.store}")
        # TODO integrate file-internal check if data was already raw_processed
        self.processed_flag = False

        # checks data and returns list of corrupt files
        corrupt_files = self.check()
        print("corrupt files")
        print(corrupt_files)
        # self.processed_flag = False
        # self.check_processed()
        self.process(corrupt_files)

    def check_processed(self):
        """Checks if the data was already processed to prevent unnecessary processing.
        Operates on self.source and stored outcome in self.processed_flag
        """
        # TODO do checks (e.g. data already exists in PROCESSED, so we don't need to process it again)
        # set processed_flag to right boolean
        raise NotImplementedError

    def process(self, corrupt_files, check_time=True):
        """Makes all the first and prior processing steps.

        TODO:single-model mode so far, deal with different models?(in storing hierachy)"""

        if not self.processed_flag:
            i = 0
            #    # testing:
            #    # select some file
            corrupt_files = []

            for project in os.listdir(self.source):
                print(project)
                # part 1:
                if project == "input4mips":

                    for experiment in os.listdir(self.source + "/" + project):
                        for var in os.listdir(
                            self.source + "/" + project + "/" + experiment
                        ):

                            # for each var the resolution should be consistent

                            for i, nom_res in enumerate(
                                os.listdir(
                                    self.source
                                    + "/"
                                    + project
                                    + "/"
                                    + experiment
                                    + "/"
                                    + var
                                )
                            ):

                                for freq in os.listdir(
                                    self.source
                                    + "/"
                                    + project
                                    + "/"
                                    + experiment
                                    + "/"
                                    + var
                                    + "/"
                                    + nom_res
                                ):

                                    for y in os.listdir(
                                        self.source
                                        + "/"
                                        + project
                                        + "/"
                                        + experiment
                                        + "/"
                                        + var
                                        + "/"
                                        + nom_res
                                        + "/"
                                        + freq
                                    ):

                                        file_dir = (
                                            self.source
                                            + "/"
                                            + project
                                            + "/"
                                            + experiment
                                            + "/"
                                            + var
                                            + "/"
                                            + nom_res
                                            + "/"
                                            + freq
                                            + "/"
                                            + y
                                            + "/"
                                        )
                                        # print('file dir', file_dir)

                                        try:
                                            file_names = os.listdir(file_dir)
                                            # print("file names", file_names)
                                            if len(file_names) > 1:
                                                print(
                                                    "WARNING: Multiple files exist where only one shouldb be."
                                                )
                                                print(file_dir)
                                                print(
                                                    "only considering first file:",
                                                    file_name,
                                                )
                                            file_name = file_names[0]
                                        except IndexError:
                                            print(
                                                "WARNING: apparently no data file available. Skipping."
                                            )
                                            print(file_dir)
                                            continue
                                        except TypeError:
                                            print(
                                                "WARNING: apparently no data file available. Skipping."
                                            )
                                            print(file_dir)
                                            continue

                                        # chunksize
                                        chunksize = RES_TO_CHUNKSIZE[freq]

                                        # check if corrupt, if yes, continue
                                        if (file_dir + file_name) in corrupt_files:
                                            print("File in corrupt files. Skipping")
                                            continue
                                        data, corrupt = self.check_corruptness(
                                            file_dir,
                                            file_name,
                                            chunksize,
                                            var,
                                            check_time=False,
                                        )
                                        if corrupt:
                                            corrupt_files.append(file_dir + file_name)
                                            print("File corrupt. Skipping")
                                            continue

                                        write_dir = file_dir = (
                                            self.store
                                            + "/"
                                            + project
                                            + "/"
                                            + experiment
                                            + "/"
                                            + var
                                            + "/"
                                            + nom_res
                                            + "/"
                                            + freq
                                            + "/"
                                            + y
                                            + "/"
                                        )
                                        path = write_dir
                                        print(path)

                                        # check if path exist, create path if necessary

                                        isExist = os.path.exists(path)

                                        if not isExist:

                                            # Create a new directory because it does not exist
                                            os.makedirs(path)
                                            print("The new directory is created!", path)

                                        outfile = path + file_name.replace(".nc", ".h5")
                                        print(outfile)

                                        if (not overwrite) and os.path.isfile(outfile):
                                            print(
                                                f"File {outfile} already exists, skipping."
                                            )
                                            continue

                                        # preprocessing stuff
                                        # do stuff
                                        # 1. deal with sectors
                                        # store it with same hierachy structure in a preprocessed foleder (self.store)
                                        print("preprocessing")

                                        print("loading")
                                        ds = data.load()
                                        print("summing over sectors")
                                        ds = self.sum_over_sectors(ds)
                                        print(ds)
                                        print("saving")
                                        ds.to_netcdf(outfile, engine="h5netcdf")

                else:
                    print("cmip6")
                    # continue

                    for model in os.listdir(self.source + "/" + project):

                        print("model", model)

                        for ensemble_member in os.listdir(
                            self.source + "/" + project + "/" + model
                        ):

                            for experiment in os.listdir(
                                self.source
                                + "/"
                                + project
                                + "/"
                                + model
                                + "/"
                                + ensemble_member
                            ):
                                print("experiment", experiment)
                                for var in os.listdir(
                                    self.source
                                    + "/"
                                    + project
                                    + "/"
                                    + model
                                    + "/"
                                    + ensemble_member
                                    + "/"
                                    + experiment
                                ):
                                    print("var", var, "resetting default")
                                    var_default_unit = ""

                                    # for each var the resolution should be consistent

                                    for nom_res in os.listdir(
                                        self.source
                                        + "/"
                                        + project
                                        + "/"
                                        + model
                                        + "/"
                                        + ensemble_member
                                        + "/"
                                        + experiment
                                        + "/"
                                        + var
                                    ):
                                        # print(f"i {i} nom {nom_res}")

                                        for freq in os.listdir(
                                            self.source
                                            + "/"
                                            + project
                                            + "/"
                                            + model
                                            + "/"
                                            + ensemble_member
                                            + "/"
                                            + experiment
                                            + "/"
                                            + var
                                            + "/"
                                            + nom_res
                                        ):
                                            # print(f"freq {freq}")

                                            for y in os.listdir(
                                                self.source
                                                + "/"
                                                + project
                                                + "/"
                                                + model
                                                + "/"
                                                + ensemble_member
                                                + "/"
                                                + experiment
                                                + "/"
                                                + var
                                                + "/"
                                                + nom_res
                                                + "/"
                                                + freq
                                            ):
                                                # print(f"year {y}")

                                                file_dir = (
                                                    self.source
                                                    + "/"
                                                    + project
                                                    + "/"
                                                    + model
                                                    + "/"
                                                    + ensemble_member
                                                    + "/"
                                                    + experiment
                                                    + "/"
                                                    + var
                                                    + "/"
                                                    + nom_res
                                                    + "/"
                                                    + freq
                                                    + "/"
                                                    + y
                                                    + "/"
                                                )

                                                # print('file dir', file_dir)
                                                try:
                                                    file_names = os.listdir(file_dir)
                                                    # print("file names", file_names)
                                                    if len(file_names) > 1:
                                                        print(
                                                            "WARNING: Multiple files exist where only one shouldb be."
                                                        )
                                                        print(file_dir)

                                                        file_name = file_names[0]
                                                        print(
                                                            "only considering first file:",
                                                            file_name,
                                                        )
                                                    else:
                                                        file_name = file_names[0]
                                                except IndexError:
                                                    print(
                                                        "WARNING: apparently no data file available. Skipping."
                                                    )
                                                    print(file_dir)
                                                    continue
                                                except TypeError:
                                                    print(
                                                        "WARNING: apparently no data file available. Skipping."
                                                    )
                                                    print(file_dir)
                                                    continue

                                                # chunksize
                                                chunksize = RES_TO_CHUNKSIZE[freq]

                                                data, corrupt = self.check_corruptness(
                                                    file_dir,
                                                    file_name,
                                                    chunksize,
                                                    var,
                                                    check_time=check_time,
                                                )
                                                if corrupt:
                                                    corrupt_files.append(
                                                        file_dir + file_name
                                                    )
                                                    continue
                                                write_dir = file_dir = (
                                                    self.store
                                                    + "/"
                                                    + project
                                                    + "/"
                                                    + experiment
                                                    + "/"
                                                    + var
                                                    + "/"
                                                    + nom_res
                                                    + "/"
                                                    + freq
                                                    + "/"
                                                    + y
                                                    + "/"
                                                )
                                                path = write_dir
                                                print(path)

                                                # check if path exist, create path if necessary

                                                isExist = os.path.exists(path)

                                                if not isExist:

                                                    # Create a new directory because it does not exist
                                                    os.makedirs(path)
                                                    print(
                                                        "The new directory is created!",
                                                        path,
                                                    )
                                                # reformat it from netcdf to h5
                                                # we can open it with xarray, transform it and then save it to .h5 with another engine
                                                # TODO: we might want to store directly as .h5 in the downloader?
                                                outfile = path + file_name.replace(
                                                    ".nc", ".h5"
                                                )
                                                print(outfile)
                                                if (not overwrite) and os.path.isfile(
                                                    outfile
                                                ):
                                                    print(
                                                        f"File {outfile} already exists, skipping."
                                                    )
                                                    continue

                                                # preprocessing stuff
                                                # do stuff
                                                # 1. deal with sectors
                                                # store it with same hierachy structure in a preprocessed foleder (self.store)
                                                print("preprocessing")
                                                ds = data.load()

                                                # TODO cmip preprocessing steps

                                                write_dir = file_dir = (
                                                    self.store
                                                    + "/"
                                                    + project
                                                    + "/"
                                                    + experiment
                                                    + "/"
                                                    + var
                                                    + "/"
                                                    + nom_res
                                                    + "/"
                                                    + freq
                                                    + "/"
                                                    + y
                                                    + "/"
                                                )
                                                path = write_dir
                                                print(path)

                                                # check if path exist, create path if necessary

                                                isExist = os.path.exists(path)

                                                if not isExist:

                                                    # Create a new directory because it does not exist
                                                    os.makedirs(path)
                                                    print(
                                                        "The new directory is created!",
                                                        path,
                                                    )
                                                # reformat it from netcdf to h5
                                                # we can open it with xarray, transform it and then save it to .h5 with another engine
                                                # TODO: we might want to store directly as .h5 in the downloader?
                                                outfile = path + file_name.replace(
                                                    ".nc", ".h5"
                                                )
                                                print(outfile)
                                                ds.to_netcdf(outfile, engine="h5netcdf")

            self.processed_flag = True
            print("finished", i)
        else:
            print("Skipping raw processing since it was already done!")
        return corrupt_files

    def sum_over_sectors(self, ds):
        print(ds)
        ds = ds.sum("sector")
        return ds

    def check_corruptness(self, file_dir, file_name, chunksize, var, check_time=True):

        try:
            data = xr.open_dataset(file_dir + file_name, chunks=chunksize)

        except ValueError:
            print(
                f"WARNING: Apparently the following file is corrupt: {file_dir+file_name}"
            )

            print("Skipping.")

            return None, True

        # check up: is variable existent in dataset?
        if var not in data.data_vars:
            print(
                f"WARNING: hierachy data variable {var} not found in actual file: \n {file_dir+file_name}."
            )
            print(f"The file contains following variables: {data.data_vars.keys()}")
            print(
                f"Skipping."
            )  # we may want to delete it? and remove it from tracking list?
            return None, True

        if check_time:
            try:
                data.time.dt.dayofyear
                # print(data.time.dt.month)
            except TypeError:
                print(
                    "WARNING: they might be a corruption in the time dimension of the dataset"
                )
                return None, True

        # TODO: include other checkups (does year, grid label and nom_res fit file _name?)
        # print("PASSED")

        return data, False

    def check(
        self,
        forcing_default_unit="kg m^-2 s^-1",
        cmip_force_consistency=True,
        check_time=True,
    ):
        """
        Checks all data for consistencey.
        Checs the follung aspects:

            - Check if units per variable are consistent: Sets all units of all files to the given default for forcing variables. For all other variables, just consistency within the files per variables are checked.
            - Corruptness.


        @params:
            forcing_default_unit [str]: Default unit all forcing data shoudl be converted to given that they currently are present in another unit. The string must be known to the unit registry by the print package: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
            cmip_force_consistency [bool]: If true, all files of one variable that do not match the unit of the first file per variable found, will be converted and overitten. If false, simply a warning message will appear.

            check_time [bool]: If true, check if time-dimension is intact, throw a warning

        @return:
            corrupt_files: [List(str)]: list of file_paths to files that seem to be corrupt
        """

        # storing potentially corrupt files
        corrupt_files = []

        # check if default forcing unit is present in print library
        try:
            ureg(forcing_default_unit)
        except pint.errors.UndefinedUnitError:
            print(
                f"WARNING: set forcing default unit not available. Make sure the unit string is known to pints unit registry."
            )
            print(
                "Please check: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt"
            )
            return

        # iterate over each var in the folder, checks if units per var are consistent
        for project in os.listdir(self.source):

            # part 1:
            if project == "input4mips":
                print("input4mips")

                for experiment in os.listdir(self.source + "/" + project):
                    for var in os.listdir(
                        self.source + "/" + project + "/" + experiment
                    ):

                        # for each var the resolution should be consistent

                        for i, nom_res in enumerate(
                            os.listdir(
                                self.source
                                + "/"
                                + project
                                + "/"
                                + experiment
                                + "/"
                                + var
                            )
                        ):

                            for freq in os.listdir(
                                self.source
                                + "/"
                                + project
                                + "/"
                                + experiment
                                + "/"
                                + var
                                + "/"
                                + nom_res
                            ):

                                for y in os.listdir(
                                    self.source
                                    + "/"
                                    + project
                                    + "/"
                                    + experiment
                                    + "/"
                                    + var
                                    + "/"
                                    + nom_res
                                    + "/"
                                    + freq
                                ):

                                    file_dir = (
                                        self.source
                                        + "/"
                                        + project
                                        + "/"
                                        + experiment
                                        + "/"
                                        + var
                                        + "/"
                                        + nom_res
                                        + "/"
                                        + freq
                                        + "/"
                                        + y
                                        + "/"
                                    )
                                    # print('file dir', file_dir)
                                    try:
                                        file_names = os.listdir(file_dir)
                                        # print("file names", file_names)
                                        if len(file_names) > 1:
                                            print(
                                                "WARNING: Multiple files exist where only one shouldb be."
                                            )
                                            print(file_dir)
                                            print(
                                                "only considering first file:",
                                                file_name,
                                            )
                                        file_name = file_names[0]
                                    except IndexError:
                                        print(
                                            "WARNING: apparently no data file available. Skipping."
                                        )
                                        print(file_dir)
                                        corrupt_files.append(file_dir + file_name)
                                        continue
                                    except TypeError:
                                        print(
                                            "WARNING: apparently no data file available. Skipping."
                                        )
                                        print(file_dir)
                                        corrupt_files.append(file_dir + file_name)
                                        continue

                                    # chunksize
                                    chunksize = RES_TO_CHUNKSIZE[freq]

                                    data, corrupt = self.check_corruptness(
                                        file_dir,
                                        file_name,
                                        chunksize,
                                        var,
                                        check_time=check_time,
                                    )
                                    if corrupt:
                                        corrupt_files.append(file_dir + file_name)
                                        continue

                                    # CHECK UNIT
                                    unit = data[var].units.replace("-", "^-")
                                    # print(unit)

                                    # if first unit extraction, check if there is a default unit given
                                    if (forcing_default_unit is None) & (i == 0):
                                        # extract unit from des
                                        forcing_default_unit = unit

                                    # if found unit equals default unit continue
                                    if forcing_default_unit == unit:
                                        continue
                                    else:
                                        # only load data to update when units mismatch as this takes some time
                                        ds = data.load()
                                        data.close()
                                        print("WARNING: mismatching units found.")
                                        print(
                                            f"Changing units from found unit {unit} to new default {forcing_default_unit}"
                                        )

                                        multiplyier = (
                                            ureg(unit)
                                            .to(forcing_default_unit)
                                            .magnitude
                                        )

                                        with xr.set_options(keep_attrs=True):
                                            ds.update({var: ds[var] * multiplyier})
                                            # print("multipyier", multiplyier)

                                            ds[var].attrs[
                                                "units"
                                            ] = forcing_default_unit
                                            #
                                            outfile = file_dir + "/" + file_name
                                            print("Overwriting file: ", outfile)
                                            ds.to_netcdf(outfile)

            # part 2: cmip data, units different from var to var, should just be consistent within var
            else:
                print("cmip6")
                continue
                for model in os.listdir(self.source + "/" + project):

                    print("model", model)

                    for ensemble_member in os.listdir(
                        self.source + "/" + project + "/" + model
                    ):

                        for experiment in os.listdir(
                            self.source
                            + "/"
                            + project
                            + "/"
                            + model
                            + "/"
                            + ensemble_member
                        ):
                            print("experiment", experiment)
                            for var in os.listdir(
                                self.source
                                + "/"
                                + project
                                + "/"
                                + model
                                + "/"
                                + ensemble_member
                                + "/"
                                + experiment
                            ):
                                print("var", var, "resetting default")
                                var_default_unit = ""

                                # for each var the resolution should be consistent

                                for nom_res in os.listdir(
                                    self.source
                                    + "/"
                                    + project
                                    + "/"
                                    + model
                                    + "/"
                                    + ensemble_member
                                    + "/"
                                    + experiment
                                    + "/"
                                    + var
                                ):
                                    # print(f"i {i} nom {nom_res}")

                                    for freq in os.listdir(
                                        self.source
                                        + "/"
                                        + project
                                        + "/"
                                        + model
                                        + "/"
                                        + ensemble_member
                                        + "/"
                                        + experiment
                                        + "/"
                                        + var
                                        + "/"
                                        + nom_res
                                    ):
                                        # print(f"freq {freq}")

                                        for y in os.listdir(
                                            self.source
                                            + "/"
                                            + project
                                            + "/"
                                            + model
                                            + "/"
                                            + ensemble_member
                                            + "/"
                                            + experiment
                                            + "/"
                                            + var
                                            + "/"
                                            + nom_res
                                            + "/"
                                            + freq
                                        ):
                                            # print(f"year {y}")

                                            file_dir = (
                                                self.source
                                                + "/"
                                                + project
                                                + "/"
                                                + model
                                                + "/"
                                                + ensemble_member
                                                + "/"
                                                + experiment
                                                + "/"
                                                + var
                                                + "/"
                                                + nom_res
                                                + "/"
                                                + freq
                                                + "/"
                                                + y
                                                + "/"
                                            )

                                            # print('file dir', file_dir)
                                            try:
                                                file_names = os.listdir(file_dir)
                                                # print("file names", file_names)
                                                if len(file_names) > 1:
                                                    print(
                                                        "WARNING: Multiple files exist where only one shouldb be."
                                                    )
                                                    print(file_dir)

                                                    file_name = file_names[0]
                                                    print(
                                                        "only considering first file:",
                                                        file_name,
                                                    )
                                                else:
                                                    file_name = file_names[0]
                                            except IndexError:
                                                print(
                                                    "WARNING: apparently no data file available. Skipping."
                                                )
                                                print(file_dir)
                                                corrupt_files.append(
                                                    file_dir + file_name
                                                )
                                                continue
                                            except TypeError:
                                                print(
                                                    "WARNING: apparently no data file available. Skipping."
                                                )
                                                print(file_dir)
                                                corrupt_files.append(
                                                    file_dir + file_name
                                                )
                                                continue

                                            # chunksize
                                            chunksize = RES_TO_CHUNKSIZE[freq]

                                            data, corrupt = self.check_corruptness(
                                                file_dir,
                                                file_name,
                                                chunksize,
                                                var,
                                                check_time=check_time,
                                            )
                                            if corrupt:
                                                corrupt_files.append(
                                                    file_dir + file_name
                                                )
                                                continue

                                            unit = data[var].units.replace("-", "^-")
                                            # print(unit)

                                            # if first unit extraction, check if there is a default unit given
                                            if var_default_unit == "":

                                                print(
                                                    f"Found unit: {unit} for var: {var}. Attempt to synchronize."
                                                )
                                                var_default_unit = unit

                                            # if found unit equals default unit continue
                                            elif var_default_unit == unit:
                                                # print("all good")
                                                continue
                                            else:
                                                if cmip_force_consistency:
                                                    # only load data to update when units mismatch as this takes some time
                                                    ds = data.load()
                                                    data.close()
                                                    print(
                                                        "WARNING: mismatching units found."
                                                    )
                                                    print(
                                                        f"Changing units from found unit {unit} to new default {var_default_unit}"
                                                    )

                                                    multiplyier = (
                                                        ureg(unit)
                                                        .to(forcing_default_unit)
                                                        .magnitude
                                                    )

                                                    with xr.set_options(
                                                        keep_attrs=True
                                                    ):
                                                        ds.update(
                                                            {var: ds[var] * multiplyier}
                                                        )
                                                        # print("multipyier", multiplyier)

                                                        ds[var].attrs[
                                                            "units"
                                                        ] = forcing_default_unit

                                                        outfile = file_dir + file_name
                                                        print(
                                                            "Overwriting file: ",
                                                            outfile,
                                                        )
                                                        ds.to_netcdf(outfile)
                                                else:
                                                    print(
                                                        f"WARNING: mismatching units found for {var} in {file_dir+file_name}. Pleace check. If you want to force synchronizing all units run the processer again and pass 'cmip_force_consistency=True"
                                                    )
                                                    print(
                                                        f"Found {unit}, previousliy found {var_default_unit}."
                                                    )
                                                    corrupt_files.append(
                                                        file_dir + file_name
                                                    )
                                                    continue
        return corrupt_files


if __name__ == "__main__":
    # for testing purposes

    # TODO use real paths here
    source = RAW_DATA
    store = PROCESSED_DATA
    raw_processer = RawProcesser(source, store)
    # raw_processer.process()
    print("Finished raw processing!")
