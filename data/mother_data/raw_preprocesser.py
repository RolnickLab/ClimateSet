from email.policy import default
from pathlib import Path
from data_paths import RAW_DATA, PROCESSED_DATA, LOAD_DATA
import os
import netCDF4 as nc
import numpy as np
import xarray as xr
import cftime
from utils.constants import RES_TO_CHUNKSIZE
import pint

overwrite = True
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
        """Init method for the RawProcesser
        Parameters:
            source (Path): Which directory should be processed. Freshly downloaded data was stored there.
            store (Path): Where to store the raw-processed files
        """
        self.source = source
        self.store = store
        # TODO integrate file-internal check if data was already raw_processed
        self.processed_flag = False
        self.check_units()
        # self.check_processed()

    def check_processed(self):
        """Checks if the data was already processed to prevent unnecessary processing.
        Operates on self.source and stored outcome in self.processed_flag
        """
        # TODO do checks (e.g. data already exists in PROCESSED, so we don't need to process it again)
        # set processed_flag to right boolean
        raise NotImplementedError

    def process(self):
        """Makes all the first and prior processing steps."""
        if not self.processed_flag:
            pass
            # TODO
            # take self.source data
            # do stuff (feel free to make new funcs)
            # store resulting data in self.store
            # self.processed_flag = True
        else:
            print("Skipping raw processing since it was already done!")
        raise NotImplementedError

    def check_units(
        self, forcing_default_unit="Kg m^-2 s^-1", cmip_force_consistency=True
    ):
        """
        Check if units per variable are consistent.
        Sets all units of all files to the given default for forcing variables.
        For all other variables, just consistency within the files per variables are checked.

        @params:
            forcing_default_unit [str]: Default unit all forcing data shoudl be converted to given that they currently are present in another unit. The string must be known to the unit registry by the print package: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
            cmip_force_consistency [bool]: If true, all files of one variable that do not match the unit of the first file per variable found, will be converted and overitten. If false, simply a warning message will appear.
        """

        # todo: check if default forcing unit is present in print
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
        # iterates over each var in the folder, checks if units per var are consistent

        for project in os.listdir(self.source):

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
                                            "WARNING: apparently no data file available. Skipping"
                                        )
                                        print(file_dir)
                                        continue
                                    except TypeError:
                                        print(
                                            "WARNING: apparently no data file available. Skipping"
                                        )
                                        print(file_dir)
                                        continue

                                    # chunksize
                                    chunksize = RES_TO_CHUNKSIZE[freq]

                                    try:
                                        data = xr.open_dataset(
                                            file_dir + file_name, chunks=chunksize
                                        )

                                    except ValueError:
                                        print(
                                            f"WARNING: Apparently the following file is corrupt: {file_dir+file_name}"
                                        )
                                        print("Skipping")
                                        continue

                                    # check up: is variable existent in dataset?
                                    if var not in data.data_vars:
                                        print(
                                            f"WARNING: hierachy data variable {var} not found in actual file: \n {file_dir+file_name}."
                                        )
                                        print(
                                            f"The file contains following variables: {ds.data_vars.keys()}"
                                        )
                                        print(
                                            f"Skipping"
                                        )  # we may want to delete it? and remove it from tracking list?
                                        continue

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
                                                    "WARNING: apparently no data file available. Skipping"
                                                )
                                                print(file_dir)
                                                continue
                                            except TypeError:
                                                print(
                                                    "WARNING: apparently no data file available. Skipping"
                                                )
                                                print(file_dir)
                                                continue

                                            # chunksize
                                            chunksize = RES_TO_CHUNKSIZE[freq]

                                            try:
                                                data = xr.open_dataset(
                                                    file_dir + file_name,
                                                    chunks=chunksize,
                                                )

                                            except ValueError:
                                                print(
                                                    f"WARNING: Apparently the following file is corrupt: {file_dir+file_name}"
                                                )
                                                print("Skipping")
                                                continue

                                            # check up: is variable existent in dataset?
                                            if var not in data.data_vars:
                                                print(
                                                    f"WARNING: hierachy data variable {var} not found in actual file: \n {file_dir+file_name}."
                                                )
                                                print(
                                                    f"The file contains following variables: {ds.data_vars.keys()}"
                                                )
                                                print(
                                                    f"Skipping"
                                                )  # we may want to delete it? and remove it from tracking list?
                                                continue

                                            unit = data[var].units.replace("-", "^-")
                                            # print(unit)

                                            # if first unit extraction, check if there is a default unit given
                                            if var_default_unit == "":
                                                print(
                                                    y,
                                                    var,
                                                    nom_res,
                                                    experiment,
                                                    ensemble_member,
                                                )
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
                                                        #
                                                        outfile = (
                                                            file_dir + "/" + file_name
                                                        )
                                                        print(
                                                            "Overwriting file: ",
                                                            outfile,
                                                        )
                                                        ds.to_netcdf(outfile)
                                        else:
                                            print(
                                                f"WARNING: mismatching units found for {var} in {file_dir+'/'+file_name}. Pleace check. If you want to force synchronizing all units run the processer again and pass 'cmip_force_consistency=True"
                                            )
                                            continue


if __name__ == "__main__":
    # for testing purposes

    # TODO use real paths here
    source = RAW_DATA
    store = PROCESSED_DATA
    raw_processer = RawProcesser(source, store)
    # raw_processer.process()
    print("Finished raw processing!")
