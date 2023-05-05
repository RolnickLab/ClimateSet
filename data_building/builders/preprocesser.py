import json

import xarray as xr

from pathlib import Path
from abc import ABC, abstractmethod

from data_building.parameters.cdo_constants import SPAT_INTERPOLATION_ALGS, TEMP_INTERPOLATION_ALGS, SPAT_AGGREGATION_ALGS, TEMP_AGGREGATION_ALGS
from data_building.utils.helper_funcs import get_single_example
from data_building.parameters.data_paths import ROOT

class ResProcesser(ABC):
    """ Abstract class for resolution processing
    """
    def __init__(self, example: Path, task: str):
        """ Init
        Params:
            example (Path): Example how the dataset should look like
                resolution-wise
            task (str): Can be chosen between "interpolate" and "aggregate"
        """
        self.example = example
        self.task = task
        self.input_dir = None
        self.output_dir = None
        self.finished_preprocessing = False
        if not self.task in ["interpolate", "aggregate"]:
            raise ValueError("Resolution processer can only perform tasks 'interpolate' or 'aggregate'.")
        super().__init__()

    def reset_processer(self):
        """ Resets the processer. Does not reset the example, only the directories.
        """
        self.input_dir = None
        self.output_dir = None
        self.finished_preprocessing = False

    def check_finished(self, input):
        """ check if resolution processing has been finished
        """
        # TODO catch in case input_dir / output_dir have not been init yet
        # TODO check if files in output_dir are the same number like in input dir
        # TODO check at least for one file that it is not empty

        raise NotImplementedError()

    def check_raw_processed(self, input_file):
        """ check if an input file has been raw processed (flag!)
        """
        # CDO command!
        raise NotImplementedError()

    def choose_alg(self, task: str, climate_var: str, json_path: Path) -> str:
        """ Returns which algorithm should be used to interpolate / aggregate
        the climate data variable provided.
        Params:
            task (str): Can be 'interpolation' or 'aggregation'.
            climate_var (str): Can be any variable if stored in the respective
                json file. Includes also ghg variables and others.
            json_path (Path): where the jason file is stored that contains
                the variable - aggregation/interpolation algorithm mapping
        Returns:
            str: name of the chosen algorithm
        """
        # read this out the json file
        with open(json_path) as file:
            res_dict = json.load(file)

        try:
            alg = res_dict[task][climate_var]
        except KeyError:
            raise KeyError("The config_res_processing.json does not contain the task {} or the variable {}".format(task, climate_var))

        if alg == "null":
            raise ValueError("Preferred algorithm for interpolation / aggregation has not been set yet.")

        return alg

    def read_var(self, sub_dir: Path, test: bool = False) -> str:
        """ Returns a string that symbolizes the climate variable or ghg variable
        present in the sub_dir.
        Params:
            sub_dir (Path): Sub directory that contains the data
            test (bool): If true, the file is opened and the climate variable
                is read out from the file. If false (default), the variable
                is read out from the file name.
        Returns:
            str: type of climate / ghg variable
        """
        # run through files, grap the first file, read out the climate variable
        path_file = get_single_example(sub_dir)

        # we rely on the file being stored with the right naming convention used here
        namings = str(path_file.name).split('_')
        dataset = namings[0]
        if dataset == "input4mips":
            # what names: BC_em_anthro, BC_em_biomassburning, BC_em_AIR-anthro
            var = ('_').join(namings[2:5])
        elif dataset == "CMIP6":
            var = namings[4]
        else:
            raise ValueError("The type of data passed is not known. File name should start with 'input4mips' or 'CMIP6'.")

        # in case we want to be 100% sure we are using the right variable
        # (drop this for efficency)
        if test:
            ds = xr.open_dataset(path_file)
            if not (var in list(ds.keys())):
                raise ValueError("Dataset does not contain the expected climate or ghg variable.")

        return var

    @abstractmethod
    def apply_subdir(self, sub_dir: Path, output_dir: Path):
        """ Abstract method used to apply resolution processers to a subdirectory.
        The subdirectory may not be nested!
        Params:
            sub_dir (Path): A non-nested directory that contains nc files that
                should be processed.
            output_dir (Path): Specifies where the processed data should be stored.
        """
        # 0. Reset operate_dir and finished_preprocessing
        self.reset_processer()
        raise NotImplementedError()

    @abstractmethod
    def interpolate(self, alg: str, input: Path, output: Path):
        """ Abstract method for interpolating the current operating subdirectory.
        Params:
            alg (str): which method should be used for interpolation.
            in (Path): Can be a a directory of nc files or a single nc file
            out (Path): Can be a a directory of nc files or a single nc file
        """
        if self.check_finished(input):
            raise RuntimeError("Preprocessing has already been done on this directory.")

    @abstractmethod
    def aggregate(self, alg: str, input: Path, output: Path):
        """ Abstract method to aggregate the current operating subdirectory.
        Params:
            alg (str): which method should be used for aggregation.
            in (Path): Can be a a directory of nc files or a single nc file
            out (Path): Can be a a directory of nc files or a single nc file
        """
        if self.check_finished(input):
            raise RuntimeError("Preprocessing has already been done on this directory.")

class SpatResProcesser(ResProcesser):
    """ Can be called to aggregate or interpolate files on the spatial axis.
    """
    # TODO test
    def apply_subdir(self, sub_dir: Path, output_dir: Path):
        """ Used to apply spatial resolution processer on a directory given an
        example dataset file. Please make sure that the subdirectory provided
        has the same climate variable across all files.
        """
        # directories are copies and prepared in the parent method
        super().apply_subdir(sub_dir, output_dir)

        # read out variable that we are interpolating
        climate_var = super().read_var(sub_dir)
        # read out from jason file how we should interpolate that
        json_res_file = ROOT / "data_building" / "parameters" / "config_res_processing.json"
        alg = super().choose_alg(task, climate_var, json_res_file)

        if self.task == "interpolate":
            self.interpolate(alg, input, output)

        elif self.task == "aggregate":
            self.aggregate(alg, input, output)

        self.finished_preprocessing = True
        print("Finished the resolution preprocessing of {} and saved it at {}.".format(sub_dir, output_dir))

    def interpolate(self, alg: str, input: Path, output: Path):
        """
        Note: The remapping functions from cdo are here run with an "example file",
        but could also be run with external weights that can be custom-made.
        Params:
            alg (str): which method should be used for interpolation.
            in (Path): Can be a a directory of nc files or a single nc file
            out (Path): Can be a a directory of nc files or a single nc file
        """
        super().interpolate(alg)

        if not alg in SPAT_INTERPOLATION_ALGS:
            raise ValueError("""The requested spatial interpolation method does not exist.
                Must be one of the followings: {}""".format(SPAT_INTERPOLATION_ALGS))
        # TODO
        # CONTINUE HERE
        # maybe move this in a separate bash script??
        # should I add this one here??: #!/bin/bash
        # first line: looping through all files in sub_dir
        bash_script = """
        find . -type f -print0 | while IFS= read -r -d $'\0' file;
            do echo "$file" ;
        done
        """

        os.system("bash -c %s" % bash_script)


        # BASH CODE STARTS HERE
        # ________________________
        # 1. bigger loop through files here
            # 2. check for each file with cdo: has it been raw processed? (FLAG!)
            # 3. if yes: CDO remapping command
            # 4. if no: output warning, write out all files that were skipped in temp
        # BASH CODE ENDS HERE

        # for the beginning: choose a file from the operating director

        # TODO run cdo command on this file
        # cdo -remapcon,other_data.nc infile outfile
        # remapycon -> the alg
        # other_data.nc -> the example file
        # infile -> selected file
        # outfile -> where it should be stored (new name + path since it has different specs?)


        raise NotImplementedError()

    def aggregate(self, alg: str, input: Path, output: Path):
        """
        Params:
            alg (str): which method should be used for aggregation.
            in (Path): Can be a a directory of nc files or a single nc file
            out (Path): Can be a a directory of nc files or a single nc file
        """
        super().aggregate(alg)
        if not alg in spat_aggregation_algs:
            raise ValueError("The requested spatial interpolation method does not exist.")
        # TODO if else cases for the different algorithms
        raise NotImplementedError()




# inherits from ResProcesser()
class TempResProcesser(ResProcesser):
    """ Can be called to aggregate or interpolate files on the temporal axis.
    """
    def apply_subdir(self, sub_dir: Path, output_dir: Path):
        """ Used to apply temporal resolution processer on a directory given an
        example dataset file.
        """
        super().apply_subdir(sub_dir, output_dir)
        raise NotImplementedError()


class EmissionProcesser():
    """ Can be called to summarize emission over sectors.
    """
    pass

class UnitsProcesser():
    """ Can be called to transform units.
    """
    # TODO add Charlie's code here if possible
    # don't forget the calendar inconsistencies
    pass

class XmipProcesser():
    """ Can be called to apply xmip preprocessing.
    """

# TODO find out where to put the co2 preprocessing
# def Cumsum co2 --> adapt input4mips
# def baseline --> substract, depending on experiment and cmip6 model
# TODO add important stuff in ghgs.json (which ghg are long living?)
class CO2Preprocesser():
    """ See: https://github.com/duncanwp/ClimateBench/blob/main/prep_input_data.ipynb
    """
