from abc import ABC, abstractmethod
import xarray as xr

from pathlib import Path

from data_building.parameters.cdo_constants import SPAT_INTERPOLATION_ALGS, TEMP_INTERPOLATION_ALGS, SPAT_AGGREGATION_ALGS, TEMP_AGGREGATION_ALGS


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
    # TODO consider moving this into the parent class
    def read_var(self, sub_dir: Path) -> str:
        """ Returns a string that symbolizes the climate variable present in the
        sub_dir.
        Params:
            sub_dir (Path): Sub directory that contains the data
        Returns:
            str: type of climate variable
        """
        # run through files, grap the first file, read out the climate variable
        # and return it
        # CONTINUE HERE
        raise NotImplementedError()

    # TODO consider moving thsi into the parent class
    def choose_alg(self, task: str, climate_var: str) -> str:
        """ Returns which algorithm should be used to spatially interpolate
        the climate data variable provided.
        """
        # read this out the jason file
        # CONTINUE HERE
        raise NotImplementedError()

    def apply_subdir(self, sub_dir: Path, output_dir: Path):
        """ Used to apply spatial resolution processer on a directory given an
        example dataset file. Please make sure that the subdirectory provided
        has the same climate variable across all files.
        """
        # directories are copies and prepared in the parent method
        super().apply_subdir(sub_dir, output_dir)

        # read out variable that we are interpolating
        climate_var = self.read_var(sub_dir)
        # read out from jason file how we should interpolate that
        alg = self.choose_alg(task, climate_var)

        if self.task == "interpolate":
            self.interpolate(alg, input, output)

        elif self.task == "aggregate":
            self.aggregate(alg, input, output)

        self.finished_preprocessing = True
        print("Finished the resolution preprocessing of {} and saved it at {}.".format(sub_dir, output_dir))

    def interpolate(self, alg: str, input: Path, output: Path):
        """
        Note: The remapping functions from cdo  are here run with an "example file",
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

        # BASH CODE STARTS HERE
        # ________________________
        # 1. bigger loop through files here
            # 2. check for each file with cdo: has it been raw processed? (FLAG!)
            # 3. if yes: CDO remapping command
            # 4. if no: output warning, write out all files that were skipped in temp
        # BASH CODE ENDS HERE



        # for the beginning: choose a file from the operating director

        # TODO run cdo command on this file
        # cdo -remapycon,other_data.nc infile outfile
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
