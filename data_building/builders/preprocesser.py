import os
import re
import json
import subprocess

import xarray as xr

from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod

from data_building.parameters.cdo_constants import SPAT_REMAPPING_ALGS, SPAT_REMAPPING_WEIGHTS, TEMP_INTERPOLATION_ALGS, TEMP_AGGREGATION_ALGS
from data_building.utils.helper_funcs import get_single_example, read_gridfile
from data_building.parameters.data_paths import ROOT

# TODO get rid off interpolate / aggregate difference if possible

class ResProcesser(ABC):
    """ Abstract class for resolution processing
    """
    def __init__(self, example: Path, task: str):
        """ Init
        Params:
            example (Path): Example how the dataset should look like
                resolution-wise
            task (str): Can be chosen between "interpolate" and "aggregate"
                or "remap". The last one includes interpolation and aggregation
                in the spatial domain.
        """
        self.example = example
        self.task = task
        self.input_dir = None
        self.output_dir = None
        self.finished_preprocessing = False
        if not self.task in ["interpolate", "aggregate", "remap", None]:
            raise ValueError("Resolution processer can only perform tasks 'interpolate' or 'aggregate' or 'remap'.")

        super().__init__()

    @abstractmethod
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

    def check_raw_processed(self, input_file) -> bool:
        """ check if an input file has been raw processed (flag!)
        Params:
            input_file (Path): the file that should be checked
        Returns:
            bool: True if it was raw processed, false if not
        """
        return xr.open_dataset(input_file).attrs["raw_processed"] == 'True'

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
        # Reset dir paths and finished_preprocessing
        self.reset_processer()
        self.input_dir = sub_dir
        self.output_dir = output_dir

    @abstractmethod
    def interpolate(self, alg: str, input: Path, output: Path):
        """ Abstract method for interpolating the current operating subdirectory.
        Params:
            alg (str): which method should be used for interpolation.
            in (Path): Can be a a directory of nc files or a single nc file
            out (Path): Can be a a directory of nc files or a single nc file
        """
        # TODO
        # if self.check_finished(input):
        #     raise RuntimeError("Preprocessing has already been done on this directory.")
        pass

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
    def __init__(self, example: Path, task: str = None, grid_file: Path = None):
        """ Init - adapted for spatial resolution processer.
        Params:
            example (Path): Example how the dataset should look like
                resolution-wise
            task (str): Can be chosen between "interpolate" and "aggregate". Can
                also be None.
        grid_file (Path): Instead of an example file, the user can also
            directly provide the target grid file (see cdo documentation).
            Set 'example = None' in that case (example will be ignored).
        """
        super().__init__(example, task)

        self.new_res = None # is set during grid file creation
        self.old_res = None

        # set grid file
        if self.example == None:
            if grid_file == None:
                raise ValueError("Grid file and example cannot be both None.")
            else:
                self.grid_file = grid_file
                grid_attrs = read_gridfile(self.grid_file)
                self.new_res = (int(float(grid_attrs["xinc"])*100), int(float(grid_attrs["yinc"])*100))
        else:
            self.grid_file = self.create_grid_from_example(example)

    def reset_processer(self):
        """ Reset the processer.
        """
        super().reset_processer()
        self.old_res = None
        # grid file, new_res and example stays fixed

    def create_grid_from_example(self, example: Path) -> Path:
        """ Creates a grid file that can be used by cdo from a given example file.
        The file is stored in a tmp under 'grid_files' and the path is returned.

        Params:
            example (Path): a file with the desired grid

        Returns:
            Path: path to the file where the grid is stored (for later reference)
        """
        # we are assuming lonlat gridtype here
        # we also assume that everything is called lon/lat after raw processing
        # we also assume that longitude is the x axis and latitude the y axis
        ds = xr.open_dataset(example)

        try:
            ds.coords["lon"]
            ds.coords["lat"]
            #ds.coords["test"]
        except KeyError:
            raise KeyError("""The given example file does not contain a longitude / latitude grid. Please consider creating your own grid file in that case. If your example should contain longitude / latitude values, make sure that the coordinates are named 'lon' and 'lat'.""")

        # fixed
        gridtype = "lonlat"

        # adapted
        xsize = len(ds.coords["lon"])
        ysize = len(ds.coords["lat"])
        xfirst = ds.lon[0].item()
        yfirst = ds.lat[0].item()
        xinc = abs(xfirst - ds.lon[1].item())
        yinc = abs(yfirst - ds.lat[1].item())

        # write out
        grid_filename = ROOT / "tmp" / "grid_files" / ("targetgrid_" + str(example.stem) + ".txt")
        with open(grid_filename, 'w') as f:
            f.write("gridtype = {}\n".format(gridtype))
            f.write("xsize    = {}\n".format(xsize))
            f.write("ysize    = {}\n".format(ysize))
            f.write("xfirst   = {}\n".format(xfirst))
            f.write("yfirst   = {}\n".format(yfirst))
            f.write("xinc     = {}\n".format(xinc))
            f.write("yinc     = {}\n".format(yinc))

        # set the new resolution in kilometers (just an approximation around the equator!!)
        # 1 degree circa 100km, i.e. 2.5 degree = 250km etc.
        self.new_res = (int(xinc * 100), int(yinc * 100))

        return grid_filename

    def create_weights(self, alg: str, input_file: Path, grid_file: Path = None):
        """ Creating weights for a given remapping function for a specific
        grid, given an example input file. The weights can then be applied
        to any file that has the same original grid as the example input file.

        Params:
            alg (str): Which algorithm should be used during remapping.
            input_file (Path): Example input file which contains the "original"
                grid.
            grid_file (Path): Path to custom grid file. If None the grid_file
                stored in the attributes is used.
        """
        # the weights are not stored as an attribute since they are changing
        # for each subdir. the grid file however stays the same!
        if grid_file is None:
            grid_file = self.grid_file

        # choose the right alg
        try:
            weights_alg = SPAT_REMAPPING_WEIGHTS[alg]
        except KeyError():
            raise KeyError("The given algorithm for remapping was not found. Check with cdo or cdo_constants.py.")

        weights_path = ROOT / "tmp" / "grid_files" / "weights" / ("weights_{}_from_{}.nc".format(grid_file.stem, input_file.stem))

        # cdo -genbil,grid_file input_file.nc weights_path
        subprocess.call([
            "cdo",
            "-s",
            "-w",
            "-{},{}".format(weights_alg, grid_file),
            input_file,
            weights_path
        ])
        return weights_path


    # TODO test
    def apply_subdir(self, sub_dir: Path, output_dir: Path, threads: int = 1):
        """ Used to apply spatial resolution processer on a directory given an
        example dataset file (provided during initialization).
        Please make sure that the subdirectory provided has the same climate
        variable across all files and the same original grid.
        Params:
            sub_dir (Path): dir of the data that should be remapped. This
                function assumes that ALL the files in this subdir have the
                same grid and can use a shared weight file for remapping!
            output_dir (Path): dir where the remapped data should be stored
            threads (int): how many OpenMP threads should be used for this.
                Default=1. For parallelizing, set this e.g. to 8.
        """
        print("Start resolution processing of {}.".format(sub_dir))
        # directories are copies and prepared in the parent method
        super().apply_subdir(sub_dir, output_dir)

        # read out variable that we are interpolating
        climate_var = super().read_var(sub_dir)
        # read out from jason file how we should interpolate that
        json_res_file = ROOT / "data_building" / "parameters" / "config_res_processing.json"
        alg = super().choose_alg("remap", climate_var, json_res_file)

        # create example weights for the sub directory
        first_example = get_single_example(sub_dir)
        weights_file = self.create_weights(alg=alg, input_file=first_example)
        self.old_res = xr.open_dataset(first_example).attrs["nominal_resolution"]

        # loop through sub_dir to remap all files in here
        for path, subdirs, files in tqdm(os.walk(sub_dir)):
            if len(files) > 0:
                for file in files:
                    # create output dir
                    full_output_path = self.create_output_path(output_dir, path, file)
                    self.remap(alg=alg, input=Path(path)/file, output=full_output_path, threads=threads, weights=weights_file)

        self.finished_preprocessing = True
        print("Finished the resolution preprocessing of {} and saved it at {}.".format(sub_dir, output_dir))

    # TODO test
    def create_output_path(self, output_dir: Path, path: str, file: str) -> Path:
        """ Creates an output path, the necessary parent directories and adapts it
        to the new resolution.
        Params:
            output_dir (Path): First part of the path for the output dir
            path (str): Can e.g. stem from os.walk - path of the current file.
            file (str): Can e.g. stem from os.walk - current file.
        Returns:
            Path: New path pointing where the output file can be stored.
        """
        topic_dir = file.split('_')[0]
        orig_out_path = Path(output_dir / topic_dir / Path(path.split(topic_dir+'/')[1]) / file)
        # replace 250_km with new resolution (match number_)
        adapted_out_path = Path(re.sub(r"\d*_km", "{}_km".format(self.new_res[0]), str(orig_out_path)))
        adapted_out_path.parent.mkdir(parents=True, exist_ok=True)
        return adapted_out_path

    # for later: interpolate between different height levels (see cdo)
    # TODO test
    def remap(self, alg: str, input: Path, output: Path, threads: int = 1, weights: Path = None):
        """ Remapping (i.e. interpolating and aggregating) data on the spatial
        level.
        Note: The remapping functions from cdo are here run with an "example file",
        but could also be run with external weights that can be custom-made.
        Params:
            alg (str): which method should be used for interpolation.
            input (Path): Single nc file
            output (Path): Single nc file
            threads (int): how many OpenMP threads should be used for this.
                Default=1. For parallelizing, set this e.g. to 8. Another cdo
                implementation is necessary for this (might work on a cluster,
                but not locally).
            weights (Path): Path to weights file for remapping. Useful when the
                same remapping is applied many times because this accelerates
                the process a lot (at least halfing runtime).
        """
        super().interpolate(alg, input, output)

        if not alg in SPAT_REMAPPING_ALGS:
            raise ValueError("""The requested spatial interpolation method does not exist.
                Must be one of the followings: {}""".format(SPAT_REMAPPING_ALGS))

        commands = [
                "cdo",
                "-s",
                "-w",
                "-P",
                str(threads),
                "", # here the different remapping funcs are filled in!
                "-setattribute,remap_alg={}".format(alg),
                "-setattribute,nominal_resolution='{} km''".format(self.new_res[0]),
                "-setattribute,orig_res='{} km'".format(self.old_res),
                input,
                output
        ]

        # remap either calculates the weights for the given input file
        if weights is None:
            # in the directory case this is set once
            self.old_res = xr.open_dataset(input).attrs["nominal_resolution"]
            # cdo -P threads -remapcon,targetgrid infile.nc outfile.nc
            commands[5] = "-{},{}".format(alg, self.grid_file)

        else: # or remap is applied with weights file (more efficient!)
            # cdo -P 1 -remap,mygrid,weights_nc infile.nc outfile.nc
            commands[5] = "-remap,{},{}".format(self.grid_file, weights)

        # call the actual commands
        subprocess.call(commands)

    def interpolate(self, alg: str, input: Path, output: Path):
        """ See docs of remap. This is only kept to fulfill abstract class
        requirements. Can still be directly applied to a single file.
        """
        remap(alg=alg, input=input, output=output)

    def aggregate(self, alg: str, input: Path, output: Path):
        """ See docs of remap. This is only kept to fulfill abstract class
        requirements. Can still be directly applied to a single file.
        """
        remap(alg=alg, input=input, output=output)


# inherits from ResProcesser()
class TempResProcesser(ResProcesser):
    """ Can be called to aggregate or interpolate files on the temporal axis.
    """
    # CONTINUE HERE
    # consider moving apply_subdir into the abstract class
    # here apply_subdir is just called + handover of a remapping function that
    # is defined within this class here

    # ATTENTION: apply_subdir is needed by every class - consider making it a util?
    # another idea: move apply_dir to an even higher class "Processer".
        # implement it in a way that it can be stacked, i.e. we can run
        # apply_subdir with a set of of different processers
    def apply_subdir(self, sub_dir: Path, output_dir: Path):
        """ Used to apply temporal resolution processer on a directory given an
        example dataset file.
        """
        super().apply_subdir(sub_dir, output_dir)
        raise NotImplementedError()


# TODO think about the right order!!

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
