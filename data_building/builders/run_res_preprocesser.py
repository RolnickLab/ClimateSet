import argparse

from pathlib import Path

from data_building.builders.preprocesser import SpatResPreprocesser, TempResPreprocesser

# args: which directory should be processed which way?
parser = argparse.ArgumentParser(description="Processing module")
parser.add_argument("-t", "--task", type=str, help="task of the processing")
parser.add_argument("-i", "--input_dir", type=str, help="input directory that should be processed")
parser.add_argument("-e", "--example_dir", type=str, help="example directory how the data should look like")

# running through directories
def run_res_preprocesser(task, input_dir, example_dir):
    """ Running preprocesser for different cases.
    """
    # figure out:
    # TODO which variable is interpolated / aggregated?
    # --> different procedure for input4mips compared to cmip6?

    example_ds = get_single_example(example_dir)

    if task == "spat_interpolate":
        # init the right preprocesser
        res_processer = SpatResPreprocesser(example_ds, task="interpolate")

    elif task == "spat_aggregate":
        res_processer = SpatResPreprocesser(example_ds, task="aggregate")

    elif task == "temp_interpolate":
        res_processer = TempResPreprocesser(example_ds, task="interpolate")

    elif task == "temp_aggregate":
        res_processer = TempResPreprocesser(example_ds, task="aggregate")

    # run through directory
        # get a whole sub-directory
        # make a copy in a temporary directory?? (or apply directly)
        # apply processer to that (CDO can handle large data chunks!)
        ds = res_processer.apply(directory)
            # TODO in processer: rename params in the dataset because of new resolution
            # TODO add to processed param what has been done (task + resolutions)
        # store dataset (somewhere else in preprocessed under different resolution)


def get_single_example(dir):
    """ Gets an example file of a directory
    Parameters:
        dir (Path):
    Return:
        xarray: the file that can be used as example
    """
    pass

if __name__ == "__main__":
    # run preprocesser with args
