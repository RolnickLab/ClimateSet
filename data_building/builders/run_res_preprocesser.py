import argparse

from pathlib import Path

from data_building.builders.preprocesser import SpatResPreprocesser, TempResPreprocesser
from data_building.utils.helper_funcs import get_single_example

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

    # TODO write a script that decides which dirs have to be processed in which way
    # TODO write a config where you store which models have to be processed in which way??

    # run through directory
        # get a whole sub-directory
        # make a copy in a temporary directory?? (or apply directly)
        # apply processer to that (CDO can handle large data chunks!)
        res_processer.apply_dir(sub_directory, store_dir)
            # TODO in processer: rename params in the dataset because of new resolution
            # TODO add to processed param what has been done (task + resolutions) [flag!]
            # store dataset (somewhere else in preprocessed under different resolution)




if __name__ == "__main__":
    # run preprocesser with args
