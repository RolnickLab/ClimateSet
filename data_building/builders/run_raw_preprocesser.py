# the preprocessing here is only applied once in the beginning
import json

from data_building.parameters.data_paths import RAW_DATA, PROCESSED_DATA
from data_building.builders.preprocesser import EmissionProcesser, UnitsProcesser, XmipProcesser

def test_preprocessing(dir):
    """ Tests if data has already been processed.
    Params:
        dir (Path): directory that might or might not have been processed.
    Returns:
        bool: True if already processed, False if it has not been processed yet
    """
    # test if sectors have been summed ...
    # run tests for different properties that have been forced through processing

    raise NotImplementedError()
    # return boolean

def run_raw_preprocesser(raw_dir, processed_dir, input4mips=True, cmip6=True):
    """ Runs the preprocessing that has to be applied to all the data.
    """
    # TODO test if the json dicts are okay!

    if input4mips:
        # read out params from json files
        with open("None.json") as json_file:
            input4mips_params = json.load(json_file)
        # preprocess Input4MIPs data
        raw_process_input4mips(raw_dir / "input4mips", processed_dir / "input4mips", **input4mips_params)

    if cmip6:
        with open("None.json") as json_file:
            cmip6_params = json.load(json_file)
        # preprocess CMIP6 data
        raw_process_cmip6(raw_dir / "CMIP6", processed_dir / "CMIP6", **cmip6_params)


# TODO add all the params
def raw_process_input4mips(input_dir, output_dir, ARGS):
    """ Apply the actual processing to the input4mips data.
    """
    # create emission processer
    # TODO import dict from config_raw_processing.json
    # --> params for all the processing happening here

    emission_processer = EmissionProcesser()

    # run through directory (input_dir)

        # for each file

            # load dataset
            # make a copy of the dataset (internally)

            ds = emission_processer.sum_sectors(ds)

            ds = emission_processer.aggregate_emissions(ds)

            # ds = emission_processer. ...(ds)

            # add to preprocessing param list what has changed

            # store dataset (under preprocessed - output_dir)


# TODO add all the params
def raw_process_cmip6(input_dir, output_dir, ARGS):
    """ Raw processing for cmip6 data
    """
    xmip_processer = XmipProcesser()
    # TODO init list of units?
    unit_processer = UnitProcesser()

    # run through directory

    # for each file
        # load dataset ds from file

        ds = xmip_preprocesser.rename(ds)
            #rename_cmip6
        ds = xmip_preprocesser.reshape(ds)
            # promote_empty_dims, broadcast_lonlat, correct_lon, correct_coordinates, (do not add, because of performance issue: replace_x_y_nominal_lat_lon)
        ds = xmip_preprocesser.units(ds)
            # correct_units
            # TODO - this only adapt the depth units (I think!) - combine this with Charlie's code
        ds = xmip_preprocesser.add_bounds(ds)
            # parse_lon_lat_bounds
            # maybe_convert_bounds_to_vertex
            # maybe_convert_vertex_to_bounds
        ds = unit_processer.adapt_units(ds)

        # store dataset (under preprocessed - output_dir)
        # add some kind of flag that this dataset has been raw-processed

if __init__ == "__main__":
    # default preprocessing that is automatically applied
    already_preprocessed = test_preprocessing(PROCESSED_DATA)
    if not already_preprocessed:
        run_raw_preprocesser(RAW_DATA, PROCESSED_DATA, input4mips=True, cmip6=True)
    else:
        print("Data was not processed since it appears that it already has been raw-processed.")
