# TODO argparse with parameter where output data should be saved and where input data is saved


# Class to generate pytorch dataset for different needs (emulator, causal, causal emulator)
class CMIP6_loader():
    """
    """

    # def init
        # --> set path to data
        # --> set variables: models, ensembles, etc., frequency, variables, grid type, etc.

    # def download (only one time needed, use wget script)
    # [best case later: init says what kind of data we want to have and download creates the correct wget script for that]

    # def preprocess (put other functions in other python file for preprocessing, only for first time)

    # def load_data (just when download and preprocess already happened, returns an error if the data is now available yet in given dir)

    # def get_torch_dataset (specify if for emulator or causal model or causal emulator(?))
