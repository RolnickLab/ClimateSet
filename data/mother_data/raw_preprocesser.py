from pathlib import Path
from datapaths import RAW_DATA, PROCESSED_DATA, LOAD_DATA

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
        """ Init method for the RawProcesser
        Parameters:
            source (Path): Which directory should be processed. Freshly downloaded data was stored there.
            store (Path): Where to store the raw-processed files
        """
        self.source = source
        self.store = store
        # TODO integrate file-internal check if data was already raw_processed
        self.processed_flag = False
        self.check_processed()

    def check_processed(self):
        """ Checks if the data was already processed to prevent unnecessary processing.
        Operates on self.source and stored outcome in self.processed_flag
        """
        # TODO do checks (e.g. data already exists in PROCESSED, so we don't need to process it again)
        # set processed_flag to right boolean
        raise NotImplementedError

    def process(self):
        """ Makes all the first and prior processing steps.
        """
        if not processed_flag:
            pass
            # TODO
            # take self.source data
            # do stuff (feel free to make new funcs)
            # store resulting data in self.store
            # self.processed_flag = True
        else:
            print("Skipping raw processing since it was already done!")
        raise NotImplementedError


if __name__ == '__main__':
    # for testing purposes

    # TODO use real paths here
    source = RAW_DATA
    store = PROCESSED_DATA
    raw_processer = RawProcesser(source, store)
    raw_processer.process()
    print("Finished raw processing!")
