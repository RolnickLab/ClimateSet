# where the raw data is downloaded, stored and deleted after preprocessing
RAW_DATA # always deleted

# this data is stored and can be used for new user requests
PROCESSED_DATA # never deleted

# this data can be used within the data loader
LOAD_DATA # deleted if necessary, can be recreated from processed_data
