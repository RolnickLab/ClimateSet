from pathlib import Path
from data_building.utils.helper_funcs import get_project_root
import os


# root of the project
ROOT_DIR = get_project_root()

# where the raw data is downloaded, stored and deleted after preprocessing
RAW_DATA = os.path.join(ROOT_DIR, "data/raw")  # always deleted

# this data is stored and can be used for new user requests
PROCESSED_DATA = os.path.join(ROOT_DIR, "data/processed")  # never deleted

# this data can be used within the data loader
LOAD_DATA = os.path.join(ROOT_DIR, "data/load")  # deleted if necessary

# directory to store meta information
META_DATA = os.path.join(ROOT_DIR, "data/meta")
