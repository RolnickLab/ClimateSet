from huggingface_hub import snapshot_download
import tarfile
from glob import glob
import os

# Path of the directory where the data will be downloaded in your local machine
local_directory = os.path.join(os.getcwd(), 'Climateset_DATA')

repo_id = "climateset/causalpaca"
repo_type = "dataset"

#snapshot_download(...) function will download the entire dataset from the HuggingFace Repository. This takes while.

print("Downloading the ClimateSet HuggingFace Repository...")
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_directory, local_dir_use_symlinks=False)
print("Done.")

print("Extracting the compressed input data...")
input_files = glob(local_directory + "/inputs/*.tar.gz")
for input_file in input_files:
    tar = tarfile.open(input_file)
    tar.extractall(path=local_directory + "/inputs/")
    tar.close()
    os.remove(input_file)
print("Done.")

print("Extracting the compressed output data...")
output_files = glob(local_directory + "/outputs/*.tar.gz")
for output_file in output_files:
    tar = tarfile.open(output_file)
    tar.extractall(path=local_directory + "/outputs/")
    tar.close()
    os.remove(output_file)
print("Done.")

print("Done. Finished downloading and extracting the Climateset data! :)")
