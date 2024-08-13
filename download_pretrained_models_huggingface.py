from huggingface_hub import hf_hub_download


# Path of the directory where the checkpoints will be downloaded in your local machine
local_directory = os.path.join(os.getcwd(), 'pretrained_models')


# snapshot_download(...) function will download the folder from the HuggingFace Repository. This takes while.
# please refer to the huggingface repo to get instructions for downloading only a subselection of the available checkpoints
repo_id = "climateset/causalpaca_models"
repo_type = "model"
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_directory, local_dir_use_symlinks=False)


