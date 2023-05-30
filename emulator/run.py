
import hydra
from omegaconf import DictConfig, OmegaConf

# TODO: function not found..
#from dotenv import load_dotenv
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
#load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    from emulator.train import run_model
    return run_model(config)


if __name__ == "__main__":
    main()