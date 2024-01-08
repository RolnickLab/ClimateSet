import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    from emulator.train import run_model

    return run_model(config)


if __name__ == "__main__":
    main()
