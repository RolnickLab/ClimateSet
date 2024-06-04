import hydra
from omegaconf import DictConfig, OmegaConf
import torch

@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    
    from emulator.train import run_model

    return run_model(config)


if __name__ == "__main__":
    print("cuda available: ",torch.cuda.is_available())
    main()
