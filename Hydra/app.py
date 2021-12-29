import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path="config", config_name="config.yaml")
def read_params(cfg : DictConfig) -> None:
    #some hydra quirks
    print(os.getcwd())
    print(hydra.utils.get_original_cwd())
    #read params
    print(f"Reading params from the config.yaml")
    print(f"batch size = {cfg.batch_size}")
    print(f"dynamically set epochs = {cfg.epochs}")

if __name__ == "__main__":
    read_params()