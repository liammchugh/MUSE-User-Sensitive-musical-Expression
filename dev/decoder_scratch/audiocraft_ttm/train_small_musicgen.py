#!/usr/bin/env python
# train_small_musicgen.py

import os
import hydra
from omegaconf import DictConfig, OmegaConf

# 1) Import AudioCraft's main training entry point
from audiocraft.train import train as audiocraft_train

@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig):
    """
    Merges all .yaml configs under the `conf/` folder using Hydra.
    Then calls AudioCraft's training pipeline with the final config.
    """
    # Show merged config
    print("========== Final Training Config ==========")
    print(OmegaConf.to_yaml(cfg))

    # 2) Run AudioCraft's training function
    #    By default, AudioCraft expects certain structure in `cfg`.
    #    You may need to adapt the keys if they differ from defaults.
    audiocraft_train(cfg)

if __name__ == "__main__":
    main()
