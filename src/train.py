import os

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src.model import DefaultModule


@hydra.main(version_base="1.2", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    model = DefaultModule(cfg)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
    )

    trainer = pl.Trainer(**cfg.trainer, logger=logger)

    print(model, trainer)


if __name__ == "__main__":
    main()
