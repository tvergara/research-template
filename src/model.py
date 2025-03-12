from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig


class DefaultModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.layer = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.model.learning_rate)
        return optimizer
