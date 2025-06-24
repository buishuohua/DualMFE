from dataclasses import dataclass
import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from typing import Optional

@dataclass
class TrainConfig:
    # TODO: Split parameters
    batch_size: int = 256
    num_epochs: int = 1000
    save_freq: int = 50
    learning_rate: float = 5e-5
    optimizer: str = "AdamW"
    scheduler: str = "ReduceLROnPlateau"

    mode: str = "min"
    use_early_stop: bool = False
    # TODO: error
    early_stop_patience: Optional[int] = 1e8

    criterion: str = "MSE"
    continue_learning: bool = False
    continue_expr: str = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: add more regularization parameters
    grad_clip: bool = True
    grad_norm: Optional[float] = 1.0

    def _post_init__(self):
        if self.grad_clip:
            self.grad_norm = 1.0
        else:
            self.grad_norm = None

        if self.use_early_stop:
            self.early_stop_patience = 10
        else:
            self.early_stop_patience = 1e8

    def get_optimizer(self, model: nn.Module, **kwargs):
        weight_decay = kwargs.get("weight_decay", 1e-6)
        if self.optimizer == "Adam":
            return Adam(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.optimizer == "AdamW":
            return AdamW(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def get_shceduler(self, optimizer: Optimizer, **kwargs):
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 100)
        if self.scheduler == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, mode=self.mode, factor=factor, patience=patience)
        elif self.scheduler == "StepLR":
            #TODO: optimizae the paramters parsing
            return StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            raise NotImplementedError

    def get_loss_fn(self):
        if self.criterion == "MSE":
            return nn.MSELoss(reduction="mean")
        elif self.criterion == "MAE":
            return nn.L1Loss(reduction="mean")
        else:
            raise NotImplementedError

    def to_dict(self):
        return self.__dict__

    def to_json(self, path: str):
        file_name = os.path.join(path, "train_config.json")
        with open(file_name, "w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))

    @classmethod
    def from_dict(cls, dict: dict):
        for key, value in dict.items():
            setattr(cls, key, value)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            dict = json.load(f)
        cls.from_dict(dict)
