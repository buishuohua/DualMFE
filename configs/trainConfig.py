from dataclasses import dataclass, field
import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Tuple


@dataclass
class OptimizerConfig:
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: dict):
        instance = cls()
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(cls, key, value)
        return instance


@dataclass
class SchedulerConfig:
    mode: str = "min"
    factor: float = 0.5
    patience: int = 100
    eps: float = 1e-8
    cooldown: int = 10
    min_lr: float = 1e-8

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: dict):
        instance = cls()
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(cls, key, value)
        return instance


@dataclass
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 1000
    save_freq: int = 50
    learning_rate: float = 5e-5
    optimizer_name: str = "AdamW"
    scheduler_name: str = "ReduceLROnPlateau"
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    use_early_stop: bool = True
    early_stop_patience: Optional[int] = None

    criterion: str = "MSE"
    continue_learning: bool = False
    continue_expr: str = None

    grad_clip: bool = True
    grad_norm: Optional[float] = None

    def __post_init__(self):
        if self.grad_clip:
            self.grad_norm = 1.0

        if self.use_early_stop:
            self.early_stop_patience = 100

    def get_optimizer(self, model: nn.Module):
        optimizer_params = self.optimizer_config.to_dict()
        optimizer_params["lr"] = self.learning_rate
        if self.optimizer_name == "Adam":
            return Adam(model.parameters(), **optimizer_params)
        elif self.optimizer_name == "AdamW":
            return AdamW(model.parameters(), **optimizer_params)
        else:
            raise NotImplementedError

    def get_shceduler(self, optimizer: Optimizer):
        scheduler_params = self.scheduler_config.to_dict()
        if self.scheduler_name == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, **scheduler_params)
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
        result = dict(self.__dict__)
        result['optimizer_config'] = self.optimizer_config.to_dict()
        result['scheduler_config'] = self.scheduler_config.to_dict()
        return result

    def to_json(self, path: str):
        file_name = os.path.join(path, "train_config.json")
        with open(file_name, "w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))

    @classmethod
    def from_dict(cls, config_dict: dict):
        instance = cls()
        if 'optimizer_config' in config_dict and isinstance(config_dict['optimizer_config'], dict):
            instance.optimizer_config = OptimizerConfig.from_dict(
                config_dict['optimizer_config'])
            del config_dict['optimizer_config']

        if 'scheduler_config' in config_dict and isinstance(config_dict['scheduler_config'], dict):
            instance.scheduler_config = SchedulerConfig.from_dict(
                config_dict['scheduler_config'])
            del config_dict['scheduler_config']

        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        instance.__post_init__()
        return instance

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
