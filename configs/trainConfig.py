from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

@dataclass
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 100
    save_freq: int = 10
    learning_rate: float = 1e-5
    optimizer: str = "Adam"
    scheduler: str = "ReduceLROnPlateau"
    mode: str = "min"
    use_early_stop: bool = False
    criterion: str = "MSE"
    continue_learning: bool = False
    continue_expr: str = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.use_early_stop:
            self.early_stop_patience = 10
        else:
            self.early_stop_patience = 1e8

    def get_optimizer(self, model: nn.Module, **kwargs):
        weight_decay = kwargs.get("weight_decay", 0.0)
        if self.optimizer == "Adam":
            return Adam(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.optimizer == "AdamW":
            return AdamW(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def get_shceduler(self, optimizer: Optimizer, **kwargs):
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 10)
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
        else:
            raise NotImplementedError
