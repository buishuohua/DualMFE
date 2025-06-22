from abc import ABC, abstractmethod
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer(ABC):
    def __init__(self, model_config, train_config, data_config):
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config

    @abstractmethod
    def build_model(self):
        pass

    def train(self):
        pass
