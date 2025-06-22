from base_trainer import BaseTrainer
import torch
import torch.nn as nn
from models.AFF import demoGRU

class GRUTrainer(BaseTrainer):
    def __init__(self, model_config, train_config, data_config, path_config):
        super().__init__(model_config, train_config, data_config, path_config)
        self.model = self.build_model()

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = demoGRU(self.data_config)
