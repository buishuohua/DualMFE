from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from models.AFF import vTransformer


class vTransformerTrainer(BaseTrainer):
    def __init__(self, model_config, train_config, data_config, path_config):
        super().__init__(model_config, train_config, data_config, path_config)

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = vTransformer(self.model_config.to_dict())
