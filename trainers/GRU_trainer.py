from base_trainer import BaseTrainer
import torch
import torch.nn as nn


class GRUTrainer(BaseTrainer):
    def __init__(self, model_config, train_config, data_config):
        super().__init__(model_config, train_config, data_config)
        self.model = self.build_model()

    def build_model(self):
        model = demoGRU(self.model_config.d_input, self.model_config.d_hidden, self.model_config.n_layers, self.model_config.dropout)
        return model
