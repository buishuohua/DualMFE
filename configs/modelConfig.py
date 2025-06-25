from dataclasses import dataclass
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Union

@dataclass
class ModelConfig():
    modelname: str
    d_input: int = 895  # TODO: automatically set
    d_model: int = 256
    n_blocks: int = 8
    d_ff: int = 1024
    ln_eps: float = 1e-5

    dropout: float = 0.2
    activation: str = "gelu"

    def to_dict(self):
        return self.__dict__

    def to_json(self, path: str):
        filename = os.path.join(path, "model_config.json")
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))

    def get_activation(self):
        if self.activation.lower() == "gelu":
            return nn.GELU()
        elif self.activation.lower() == "relu":
            return nn.ReLU()
        elif self.activation.lower() == "elu":
            return nn.ELU()
        else:
            raise NotImplementedError(
                f"Activation {self.activation} not implemented")

    @classmethod
    def from_dict(cls, config_dict: dict):
        instance = cls()
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @staticmethod
    def create_model_config(modelname: str, **kwargs):
        if modelname == "GRU":
            return GRUConfig(**kwargs)
        elif modelname == "vTransformer":
            return vTransformerConfig(**kwargs)
        else:
            raise NotImplementedError(f"Model {modelname} not implemented")


class GRUConfig(ModelConfig):
    def __init__(self, **kwargs):
        kwargs['modelname'] = "GRU"
        super().__init__(**kwargs)
        self.n_layers = kwargs.get('n_layers', 4)
        self.d_hidden = kwargs.get('d_hidden', 256)
        self.bidirection = kwargs.get('bidirection', False)


class vTransformerConfig(ModelConfig):
    # TODO: PE type
    def __init__(self, **kwargs):
        kwargs['modelname'] = "vTransformer"
        super().__init__(**kwargs)
        self.n_heads = kwargs.get('n_heads', 8)
        self.pe = kwargs.get('pe', "relative")
