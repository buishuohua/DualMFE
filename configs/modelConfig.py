from dataclasses import dataclass
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Union

@dataclass
class ModelConfig:
    modelname: str
    d_feature: int = 895
    seq_len: int = 120
    d_model: int = 256
    n_blocks: int = 8
    d_ff: int = 1024
    ln_eps: float = 1e-5
    dropout: float = 0.2
    activation: str = "gelu"

    # Transformer parameters
    n_heads: Optional[int] = None
    mask_flag: Optional[bool] = None
    attn_dropout: Optional[float] = None
    pe: Optional[str] = None

    def __post_init__(self):
        if self.modelname == "vTransformer":
            if self.n_heads is None:
                self.n_heads = 8
            if self.mask_flag is None:
                self.mask_flag = True
            if self.attn_dropout is None:
                self.attn_dropout = 0.2
            if self.pe is None:
                self.pe = "relative"

        elif self.modelname == "iTransformer":
            if self.n_heads is None:
                self.n_heads = 8
            if self.mask_flag is None:
                self.mask_flag = False
            if self.attn_dropout is None:
                self.attn_dropout = 0.2

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

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
