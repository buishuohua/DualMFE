from dataclasses import dataclass
import json

@dataclass
class DataConfig:
    val_size : float = 0.2
    seq_len : int = 120
    stride : int = 120

    def to_json(self, path: str):
        pass
