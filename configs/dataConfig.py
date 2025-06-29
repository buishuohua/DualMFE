from dataclasses import dataclass
import json
import os

@dataclass
class DataConfig:
    val_size : float = 0.2
    seq_len : int = 120
    stride : int = 120
    start_timestep: str = "2023-03-01 00:00:00"
    feature_engineering: bool = False

    def to_dict(self):
        return self.__dict__

    def to_json(self, path: str):
        filename = os.path.join(path, "data_config.json")
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))

    @classmethod
    def from_dict(cls, config_dict: dict):
        instance = cls()
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    @classmethod
    def from_json(self, path: str):
        with open(path, "r") as f:
            config_dict = json.load(f)
        return self.from_dict(config_dict)
