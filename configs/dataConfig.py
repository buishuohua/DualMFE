from dataclasses import dataclass
import json
import os

@dataclass
class DataConfig:
    val_size : float = 0.2
    seq_len : int = 120
    stride : int = 120
    feature_engineering: bool = False

    def to_dict(self):
        return self.__dict__

    def to_json(self, path: str):
        filename = os.path.join(path, "data_config.json")
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))

    @classmethod
    def from_dict(cls, dict: dict):
        for key, value in dict.items():
            setattr(cls, key, value)

    @classmethod
    def from_json(self, path: str):
        with open(path, "r") as f:
            dict = json.load(f)
        self.from_dict(dict)
