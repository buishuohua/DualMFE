from dataclasses import dataclass
import os
import json
@dataclass
class ModelConfig():
    modelname: str = "GRU"
    d_input: int = 895
    n_layers: int = 2
    d_hidden: int = 256
    bidirection: bool = False
    dropout: float = 0.1

    def _post_init__(self):
        """_summary_
            set model specific parameters
        """
        pass

    def to_dict(self):
        return self.__dict__

    def to_json(self, path: str):
        filename = os.path.join(path, "model_config.json")
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))

    @classmethod
    def from_dict(cls, dict: dict):
        for key, value in dict.items():
            setattr(cls, key, value)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            dict = json.load(f)
        cls.from_dict(dict)
