from dataclasses import dataclass

@dataclass
class ModelConfig():
    modelname: str = "GRU"
    layers: int = 2
    d_hidden: int = 256
    bidirection: bool = False

    def _post_init__(self):
        """_summary_
            set model specific parameters
        """
        pass

    @staticmethod
    def from_json(path: str):
        pass
