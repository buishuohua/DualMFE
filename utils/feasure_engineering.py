import pandas as pd

def LOB_Imbalance(data_path: str, position: int = 0):
    data = pd.read_parquet(data_path)
    imbalance = (data["bid_qty"] - data["ask_qty"]) / (data["bid_qty"] + data["ask_qty"])
    data.insert(position, "LOB_Imbalance", imbalance)


def Trade_Imbalance(data_path: str, position: int = 0):
    data = pd.read_parquet(data_path)
    imbalance = (data["buy_qty"] - data["sell_qty"]) / (data["buy_qty"] + data["sell_qty"])
    data.insert(position, "TradeImbalance", imbalance)
