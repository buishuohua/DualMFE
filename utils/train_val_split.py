from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
from typing import Tuple


def train_val_split(data_path: str, val_size: float = 0.2) -> Tuple[DataFrame, DataFrame]:
    data = pd.read_parquet(data_path)
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    unique_dates = pd.Series(data.index.date.unique())
    train_dates, val_dates = train_test_split(unique_dates, test_size=val_size)
    train = data[data.index.date.isin(train_dates)]
    val = data[data.index.date.isin(val_dates)]
    return train, val
