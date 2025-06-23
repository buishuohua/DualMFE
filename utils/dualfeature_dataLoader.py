import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader

class DualFeature_Dataset(Dataset):
    def __init__(self, df: DataFrame, seq_len: int = 120, stride: int = 120):
        self.df = df
        self.seq_len = seq_len
        self.stride = stride
        self._split_features()

    def _split_features(self):
        if not hasattr(self, "labels"):
            self.labels = torch.tensor(self.df["label"].values, dtype = torch.float32)

        features_columns = self.df.columns.drop("label")
        anonymized_features = features_columns[features_columns.str.startswith("X_")]
        known_features = features_columns[~features_columns.str.startswith("X_")]

        if not hasattr(self, "a_features"):
            self.a_features = torch.tensor(self.df[anonymized_features].values, dtype = torch.float32)
        if not hasattr(self, "k_features"):
            self.k_features = torch.tensor(self.df[known_features].values, dtype = torch.float32)
        if not hasattr(self, "indices"):
            self.indices = torch.arange(0, len(self.labels) - self.seq_len + 1, self.stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len

        k_features = self.k_features[start_idx: end_idx]
        a_features = self.a_features[start_idx: end_idx]
        labels = self.labels[start_idx: end_idx]
        return k_features, a_features, labels

class DualFeature_DataLoader(DataLoader):
    # TODO: add num_workers as a hyperparameter, batchsize not used
    def __init__(self, dataset: DualFeature_Dataset, batch_size = 256, shuffle = True, drop_last = True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
