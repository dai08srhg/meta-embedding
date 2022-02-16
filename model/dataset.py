import torch
import pandas as pd


class MetaEmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset for train base model
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.from_numpy(X.values).long()
        self.y = torch.from_numpy(y.values).float()

        self.data_num = len(self.X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
