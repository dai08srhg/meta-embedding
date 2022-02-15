import torch
import pandas as pd
from typing import List


class BaseTrainDataset(torch.utils.data.Dataset):
    """
    Dataset for train base model
    """

    def __init__(self, df: pd.DataFrame):
        df_feature = df.drop(columns='click')

        self.X = torch.from_numpy(df_feature.values).long()
        self.y = torch.from_numpy(df['click'].values).float()

        self.data_num = len(self.X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MetaTrainDataset(torch.utils.data.Dataset):
    """
    Dataset for train meta-embedding
    """

    def __init__(self, df: pd.DataFrame, ad_id: str, ad_features: List[str]):
        df_feature = df.drop(columns=['click', ad_id])
        df_ad_feature = df[ad_features]

        self.X = torch.from_numpy(df_feature.values).long()
        self.X_ad = torch.from_numpy(df_ad_feature.values).long()
        self.y = torch.from_numpy(df['click'].values).float()

        self.data_num = len(self.X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.X[idx], self.X_ad[idx], self.y[idx]
