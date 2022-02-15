import pandas as pd
from typing import List, Tuple
import numpy as np
from category_encoders import OrdinalEncoder
import torch
from torch import nn, optim
from model.model import BaseModel, MetaNetwork, CtrPredictor
from model.dataset import BaseTrainDataset, MetaTrainDataset
import tqdm
import hydra


def load_dataset() -> pd.DataFrame:
    """
    Load dataset from local storage.
    Returns:
        pd.DataFrame: dataset
    """
    df = pd.read_pickle('/workspace/data/sample.pkl')
    return df


def get_embedding_size(df: pd.DataFrame, embedding_dim: int) -> List[Tuple[int, int]]:
    """
    Get embedding size
    Args:
        df (pd.DataFrame): Train dataset
        embedding_dim (int): Number of embedded dimensions
    Returns:
        List[Tuple[int, int]]: List of (Unique number of categories, embedding_dim)
    """
    # Extract feature columns
    df_feature = df.drop(columns='click')

    # Get embedding layer size
    max_idxs = list(df_feature.max())
    embedding_sizes = []
    for i in max_idxs:
        embedding_sizes.append((int(i + 1), embedding_dim))

    return embedding_sizes


def train(df: pd.DataFrame, ad_id: str, ad_features: List[str]):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Build dataset
    dataset = BaseTrainDataset(df)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)

    # Build model
    embedding_sizes = get_embedding_size(df, 5)
    base_model = BaseModel(embedding_sizes)
    print(base_model)

    epochs = 2
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(base_model.parameters(), lr=0.01, weight_decay=0.0001)

    # Start fitting
    base_model.train()
    for epoch in range(epochs):
        for i, (inputs, click) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = inputs.to(device)
            click = torch.unsqueeze(click.to(device), 1)

            # Initialize gradient
            optimizer.zero_grad()
            # caluculate losses
            p_ctr = base_model(inputs)
            loss = loss_fn(p_ctr, click)
            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()

    # Train Meta-Embedding
    meta_dataset = MetaTrainDataset(df, ad_id, ad_features)
    mata_dataloader = torch.utils.data.DataLoader(meta_dataset, batch_size=3, shuffle=True)

    # Build model
    ad_id_idx = df.columns.get_loc('ad_id')
    ad_feature_idxs = [df.columns.get_loc(column) for column in ad_features]
    meta_network = MetaNetwork(base_model, ad_id_idx, ad_feature_idxs, 5)

    optimizer = optim.SGD(meta_network.parameters(), lr=0.01, weight_decay=0.0001)

    for epoch in range(epochs):
        for i, (inputs, inputs_ad, click) in tqdm.tqdm(enumerate(mata_dataloader), total=len(mata_dataloader)):
            inputs = inputs.to(device)
            inputs_ad = inputs_ad.to(device)
            click = torch.unsqueeze(click.to(device), 1)

            # Initialize gradient
            optimizer.zero_grad()
            # caluculate losses
            p_ctr = meta_network(inputs_ad, inputs)
            loss = loss_fn(p_ctr, click)
            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()

    # Predict
    predictor = CtrPredictor(base_model, meta_network)
    x = np.array([[0, 2, 3, 4, 5, 6, 7], [0, 1, 3, 4, 1, 2, 1]])
    x_ad = np.array([[2, 3, 4], [1, 3, 4]])
    x = torch.from_numpy(x).long()
    x_ad = torch.from_numpy(x_ad).long()
    p = predictor(x, x_ad)
    print(p)


@hydra.main(config_path='conf', config_name='conf')
def main(conf):
    df = load_dataset()
    print(df)

    # Encode categorical columns
    ad_id = conf.features.ad_id
    ad_features = list(conf.features.ad_features)
    other_features = list(conf.features.other_features)
    categorical_columns = [ad_id] + ad_features + other_features
    encoder = OrdinalEncoder(cols=categorical_columns, handle_unknown='impute').fit(df)
    df = encoder.transform(df)
    print(df)

    train(df, ad_id, ad_features)


if __name__ == '__main__':
    main()
