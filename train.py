import pandas as pd
from typing import List, Tuple
from category_encoders import OrdinalEncoder
import torch
from torch import nn, optim
from model.model import BaseModel, MetaNetwork, CtrPredictor
from model.dataset import MetaEmbeddingDataset
import tqdm
import hydra


def get_embedding_size(df: pd.DataFrame, embedding_dim: int) -> List[Tuple[int, int]]:
    """
    Get embedding size
    Args:
        df (pd.DataFrame): Train dataset
        embedding_dim (int): Number of embedded dimensions
    Returns:
        List[Tuple[int, int]]: List of (Unique number of categories, embedding_dim)
    """
    # Get embedding layer size
    max_idxs = list(df.max())
    embedding_sizes = []
    for i in max_idxs:
        embedding_sizes.append((int(i + 1), embedding_dim))

    return embedding_sizes


def train_base_model(X_train: pd.DataFrame, X_valid: pd.Series, y_train: pd.DataFrame, y_valid: pd.Series):
    """
    baseモデルを学習
    """
    train_dataset = MetaEmbeddingDataset(X_train, y_train)
    valid_dataset = MetaEmbeddingDataset(X_valid, y_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=3, shuffle=True)

    # Build model
    embedding_sizes = get_embedding_size(X_train, 5)
    base_model = BaseModel(embedding_sizes)

    # 設定
    epochs = 2
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(base_model.parameters(), lr=0.001, weight_decay=0.0001)

    # 学習開始
    min_valid_loss = 100.0
    for epoch in range(epochs):
        base_model.train()
        train_loss = 0.0
        valid_loss = 0.0
        for i, (inputs, click) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            click = torch.unsqueeze(click, 1)

            # Initialize gradient
            optimizer.zero_grad()
            # caluculate losses
            p_ctr = base_model(inputs)
            loss = loss_fn(p_ctr, click)
            train_loss += loss.item()
            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()

        # Validation Loop
        with torch.no_grad():
            base_model.eval()
            for inputs, click in valid_loader:
                click = torch.unsqueeze(click, 1)
                p_ctr = base_model(inputs)
                loss = loss_fn(p_ctr, click)
                valid_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        if valid_loss < min_valid_loss:
            print('パラメータ保存')
            min_valid_loss = valid_loss
            torch.save(base_model.state_dict(), '/workspace/storage/model/base_model_params.pth')


def train_meta_embedding(X_train: pd.DataFrame, X_valid: pd.Series, y_train: pd.DataFrame, y_valid: pd.Series,
                         target_cols: List[str], meta_cols: List[str]):
    """
    Meta-Embeddingの学習
    """
    # それぞれの特徴量のインデックス取得
    target_idxs = [X_train.columns.get_loc(col) for col in target_cols]
    meta_idxs = [X_train.columns.get_loc(col) for col in meta_cols]

    # データセット作成
    train_dataset = MetaEmbeddingDataset(X_train, y_train)
    valid_dataset = MetaEmbeddingDataset(X_valid, y_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=3, shuffle=True)

    # モデル構築
    embedding_sizes = get_embedding_size(X_train, 5)  # 全ての特徴量の埋め込み次元 = 5
    base_model = BaseModel(embedding_sizes)
    # ベースモデルのベストパラメータロード
    base_model.load_state_dict(torch.load('/workspace/storage/model/base_model_params.pth'))
    meta_model = MetaNetwork(base_model, target_idxs, meta_idxs)

    # 設定
    epochs = 2
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(base_model.parameters(), lr=0.001, weight_decay=0.0001)

    # 学習開始
    min_valid_loss = 100.0
    for epoch in range(epochs):
        base_model.train()
        train_loss = 0.0
        valid_loss = 0.0
        for i, (inputs, click) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            click = torch.unsqueeze(click, 1)

            # Initialize gradient
            optimizer.zero_grad()
            # caluculate losses
            p_ctr = meta_model(inputs)
            loss = loss_fn(p_ctr, click)
            train_loss += loss.item()
            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()

        # Validation Loop
        with torch.no_grad():
            base_model.eval()
            for inputs, click in valid_loader:
                click = torch.unsqueeze(click, 1)
                p_ctr = meta_model(inputs)
                loss = loss_fn(p_ctr, click)
                valid_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        if valid_loss < min_valid_loss:
            print('パラメータ保存')
            min_valid_loss = valid_loss
            torch.save(meta_model.state_dict(), '/workspace/storage/model/meta_model_params.pth')


def predict(X, X_train, target_cols, meta_cols):
    """
    テストデータ推論
    """
    # 推論モデル構築
    embedding_sizes = get_embedding_size(X_train, 5)
    target_idxs = [X_train.columns.get_loc(col) for col in target_cols]
    meta_idxs = [X_train.columns.get_loc(col) for col in meta_cols]
    # ベースモデル構築
    base_model = BaseModel(embedding_sizes)
    base_model.load_state_dict(torch.load('/workspace/storage/model/base_model_params.pth'))
    # metaモデル構築
    meta_model = MetaNetwork(base_model, target_idxs, meta_idxs)
    meta_model.load_state_dict(torch.load('/workspace/storage/model/meta_model_params.pth'))
    # 推論モデル構築
    ctr_predictor = CtrPredictor(meta_model, target_idxs, meta_idxs)
    print(X)

    print('Meta-Embeddingで推論')
    ctr_predictor.eval()
    with torch.no_grad():
        p = ctr_predictor(X)
    print(p)

    print('ベースモデルで推論')
    base_model.eval()
    with torch.no_grad():
        p = base_model(X)
    print(p)


@hydra.main(config_path='conf', config_name='conf')
def main(conf):
    df = pd.read_pickle('/workspace/storage/data/sample.pkl')

    # Encode categorical columns
    target_cols = list(conf.features.target_features)
    meta_cols = list(conf.features.meta_features)
    other_cols = list(conf.features.other_features)
    supervised_col = conf.features.supervised
    categorical_columns = target_cols + meta_cols + other_cols
    encoder = OrdinalEncoder(cols=categorical_columns, handle_unknown='impute').fit(df)
    df = encoder.transform(df)

    # train_test split
    X = df.drop(columns=supervised_col)
    y = df[supervised_col]

    train_base_model(X, X, y, y)
    train_meta_embedding(X, X, y, y, target_cols, meta_cols)

    # テストデータ予測
    df_test = pd.read_pickle('/workspace/storage/data/sample_test.pkl')
    df_test = encoder.transform(df_test)
    X_test = torch.from_numpy(df_test.drop(columns=supervised_col).values).long()
    predict(X_test, X, target_cols, meta_cols)


if __name__ == '__main__':
    main()
