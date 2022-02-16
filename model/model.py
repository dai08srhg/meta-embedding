import torch
import torch.nn as nn
from typing import List, Tuple


class BaseModel(nn.Module):
    """
    ベースとなる予測モデル
    """

    def __init__(self, embedding_sizes: List[Tuple[int, int]]):
        super(BaseModel, self).__init__()

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(unique_size, embedding_dim) for unique_size, embedding_dim in embedding_sizes])

        input_dim = 0
        for _, embedding_dim in embedding_sizes:
            input_dim += embedding_dim

        self.predictor = nn.Sequential(nn.Linear(in_features=input_dim, out_features=64), nn.ReLU(),
                                       nn.Linear(in_features=64, out_features=1), nn.Sigmoid())

    def forward(self, inputs):
        embeddings = [embedding_layer(inputs[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
        h = torch.cat(embeddings, dim=1)  # concat
        p = self.predictor(h)
        return p


class MetaNetwork(nn.Module):
    """
    NN for train meta-embedding
    """

    def __init__(self, base_model: BaseModel, target_idxs: List[int], meta_idxs: List[int]):
        """
        Meta-Embedding network

        Args:
            base_model (BaseModel): 学習済みのベースモデル
            target_idxs (List[int]): Meta-Embedding対象の特徴量のインデックス
            meta_idxs (List[int]): Meta-Embeddingの入力に使う特徴量のインデックス
        """
        super(MetaNetwork, self).__init__()

        # 全特徴量の学習済み埋め込み層
        self.embedding_layers = base_model.embedding_layers

        # trainable network (Meta-Embedding対象特徴量ごとに定義)
        input_dim = len(meta_idxs)  # Meta-Embeddingに入力する特徴量の数
        self.meta_linear_layers = nn.ModuleList()
        for i in target_idxs:
            output_dim = base_model.embedding_layers[i].embedding_dim
            self.meta_linear_layers.append(nn.Linear(in_features=input_dim, out_features=output_dim))

        # predictor
        self.predictor = base_model.predictor

        # fix params
        for param in self.embedding_layers.parameters():
            param.requires_grad = False
        for param in self.predictor.parameters():
            param.requires_grad = False

        all_idx = list(range(len(base_model.embedding_layers)))
        self.target_idxs = target_idxs
        self.meta_idxs = meta_idxs
        self.other_idxs = [i for i in all_idx if i not in target_idxs]  # 通常のEmbeddingを行う特徴量のインデック

    def forward(self, inputs):
        # Meta-Embedding対象以外の特徴量を埋め込み
        embeddings = [self.embedding_layers[i](inputs[:, i]) for i in self.other_idxs]

        # Meta-embedding
        meta_feature_embeddings = [self.embedding_layers[i](inputs[:, i]) for i in self.meta_idxs]
        meta_vec = [torch.mean(embedding, dim=1).unsqueeze(1) for embedding in meta_feature_embeddings]  # pooling
        meta_vec = torch.cat(meta_vec, dim=1)
        meta_embeddings = [linear_layer(meta_vec) for linear_layer in self.meta_linear_layers]
        meta_embeddings = torch.cat(meta_embeddings, dim=1)

        embeddings.insert(0, meta_embeddings)
        h = torch.cat(embeddings, dim=1)  # concat

        p = self.predictor(h)
        return p


class CtrPredictor(nn.Module):
    """
    推論用のモデル
    """

    def __init__(self, meta_model: MetaNetwork, target_idxs, meta_idxs):
        """
        推論用モデル
        Args:
            base_model (BaseModel): 学習済みのベースモデル
            meta_model (MetaNetwork): 学習済みのMetaNetwork
            target_idxs (List[int]): Meta-Embedding対象の特徴量のインデックス
            meta_idxs (List[int]): Meta-Embeddingの入力に使う特徴量のインデックス
        """
        super(CtrPredictor, self).__init__()

        # 前特徴量のEmbedidng層
        self.embedding_layers = meta_model.embedding_layers
        # Meta-Embedding network
        self.meta_linear_layers = meta_model.meta_linear_layers
        # 識別層
        self.predictor = meta_model.predictor

        all_idx = list(range(len(meta_model.embedding_layers)))
        self.target_idxs = target_idxs
        self.meta_idxs = meta_idxs
        self.other_idxs = [i for i in all_idx if i not in target_idxs]  # 通常のEmbeddingを行う特徴量のインデックス

    def forward(self, inputs):
        # Meta-Embedding対象以外の特徴量の埋め込み
        embeddings = [self.embedding_layers[i](inputs[:, i]) for i in self.other_idxs]

        minibatch = inputs.size(0)
        # サンプルごとに処理
        target_embeddings = []
        for i in range(minibatch):
            # 特徴量ごとに処理
            embeddings_j = []
            for j in self.target_idxs:
                if inputs[i][j] != 0:
                    # 通常のEmbedding (既知のID)
                    embedding = self.embedding_layers[j](inputs[i][j])
                else:
                    # Meta-Embedding (未知のID)
                    meta_feature_embeddings = [self.embedding_layers[k](inputs[i][k]) for k in self.meta_idxs]
                    meta_vec = [torch.mean(embedding) for embedding in meta_feature_embeddings]
                    meta_vec = torch.stack(meta_vec)
                    embedding = self.meta_linear_layers[j](meta_vec)
                embeddings_j.append(embedding)
            embeddings_j = torch.cat(embeddings_j)
            target_embeddings.append(embeddings_j)
        target_embeddings = torch.stack(target_embeddings)

        embeddings.insert(0, target_embeddings)
        h = torch.cat(embeddings, dim=1)  # concat

        p = self.predictor(h)
        return p
