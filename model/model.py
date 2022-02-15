import torch
import torch.nn as nn
from typing import List, Tuple


class BaseModel(nn.Module):

    def __init__(self, embedding_sizes: List[Tuple[int, int]]):
        super(BaseModel, self).__init__()

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(unique_size, embedding_dim) for unique_size, embedding_dim in embedding_sizes])

        input_dim = 0
        for _, embedding_dim in embedding_sizes:
            input_dim += embedding_dim

        self.linear = nn.Linear(in_features=input_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        embeddings = [embedding_layer(inputs[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
        h = torch.cat(embeddings, dim=1)  # concat
        h = self.linear(h)
        p = self.sigmoid(h)
        return p


class MetaNetwork(nn.Module):
    """
    NN for train meta-embedding
    """

    def __init__(self, base_model: BaseModel, ad_id_idx: int, ad_feature_idxs: List[int], output_dim: int):
        """
        Meta-Embedding network

        Args:
            base_model (BaseModel): trained base_model
            ad_feature_idxs: indexs of ad
        """
        super(MetaNetwork, self).__init__()

        # ad id以外の埋め込み層
        self.embedding_layers = nn.ModuleList()
        for idx in range(len(base_model.embedding_layers)):
            if idx != ad_id_idx:
                self.embedding_layers.append(base_model.embedding_layers[idx])

        # 広告特徴量の埋め込み層
        self.ad_feature_embedding_layers = nn.ModuleList()
        for i in ad_feature_idxs:
            self.ad_feature_embedding_layers.append(base_model.embedding_layers[i])

        # trainable network
        input_dim = len(self.ad_feature_embedding_layers)
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

        # predictor
        self.predictor = base_model.linear
        self.sigmoid = nn.Sigmoid()

        # fix params
        for param in self.embedding_layers.parameters():
            param.requires_grad = False
        for param in self.ad_feature_embedding_layers.parameters():
            param.requires_grad = False
        for param in self.predictor.parameters():
            param.requires_grad = False

    def forward(self, ad_feature_inputs, feature_inputs):
        # embedding
        embeddings = [embedding_layer(feature_inputs[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]

        # meta-embedding
        ad_feature_embeddings = [
            embedding_layer(ad_feature_inputs[:, i])
            for i, embedding_layer in enumerate(self.ad_feature_embedding_layers)
        ]
        h_ad = [torch.mean(embedding, dim=1).unsqueeze(1) for embedding in ad_feature_embeddings]  # average pooling
        h_ad = torch.cat(h_ad, dim=1)
        meta_embedding = self.linear(h_ad)

        # predict
        embeddings.insert(0, meta_embedding)
        h = torch.cat(embeddings, dim=1)
        p = self.predictor(h)
        p = self.sigmoid(p)

        return p


class CtrPredictor(nn.Module):

    def __init__(self, base_model: BaseModel, meta_network: MetaNetwork):
        super(CtrPredictor, self).__init__()

        self.embedding_layers = base_model.embedding_layers

        # Meta-Embedding network
        self.ad_feature_embedding_layers = meta_network.ad_feature_embedding_layers
        self.meta_embedding_generator = meta_network.linear

        # predict layer
        self.predictor = base_model.linear
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, ad_inputs):
        ad_id_inputs = inputs[:, 0]

        # ad_id以外の特徴量をembedding
        embedding_nums = len(self.embedding_layers)
        embeddings = [self.embedding_layers[i](inputs[:, i]) for i in range(1, embedding_nums)]

        # ad_idのembedding
        ad_id_embeddings = []
        for i in range(inputs.size(0)):
            input = ad_id_inputs[i]
            if input != 0:
                # 学習データ中に含まれる既知のad_idの場合
                embedding = self.embedding_layers[0](input)
                ad_id_embeddings.append(embedding)
            else:
                # 学習データに含まれない未知のad_idの場合
                ad_feature_input = ad_inputs[i]
                ad_feature_embeddings = [
                    embedding_layer(ad_feature_input[i])
                    for i, embedding_layer in enumerate(self.ad_feature_embedding_layers)
                ]
                h_ad = [torch.mean(embedding) for embedding in ad_feature_embeddings]
                h_ad = torch.stack(h_ad)
                meta_embedding = self.meta_embedding_generator(h_ad)
                ad_id_embeddings.append(meta_embedding)
        ad_id_embeddings = torch.stack(ad_id_embeddings)
        embeddings.insert(0, ad_id_embeddings)

        h = torch.cat(embeddings, dim=1)  # concat
        p = self.predictor(h)
        p = self.sigmoid(p)

        return p
