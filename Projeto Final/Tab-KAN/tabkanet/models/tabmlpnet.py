from typing import Dict, List
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from torch.nn import BatchNorm1d
from tabkanet.models.base_blocks import Activation, ColumnEmbedding

class CatEncoder(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, int]],embedding_dim: int):
        super(CatEncoder, self).__init__()
        self.vocabulary = vocabulary
        self.embedding_dim=embedding_dim
        self.columnembedding=ColumnEmbedding(vocabulary, embedding_dim)

    def forward(self, x: torch.Tensor,  continuous_x_res:torch.Tensor):
        batch_size = x.size(0)
        x = [self.columnembedding(x[:, i], col) for i, col in enumerate(self.vocabulary)]
        x = torch.stack(x, dim=1)
        x = torch.cat((x, continuous_x_res), dim=1)  
        return x


class NoiseLearnModule(nn.Module):
    def __init__(self, feature_dim, n_bins=4, noise_scale=0.1):
        super(NoiseLearnModule, self).__init__()
        # 为每个特征的每个分位数范围创建独立的噪声参数
        self.noise_params = nn.Parameter(torch.zeros(feature_dim * n_bins))
        # 控制噪声的尺度
        self.noise_scale = noise_scale

    def forward(self,  x, bins):
        # x shape: (batch_size, feature_dim)
        # bins shape: (feature_dim, n_bins + 1), 存储每个特征的分位数边界
        batch_size, feature_dim = x.shape
        # 扩展噪声参数以匹配特征的分位数范围
        noise_params = self.noise_params.view(feature_dim, -1)
        # 初始化噪声，形状为 (batch_size, feature_dim)
        noise = torch.zeros_like(x)
        
        # 对每个特征进行处理
        for i in range(feature_dim):
            # 计算每个样本在当前特征中属于哪个分位数区间
            feature_values = x[:, i]
            for j in range(len(bins[i]) - 1):
                lower_bound = bins[i][j]
                upper_bound = bins[i][j + 1]
                # 确定当前特征值是否在当前区间内
                in_range = (feature_values >= lower_bound) & (feature_values < upper_bound)
                # 如果在区间内，应用对应的噪声参数
                if torch.any(in_range):
                    noise[in_range, i] = torch.randn(in_range.sum(), device=x.device) * torch.sigmoid(noise_params[i, j]) * self.noise_scale
        
        # 将噪声添加到输入数据中
        return x + noise






class NumEncoder(nn.Module):
    def __init__(self, num_features: int):
        """
        Continuous feature encoder.

        Parameters:
        - num_features (int): Number of continuous features.
        """
        super(NumEncoder, self).__init__()
        self.norm = nn.LayerNorm([num_features])
        
    def forward(self, x: torch.Tensor):
        return self.norm(x)


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.batch_norm = BatchNorm1d(num_numerical_types)

    def forward(self, x):
        x = self.batch_norm(x)  
        return x

class NumEncoderTransformer(nn.Module):
    def __init__(self, num_continuous_features: int, embedding_dim: int,learninable_noise:bool,bins):
        super(NumEncoderTransformer, self).__init__()
        self.norm = nn.LayerNorm([num_continuous_features])
        self.num_continuous_features = num_continuous_features
        self.mlpclassifier1=nn.Linear(num_continuous_features,max(2*num_continuous_features+1,embedding_dim))
        self.mlpclassifier2=nn.Linear(max(2*num_continuous_features+1,embedding_dim),embedding_dim * num_continuous_features)
        self.numerical_embedder = NumericalEmbedder(1, num_continuous_features)

        if learninable_noise:
            self.noise_module = NoiseLearnModule(num_continuous_features)

        self.learninable_noise=learninable_noise

        self.bins=bins



    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        if self.learninable_noise:
            x = self.noise_module(x,self.bins)

        x = self.numerical_embedder(x)
        x = x.squeeze(-1) 
        x = self.mlpclassifier1(x)
        x = self.mlpclassifier2(x)
        return x





class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str, dropout_rate: float):
        """
        MLP block.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - activation (str): Activation function.
        - dropout_rate (float): Dropout rate.
        """
        super(MLPBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm([output_dim]),
            Activation(activation),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int], activation: str, 
                 dropout_rate: float):
        """
        MLP model.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - hidden_dims (List[int]): List of hidden layer dimensions.
        - activation (str): Activation function.
        - dropout_rate (float): Dropout rate.
        """
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims
        self.model = nn.Sequential(*(
            [
                MLPBlock(
                    dims[i], dims[i + 1], 
                    activation, dropout_rate) 
                for i in range(len(dims) - 1)] + \
                [nn.Linear(dims[-1], output_dim)]))
        
    def forward(self, x):
        return self.model(x)
from typing import Dict, List
from typing import Optional, Dict, Any
from torch import Tensor

class TabMLPNet(nn.Module):
    def __init__(self, 
                 output_dim: int, vocabulary: Dict[str, Dict[str, int]], num_continuous_features: int,
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, attn_dropout_rate: float,
                 mlp_hidden_dims: List[int], activation: str, ffn_dropout_rate: float,learninable_noise:bool, bins: Optional[List[Tensor]]):
        super(TabMLPNet, self).__init__()
        self.embedding_dim=embedding_dim
        self.len_vocabulary=len(vocabulary)
        self.num_continuous_features = num_continuous_features
        self.tranformer_model =  nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                    d_model=embedding_dim, 
                    nhead=nhead,
                    dim_feedforward=dim_feedforward, 
                    dropout=attn_dropout_rate,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True),
                    num_layers=num_layers,
                    norm=nn.LayerNorm([embedding_dim]))
        self.encoders = nn.ModuleDict({
            'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim),
            'continuous_feature_encoder': NumEncoderTransformer(num_continuous_features, embedding_dim,learninable_noise,bins),
        })

        self.classifier = MLP(embedding_dim * (len(vocabulary) + num_continuous_features), output_dim, mlp_hidden_dims, activation, ffn_dropout_rate)



    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor):
        continuous_x = self.encoders['continuous_feature_encoder'](continuous_x)
        batch_size = continuous_x.size(0)

        if self.len_vocabulary==0:
            x = continuous_x.view(batch_size, self.num_continuous_features, self.embedding_dim)
            x = self.tranformer_model(x).view(batch_size, -1)
            x = self.classifier(x)
        else:
            continuous_x = continuous_x.view(batch_size, self.num_continuous_features, self.embedding_dim)
            x = self.encoders['categorical_feature_encoder'](categorical_x,continuous_x)
            x = self.tranformer_model(x).view(batch_size, -1)
            x = self.classifier(x)
        return x
    








