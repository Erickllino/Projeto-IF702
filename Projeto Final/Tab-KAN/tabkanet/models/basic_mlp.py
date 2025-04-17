import torch
import torch.nn as nn
import torch.nn.functional as F
from tabkanet.models.base_blocks import Activation, ColumnEmbedding
from typing import Dict, List
import math





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

class BasicNet(nn.Module):
    
    def __init__(self, 
                 output_dim: int, vocabulary: Dict[str, Dict[str, int]], num_continuous_features: int,
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, attn_dropout_rate: float,
                 mlp_hidden_dims: List[int], activation: str, ffn_dropout_rate: float,learninable_noise:bool, geoaffine:bool):
        super(BasicNet, self).__init__()
        self.classifier = MLP( len(vocabulary) + num_continuous_features, output_dim, mlp_hidden_dims, activation, ffn_dropout_rate)
        self.cat_count=len(vocabulary)
        self.norm = nn.LayerNorm([num_continuous_features])

    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor):
        continuous_x=self.norm(continuous_x)

        if self.cat_count==0:
            x = self.classifier(continuous_x)
        else:
            x = torch.cat([categorical_x, continuous_x], dim=-1)
            x = self.classifier(x)
        return x
    




