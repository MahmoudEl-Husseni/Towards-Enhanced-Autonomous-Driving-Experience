import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData, Data
from torch_geometric.nn import GCNConv, global_mean_pool, TransformerConv
from torch_geometric.data import DataLoader as GDataLoader


class GNN_Layer(nn.Module):
    '''
    GNN_Layer
    A Graph Neural Network layer using TransformerConv followed by an activation function.
    
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        n_heads (int): Number of attention heads. Default is 1.
        pool (bool): Whether to apply global mean pooling. Default is False.

    Attributes:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        n_heads (int): Number of attention heads.
        transformer_conv (TransformerConv): Transformer convolution layer.
        act (function): Activation function (ReLU).
        pool (bool): Flag to apply global mean pooling.
    '''
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 1, pool: bool = False):
        super(GNN_Layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.transformer_conv = TransformerConv(in_dim, out_dim, heads=self.n_heads)
        self.act = F.relu
        self.pool = pool

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        x = self.transformer_conv(x, edge_index)
        x = self.act(x)

        if self.pool:
            x = global_mean_pool(x, data.batch)

        return x


class GraphVectorEncoder(nn.Module):
    '''
    GraphVectorEncoder
    Encodes graph data into vector representations using multiple GNN_Layer instances.
    
    Args:
        d_model (int): Dimension of the model.
        d_hidden (int): Dimension of the hidden layer.
        d_out (int): Dimension of the output layer.
        n_heads (int): Number of attention heads.

    Attributes:
        layer1 (GNN_Layer): First GNN layer.
        layer2 (GNN_Layer): Second GNN layer.
        layer3 (GNN_Layer): Third GNN layer with pooling.
    '''
    def __init__(self, d_model: int, d_hidden: int, d_out: int, n_heads: int):
        super(GraphVectorEncoder, self).__init__()
        self.layer1 = GNN_Layer(d_model, d_hidden, n_heads)
        self.layer2 = GNN_Layer(d_hidden * n_heads, d_hidden, n_heads)
        self.layer3 = GNN_Layer(d_hidden * n_heads, d_out, 1, pool=True)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.layer1(data)
        data = Batch(x=x, edge_index=data.edge_index, batch=data.batch)
        x = self.layer2(data)
        data = Batch(x=x, edge_index=data.edge_index, batch=data.batch)
        x = self.layer3(data)
        return x
