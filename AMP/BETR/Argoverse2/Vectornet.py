from config import *
from Decoder import *
from transformer import *
from graph_layers import *
from utils.geometry import *

import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch_geometric.data import DataLoader as GDataLoader

class VectorEncoder(nn.Module):
    """
    Vector Encoder Module

    Encodes vector representations using multiple transformer encoder layers with positional encoding.

    Args:
        d_model (int): The dimension of the model (embedding size).
        n_heads (int): The number of attention heads in the transformer layers.
        hidden_dim (int): The dimension of the hidden layer in the transformer.
        hidden_nheads (int): The number of attention heads in the hidden transformer layers.
        output_dim (int): The dimension of the output features.

    Attributes:
        p_enc (PositionalEncoding): Positional encoding layer.
        attn_layer1 (TransformerEncoderLayer): First transformer encoder layer.
        attn_layer2 (TransformerEncoderLayer): Second transformer encoder layer.
        attn_layer3 (TransformerEncoderLayer): Third transformer encoder layer.
    """
    def __init__(self, d_model, n_heads, hidden_dim, hidden_nheads, output_dim):
        super(VectorEncoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.hidden_nheads = hidden_nheads
        self.output_dim = output_dim

        self.p_enc = PositionalEncoding(self.d_model, max_len=60)
        self.attn_layer1 = TransformerEncoderLayer(self.d_model, self.n_heads, self.hidden_dim, self.hidden_dim)
        self.attn_layer2 = TransformerEncoderLayer(self.hidden_dim, self.hidden_nheads, self.hidden_dim, self.hidden_dim)
        self.attn_layer3 = TransformerEncoderLayer(self.hidden_dim, self.hidden_nheads, self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        Forward pass for vector encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, seq_len, output_dim).
        """
        x = self.p_enc(x)
        x, self.attn_weights1 = self.attn_layer1(x)
        x, self.attn_weights2 = self.attn_layer2(x)
        x, self.attn_weights3 = self.attn_layer3(x)
        return x


class LocalVectorNet(nn.Module):
    """
    Local Vector Network

    Encodes local vector representations for different entities (agents, objects, lanes) using either graph-based or vector-based encoders.

    Args:
        graph_encoder (bool, optional): Whether to use graph-based encoders. Defaults to False.

    Attributes:
        graph_encoder (bool): Flag indicating if graph-based encoders are used.
        agent_encoder (VectorEncoder or GraphVectorEncoder): Encoder for agent vectors.
        obj_encoder (VectorEncoder or GraphVectorEncoder): Encoder for object vectors.
        lane_encoder (VectorEncoder or GraphVectorEncoder): Encoder for lane vectors.
    """
    def __init__(self, graph_encoder=False):
        super(LocalVectorNet, self).__init__()
        self.graph_encoder = graph_encoder

        if graph_encoder:
            self.agent_encoder = GraphVectorEncoder(**GRAPH_AGENT_ENC)
            self.obj_encoder = GraphVectorEncoder(**GRAPH_OBJ_ENC)
            self.lane_encoder = GraphVectorEncoder(**GRAPH_LANE_ENC)
        else:
            self.agent_encoder = VectorEncoder(**AGENT_ENC)
            self.obj_encoder = VectorEncoder(**OBJ_ENC)
            self.lane_encoder = VectorEncoder(**LANE_ENC)

    def forward(self, x):
        """
        Forward pass for local vector encoding.

        Args:
            x (tuple): Tuple of tensors (agent_vectors, obj_vectors, lane_vectors).

        Returns:
            tuple: Encoded agent vectors, object vectors, and lane vectors.
        """
        agent_vectors, obj_vectors, lane_vectors = x
        agent_vectors = agent_vectors.to(DEVICE)
        obj_vectors = obj_vectors.to(DEVICE)
        lane_vectors = lane_vectors.to(DEVICE)

        agent_encoded = self.agent_encoder(agent_vectors)
        encoded_obj_vectors = self.obj_encoder(obj_vectors)
        encoded_lane_vectors = self.lane_encoder(lane_vectors)

        if EXPERIMENT_NAME == 'Argo-GNN-GNN':
            return agent_encoded, encoded_obj_vectors, encoded_lane_vectors

        agent_encoded = torch.mean(agent_encoded, axis=1)
        encoded_obj_vectors = torch.mean(encoded_obj_vectors, axis=1)
        encoded_lane_vectors = torch.mean(encoded_lane_vectors, axis=1)

        return agent_encoded, encoded_obj_vectors, encoded_lane_vectors

    def gnn_gnn_encoder(self, batch):
        """
        Encoder using Graph Neural Networks (GNN).

        Args:
            batch (tuple): Tuple of tensors (agent_data, obj_data, lane_data).

        Returns:
            tuple: Encoded agent vectors, object vectors, and lane vectors.
        """
        agent_graph_data = create_agent_graph_data(batch[0], 59)
        obj_graph_data = create_obj_graph(batch[1], 60)
        lane_graph_data = create_obj_graph(batch[2], 35)

        out = self([agent_graph_data, obj_graph_data, lane_graph_data])
        return out

    def to_trans(self, batch):
        """
        Converts input batch to tensor format suitable for transformers.

        Args:
            batch (tuple): Tuple of tensors (agent_vectors, obj_vectors, lane_vectors).

        Returns:
            torch.Tensor: Concatenated tensor of shape (batch_size, seq_len, feature_dim).
        """
        out = self.forward(batch[:-3])

        agent = out[0]
        agent = torch.cat([agent, torch.zeros((len(agent), 1), device=DEVICE)], 1)
        agent = agent.reshape(-1, 1, AGENT_ENC['output_dim'] + 1)

        obj = out[1]
        obj = torch.cat([obj, torch.ones((len(obj), 1), device=DEVICE)], 1)
        obj = obj.reshape(-1, OBJ_PAD_LEN, OBJ_ENC['output_dim'] + 1)

        lane = out[2]
        lane = torch.cat([lane, 2 * torch.ones((len(lane), 1), device=DEVICE)], 1)
        lane = lane.reshape(-1, LANE_PAD_LEN, LANE_ENC['output_dim'] + 1)

        data = torch.cat([agent, obj, lane], 1)
        return data

    def to_graph_data(self, batch):
        """
        Converts input batch to graph data format for GNN processing.

        Args:
            batch (tuple): Tuple of tensors (agent_data, obj_data, lane_data, gt, n_objs, n_lanes).

        Returns:
            Batch: Batch of graph data for GNN processing.
        """
        agent_data, obj_data, lane_data, gt, n_objs, n_lanes = batch
        if EXPERIMENT_NAME == 'Argo-GNN-GNN':
            out = self.gnn_gnn_encoder(batch)
        else:
            out = self(batch[:-3])

        batch_data = []
        batches = []

        agent = out[0]
        agent = torch.cat([agent, torch.zeros((len(agent), 1), device=DEVICE)], 1)

        obj = out[1]
        obj = torch.cat([obj, torch.ones((len(obj), 1), device=DEVICE)], 1)

        lane = out[2]
        lane = torch.cat([lane, 2 * torch.ones((len(lane), 1), device=DEVICE)], 1)

        for i in range(0, len(n_lanes) - 1):
            data_raw = torch.cat([agent[i].unsqueeze(0), obj[int(n_objs[i]):int(n_objs[i + 1])], lane[int(n_lanes[i]):int(n_lanes[i + 1])]])
            num_nodes = len(data_raw)

            data = Data()
            data.x = data_raw
            edge_index = fc_graph(num_nodes).to(DEVICE)
            data.edge_index = edge_index

            batch_data.append(data)
            batches += [*[i] * num_nodes]

        base_data = Batch.from_data_list(batch_data)
        base_data.batch = batches
        base_data.y = gt

        return base_data


class GlobalEncoder(nn.Module):
    """
    Global Encoder Module

    Encodes global features using a Transformer-based convolutional layer followed by global mean pooling.

    Args:
        in_dim (int): The dimension of the input features.
        out_dim (int): The dimension of the output features.

    Attributes:
        transformer_conv (TransformerConv): Transformer-based convolutional layer.
    """
    def __init__(self, in_dim, out_dim):
        super(GlobalEncoder, self).__init__()
        self.transformer_conv = TransformerConv(in_dim, out_dim)

    def forward(self, data):
        """
        Forward pass for global encoding.

        Args:
            data (Data): Input graph data containing node features and edge indices.

        Returns:
            torch.Tensor: Encoded global features of shape (batch_size, out_dim).
        """
        x, edge_index = data.x, data.edge_index
        x = self.transformer_conv(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        return x


class VectorNet(nn.Module):
    """
    VectorNet Model

    Combines local vector encoding with global encoding to produce final predictions. Supports different configurations based on experiment name.

    Attributes:
        local_encoder (LocalVectorNet): Local vector encoder for different vector representations.
        global_encoder (GlobalEncoder or TransformerEncoderLayer): Global encoder for feature aggregation.
        decoder (Decoder): Final decoder layer for generating predictions.
    """
    def __init__(self):
        super(VectorNet, self).__init__()

        if EXPERIMENT_NAME == 'Argo-GNN-GNN':
            self.local_encoder = LocalVectorNet(graph_encoder=True)
        else:
            self.local_encoder = LocalVectorNet()

        self.local_encoder.to(DEVICE)

        if EXPERIMENT_NAME == 'Argo-pad':
            self.global_encoder = TransformerEncoderLayer(**GLOBAL_ENC_TRANS)
        else:
            self.global_encoder = GlobalEncoder(**GLOBAL_ENC)

        self.global_encoder.to(DEVICE)

        self.decoder = Decoder(**DECODER)
        self.decoder.to(DEVICE)

    def forward(self, x):
        """
        Forward pass for the VectorNet model.

        Args:
            x (tuple): Tuple of tensors (agent_vectors, obj_vectors, lane_vectors, other_data).

        Returns:
            torch.Tensor: Final output predictions.
        """
        if EXPERIMENT_NAME == 'Argo-pad':
            encoded_vectors = self.local_encoder.to_trans(x)
            encoded_vectors, global_attention_weights = self.global_encoder(encoded_vectors)
            latent_vector = encoded_vectors.mean(axis=1)
        else:
            graph_data = self.local_encoder.to_graph_data(x)
            graphloader = GDataLoader(graph_data, batch_size=len(graph_data))
            latent_vector = self.global_encoder(next(iter(graphloader)))

        out = self.decoder(latent_vector)
        return out
