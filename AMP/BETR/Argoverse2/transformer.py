import torch 
import torch.nn as nn
import torch.nn.functional as F
from config import *

class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module for Transformers

    Adds positional encodings to input embeddings to provide information about the position of tokens in the sequence.
    Uses sine and cosine functions of different frequencies for encoding.

    Args:
        d_model (int): The dimension of the model (embedding size).
        max_len (int, optional): The maximum length of the sequence. Defaults to 60.
    
    Attributes:
        encoding (torch.Tensor): Tensor containing the positional encodings.
    """
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2:
            self.encoding[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        """
        Forward pass for positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encoding added to the input.
        """
        return x + self.encoding[:, :x.shape[1]].to(DEVICE)


class Feedforward(nn.Module):
    """
    Feedforward Neural Network Module

    A simple feedforward network used in the Transformer architecture. Consists of two linear layers with ReLU activation and dropout.

    Args:
        d_model (int): The dimension of the input features.
        d_ff (int): The dimension of the hidden layer.
        output_dim (int): The dimension of the output features.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, d_model, d_ff, output_dim, dropout=0.1):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiheadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Module

    Applies multi-head self-attention mechanism to the input sequences. Scales and computes attention weights to capture relationships between tokens.

    Args:
        input_dim (int): The dimension of the input features.
        num_heads (int): The number of attention heads.

    Attributes:
        query (nn.Linear): Linear layer for queries.
        key (nn.Linear): Linear layer for keys.
        value (nn.Linear): Linear layer for values.
        fc_out (nn.Linear): Linear layer for output.
    """
    def __init__(self, input_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim, device=DEVICE)
        self.key = nn.Linear(input_dim, input_dim, device=DEVICE)
        self.value = nn.Linear(input_dim, input_dim, device=DEVICE)
        self.fc_out = nn.Linear(input_dim, input_dim, device=DEVICE)

    def forward(self, query, key, value):
        """
        Forward pass for multi-head self-attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, input_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, input_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim).
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size = key.shape[0]
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Split the queries, keys, and values into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scaled_attention = torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        attention_weights = F.softmax(scaled_attention, dim=-1)
        output = torch.matmul(attention_weights, value)

        # Concatenate and linearly project the output
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dim)
        output = self.fc_out(output)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer

    Implements a single encoder layer in the Transformer model, which consists of multi-head self-attention and feedforward network components.

    Args:
        d_model (int): The dimension of the input features.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the hidden layer in the feedforward network.
        output_dim (int): The dimension of the output features.
        dropout (float, optional): Dropout rate. Defaults to 0.1.

    Attributes:
        self_attention (MultiheadSelfAttention): Multi-head self-attention mechanism.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        convert (nn.Linear): Linear layer for dimensionality conversion.
        act (nn.ReLU): ReLU activation function.
        layernorm_convert (nn.LayerNorm): Layer normalization after conversion.
        feedforward (Feedforward): Feedforward network.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, d_model, num_heads, d_ff, output_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multihead Self-Attention
        self.self_attention = MultiheadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model).to(DEVICE)

        # Conversion layer
        self.convert = nn.Linear(d_model, output_dim).to(DEVICE)
        self.act = nn.ReLU().to(DEVICE)
        self.layernorm_convert = nn.LayerNorm(output_dim).to(DEVICE)

        # Feedforward
        self.feedforward = Feedforward(output_dim, d_ff, output_dim, dropout)
        self.norm2 = nn.LayerNorm(output_dim).to(DEVICE)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor after applying self-attention, feedforward network, and layer normalization.
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        # Self-Attention
        attention_output, att_weights = self.self_attention(x, x, x)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Conversion layer
        x = self.convert(x)
        x = self.act(x)
        x = self.layernorm_convert(x)

        # Feedforward
        ff_output = self.feedforward(x)
        ff_output = x + self.dropout(ff_output)
        x = self.norm2(ff_output)

        return x, att_weights
