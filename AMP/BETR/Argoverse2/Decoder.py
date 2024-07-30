import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    '''
    DenseLayer
    A fully connected layer followed by batch normalization, activation, and dropout.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        alpha (float): Dropout rate. Default is 0.1.
        act (str): Activation function name. Default is 'relu'.

    Attributes:
        alpha (float): Dropout rate.
        fc (nn.Linear): Fully connected layer.
        bn (nn.BatchNorm1d): Batch normalization layer.
        act (nn.Module): Activation function.
        act_name (str): Name of the activation function.
        do (nn.Dropout): Dropout layer.
    '''
    def __init__(self, in_dim, out_dim, alpha=0.1, act='relu'):
        super(DenseLayer, self).__init__()
        self.alpha = alpha
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU() if act == 'relu' else None
        self.act_name = act
        self.do = nn.Dropout(self.alpha)

    def forward(self, x):
        x = self.fc(x)
        if len(x) > 1:
            x = self.bn(x)
        
        if self.act is not None: 
            x = self.act(x)
            x = self.do(x)
        
        return x


class Decoder(nn.Module):
    '''
    Decoder
    A decoder network consisting of multiple DenseLayer instances.

    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        out_dim (int): Output dimension.

    Attributes:
        fc1 (DenseLayer): First dense layer.
        fc2 (DenseLayer): Second dense layer.
        fc3 (DenseLayer): Third dense layer.
    '''
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()

        self.fc1 = DenseLayer(in_dim, hidden_dim, alpha=0.2)
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, alpha=0.1)
        self.fc3 = DenseLayer(hidden_dim, out_dim, act=None)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
