import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd

class GraphConvolution(nn.Module):
  """
  Graph convolution layer from Kipf and Welling's paper:

  Semi-Supervised Classification with Graph Convolutional Networks
  https://arxiv.org/abs/1609.02907

  Parameters
  ----------
  in_features: int
    The number of input features of each node in the graph.

  out_features: int
    The number of output features for nodes in the graph.

  bias: bool, default: False
    Bias of the layer.
  """
  def __init__(self, in_features, out_features, bias=False):
    super(GraphConvolution, self).__init__()
    self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
    if bias:
      self.bias = nn.Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    """Initialize weight and bias for the layer."""
    std = 1. / math.sqrt(self.weights.size(1))
    self.weights.data.uniform_(-std, std)
    if self.bias is not None:
      self.bias.data.uniform_(-std, std)

  def forward(self, features, adj):
    """
    Forward the input through the layer.

    Parameters
    ----------
    features: tensor shape of (n_nodes, in_features)
      Input features of the nodes in the graph.

    adj: tensor shape of (n_nodes, n_nodes)
      Normalized adjacency matrix of the graph.

    Returns
    -------
    output: tensor shape of (n_nodes, out_features)
      Output of the layer.
    """
    support = torch.mm(features, self.weights)
    output = torch.spmm(adj, support)
    if self.bias is not None:
      output += self.bias
    return output

class STGNN(nn.Module):
  """
  An implementation of Spatio-Temporal Graph Neural Network model
  from Amol Kapoor et. al. paper:
  
  Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks
  https://arxiv.org/abs/2007.03113

  Parameters
  ----------
  temp_feat: int, default: 7
    The number of time steps (context length) in the temporal features.
  
  in_feat: int, default: 64
    The number of input features to the Graph convolution layer,
    or output units of the first Dense layer.

  hidden_feat: int, default: 32
    The number of units in graph hidden layer.

  out_feat: int, default: 32
    The number of output units of the last graph convolution layer.
  
  pred_feat: int, default: 1
    The number of time steps ahead to predict.

  drop_rate: float, default: 0
    Drop rate of Dropout layer.

  bias: bool, default: False,
    Bias the graph convolution layers.
  """
  def __init__(self,
               temp_feat=7,
               in_feat=64,
               hidden_feat=32,
               out_feat=32,
               pred_feat=1,
               drop_rate=0.,
               bias=False):
    super(STGNN, self).__init__()
    self.time_steps = temp_feat
    self.linear1 = nn.Linear(temp_feat, in_feat)
    self.layer1 = GraphConvolution(in_feat, hidden_feat, bias=bias)
    self.layer2 = GraphConvolution(hidden_feat + in_feat, out_feat, bias=bias)
    self.linear2 = nn.Linear(out_feat + in_feat, pred_feat)

    self.dropout = nn.Dropout(p=drop_rate)
    self.drop_rate = drop_rate

  def forward(self, temporal_features, adj):
    """
    Forward the input through the model.

    Parameters
    ----------
    features: tensor shape of (n_nodes, temp_feat)
      Temporal features of the nodes in the graph through temp_feat time steps.

    adj: tensor shape of (n_nodes, n_nodes)
      Normalized adjacency matrix of the graph.

    Returns
    -------
    output: tensor shape of (n_nodes, pred_feat)
      Prediction of the model for each node.
    """
    h0 = F.relu(self.linear1(temporal_features))
    h0 = F.dropout(h0, p=self.drop_rate, training=self.training)

    h1 = F.relu(self.layer1(h0, adj))
    h1 = F.dropout(h1, p=self.drop_rate, training=self.training)
    h1 = torch.cat([h1, h0], dim=1)

    h2 = F.relu(self.layer2(h1, adj))
    h2 = F.dropout(h2, p=self.drop_rate, training=self.training)
    h2 = torch.cat([h2, h0], dim=1)

    output = self.linear2(h2)
    return output


class ModifiedSTGNN(nn.Module):
  """
  An implementation of Spatio-Temporal Graph Neural Network model
  from Amol Kapoor et. al. paper:
  
  Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks
  https://arxiv.org/abs/2007.03113

  Parameters
  ----------
  temp_feat: int, default: 7
    The number of time steps (context length) in the temporal features.
  
  in_feat: int, default: 64
    The number of input features to the Graph convolution layer,
    or output units of the first Dense layer.

  hidden_feat: int, default: 32
    The number of units in graph hidden layer.

  out_feat: int, default: 32
    The number of output units of the last graph convolution layer.
  
  pred_feat: int, default: 1
    The number of time steps ahead to predict.

  drop_rate: float, default: 0
    Drop rate of Dropout layer.

  bias: bool, default: False,
    Bias the graph convolution layers.
  """
  def __init__(self,
               temp_feat=7,
               in_feat=64,
               hidden_feat=32,
               out_feat=32,
               pred_feat=1,
               drop_rate=0.,
               bias=False):
    super(ModifiedSTGNN, self).__init__()
    self.time_steps = temp_feat
    self.linear1 = nn.Linear(temp_feat, in_feat)

    self.theta1 = nn.Parameter(torch.FloatTensor(in_feat,
                                                 hidden_feat))
    
    self.theta2 = nn.Parameter(torch.FloatTensor(hidden_feat + in_feat,
                                                 out_feat))

    self.linear2 = nn.Linear(out_feat + in_feat, pred_feat)

    self.dropout = nn.Dropout(p=drop_rate)
    self.drop_rate = drop_rate
    self.reset_parameters()

  def reset_parameters(self):
    """Initialize weights for the model."""
    std = 1. / math.sqrt(self.theta1.size(1))
    self.theta1.data.uniform_(-std, std)

    std = 1. / math.sqrt(self.theta2.size(1))
    self.theta2.data.uniform_(-std, std)

  def forward(self, temporal_features, adj):
    """
    Forward the input through the model.

    Parameters
    ----------
    features: tensor shape of (batch_size, n_nodes, temp_feat)
      Temporal features of the nodes in the graph through temp_feat time steps.

    adj: tensor shape of (n_nodes, n_nodes)
      Normalized adjacency matrix of the graph.

    Returns
    -------
    output: tensor shape of (batch_size, n_nodes, pred_feat)
      Prediction of the model for each node.
    """
    h0 = F.relu(self.linear1(temporal_features))
    h0 = F.dropout(h0, p=self.drop_rate, training=self.training)

    h1 = torch.matmul(h0, self.theta1)
    h1 = torch.einsum('ij,jkl->kil', [adj, h1.permute(1, 0, 2)])
    h1 = F.relu(h1)
    h1 = F.dropout(h1, p=self.drop_rate, training=self.training)
    h1 = torch.cat([h1, h0], dim=-1)

    h2 = torch.matmul(h1, self.theta2)
    h2 = torch.einsum('ij,jkl->kil', [adj, h2.permute(1, 0, 2)])
    h2 = F.relu(h2)
    h2 = F.dropout(h2, p=self.drop_rate, training=self.training)
    h2 = torch.cat([h2, h0], dim=-1)

    output = self.linear2(h2).squeeze(-1)

    return output

class ProposedSTGNN(nn.Module):
  """
  Our proposed model for the Spatio-Temporal forecasting problem.

  Parameters
  ----------
  n_nodes: int
    The number of nodes in the graph.
  
  time_steps: int, default: 7
    The number of time steps (context length) in the temporal features.

  predicted_time_steps: int, default: 1
    The number of time steps ahead to predict.

  in_channels: int, default: 1
    The number of input channels (of temporal features/time series).

  spatial_channels: int, default: 16
    The number of channels in the first graph convolution layer.

  spatial_hidden_channels: int, default: 16
    The number of channels in the hidden graph convolution layer.
  
  spatial_out_channels: int, default: 16
    The number of channels in the first graph convolution layer.

  out_channels: int, default: 16
    The number of channels in the last convolution layer.

  temporal_kernel: int, default: 3
    The number of kernels for 1D convolution.

  drop_rate: float, default: 0
    The drop rate of dropout layer.

  batch_norm: bool, default: False
    Batch normalize the embedding of nodes after the first convolution layer.
  """
  def __init__(self,
               n_nodes,
               time_steps=7,
               predicted_time_steps=1,
               in_channels=1,
               spatial_channels=16,
               spatial_hidden_channels=16,
               spatial_out_channels=16,
               out_channels=16,
               temporal_kernel=3,
               drop_rate=0.,
               batch_norm=False):
    
    super(ProposedSTGNN, self).__init__()
    self.drop_rate = drop_rate
    self.time_steps = time_steps
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=spatial_channels,
                           kernel_size=(1, temporal_kernel))
    
    self.theta1 = nn.Parameter(torch.FloatTensor(spatial_channels,
                                                 spatial_hidden_channels))
    
    self.theta2 = nn.Parameter(torch.FloatTensor(spatial_hidden_channels + spatial_channels,
                                                 spatial_out_channels))
    
    self.conv2 = nn.Conv2d(in_channels=spatial_out_channels + spatial_channels,
                           out_channels=out_channels,
                           kernel_size=(1, temporal_kernel))

    if batch_norm:
      self.batch_norm = nn.BatchNorm2d(n_nodes)
    else:
      self.batch_norm = None
    
    self.fc = nn.Linear((time_steps - (temporal_kernel - 1) * 2) * (out_channels),
                         predicted_time_steps)

    self.reset_parameters()

  def reset_parameters(self):
    """Initialize weights for the model."""
    std = 1. / math.sqrt(self.theta1.size(1))
    self.theta1.data.uniform_(-std, std)

    std = 1. / math.sqrt(self.theta2.size(1))
    self.theta2.data.uniform_(-std, std)

  def forward(self, input, adj):
    """
    Parameters
    ----------
    input: tensor shape of (batch_size, n_nodes, time_steps, in_feautures)
      Input to the model.

    adj: tensor shape of (n_nodes, n_nodes)
      Normalized adjacency matrix of the graph.

    Returns
    -------
    output: tensor shape of (batch_size, n_nodes, predicted_time_steps)
      Predictions of the model for each node.
    """
    input = input.permute(0, 3, 1, 2)
    h0 = self.conv1(input)
    h0 = h0.permute(0, 2, 3, 1)
    h0 = F.relu(h0)
    
    if self.batch_norm is not None:
      h0 = self.batch_norm(h0)

    h0 = F.dropout(h0, p=self.drop_rate, training=self.training)

    h1 = torch.matmul(h0, self.theta1)
    h1 = torch.einsum('ij,jklm->kilm', [adj, h1.permute(1, 0, 2, 3)])

    h1 = F.relu(h1)
    h1 = F.dropout(h1, p=self.drop_rate, training=self.training)
    h1 = torch.cat([h1, h0], dim=-1)

    h2 = torch.matmul(h1, self.theta2)
    h2 = torch.einsum('ij,jklm->kilm', [adj, h2.permute(1, 0, 2, 3)])
    
    h2 = F.relu(h2)
    h2 = F.dropout(h2, p=self.drop_rate, training=self.training)
    h2 = torch.cat([h2, h0], dim=-1)

    h2 = self.conv2(h2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    h2 = F.relu(h2)

    output = self.fc(h2.reshape(h2.size(0), h2.size(1), -1))

    return output

class ProposedNoSkipConnectionSTGNN(nn.Module):
  """
  Our proposed model for the Spatio-Temporal forecasting problem
  without skip connection.

  Parameters
  ----------
  n_nodes: int
    The number of nodes in the graph.
  
  time_steps: int, default: 7
    The number of time steps (context length) in the temporal features.

  predicted_time_steps: int, default: 1
    The number of time steps ahead to predict.

  in_channels: int, default: 1
    The number of input channels (of temporal features/time series).

  spatial_channels: int, default: 16
    The number of channels in the first graph convolution layer.

  spatial_hidden_channels: int, default: 16
    The number of channels in the hidden graph convolution layer.
  
  spatial_out_channels: int, default: 16
    The number of channels in the first graph convolution layer.

  out_channels: int, default: 16
    The number of channels in the last convolution layer.

  temporal_kernel: int, default: 3
    The number of kernels for 1D convolution.

  drop_rate: float, default: 0
    The drop rate of dropout layer.

  batch_norm: bool, default: False
    Batch normalize the embedding of nodes after the first convolution layer.
  """
  def __init__(self,
               n_nodes,
               time_steps=7,
               predicted_time_steps=1,
               in_channels=1,
               spatial_channels=16,
               spatial_hidden_channels=16,
               spatial_out_channels=16,
               out_channels=16,
               temporal_kernel=3,
               drop_rate=0.,
               batch_norm=False):
    
    super(ProposedNoSkipConnectionSTGNN, self).__init__()
    self.drop_rate = drop_rate
    self.time_steps = time_steps
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=spatial_channels,
                           kernel_size=(1, temporal_kernel))
    
    self.theta1 = nn.Parameter(torch.FloatTensor(spatial_channels,
                                                 spatial_hidden_channels))
    
    self.theta2 = nn.Parameter(torch.FloatTensor(spatial_hidden_channels,
                                                 spatial_out_channels))
    
    self.conv2 = nn.Conv2d(in_channels=spatial_out_channels,
                           out_channels=out_channels,
                           kernel_size=(1, temporal_kernel))

    if batch_norm:
      self.batch_norm = nn.BatchNorm2d(n_nodes)
    else:
      self.batch_norm = None
    
    self.fc = nn.Linear((time_steps - (temporal_kernel - 1) * 2) * (out_channels),
                         predicted_time_steps)

    self.reset_parameters()

  def reset_parameters(self):
    """Initialize weights for the model."""
    std = 1. / math.sqrt(self.theta1.size(1))
    self.theta1.data.uniform_(-std, std)

    std = 1. / math.sqrt(self.theta2.size(1))
    self.theta2.data.uniform_(-std, std)

  def forward(self, input, adj):
    """
    Parameters
    ----------
    input: tensor shape of (batch_size, n_nodes, time_steps, in_feautures)
      Input to the model.

    adj: tensor shape of (n_nodes, n_nodes)
      Normalized adjacency matrix of the graph.

    Returns
    -------
    output: tensor shape of (batch_size, n_nodes, predicted_time_steps)
      Predictions of the model for each node.
    """
    input = input.permute(0, 3, 1, 2)
    h0 = self.conv1(input)
    h0 = h0.permute(0, 2, 3, 1)
    h0 = F.relu(h0)
    
    if self.batch_norm is not None:
      h0 = self.batch_norm(h0)

    h0 = F.dropout(h0, p=self.drop_rate, training=self.training)

    h1 = torch.matmul(h0, self.theta1)
    h1 = torch.einsum('ij,jklm->kilm', [adj, h1.permute(1, 0, 2, 3)])

    h1 = F.relu(h1)
    h1 = F.dropout(h1, p=self.drop_rate, training=self.training)
    # h1 = torch.cat([h1, h0], dim=-1)

    h2 = torch.matmul(h1, self.theta2)
    h2 = torch.einsum('ij,jklm->kilm', [adj, h2.permute(1, 0, 2, 3)])
    
    h2 = F.relu(h2)
    h2 = F.dropout(h2, p=self.drop_rate, training=self.training)
    # h2 = torch.cat([h2, h0], dim=-1)

    h2 = self.conv2(h2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    h2 = F.relu(h2)

    output = self.fc(h2.reshape(h2.size(0), h2.size(1), -1))

    return output