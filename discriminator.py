# -*- coding=utf-8 -*-


import torch.nn as nn
from torch_geometric.nn import GATConv


class LinearDiscriminator(nn.Module):
	def __init__(self, batch_size, seq_len, graph_size, feature_channels):
		super(LinearDiscriminator, self).__init__()
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.graph_size = graph_size
		self.feature_channels = feature_channels
		self.linear_module = nn.Sequential(
			nn.Linear(in_features=self.seq_len * self.graph_size * self.feature_channels, out_features=512),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=512, out_features=256),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=256, out_features=128),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=128, out_features=1)
		)
	def forward(self, model_input):
		linear_module_input = model_input.view(self.batch_size, self.seq_len * self.graph_size * self.feature_channels)
		score = self.linear_module(linear_module_input)
		return score


class GraphConvolutionalDiscriminator(nn.Module):
	def __init__(self, batch_size, seq_len, graph_size, feature_channels):
		def _conv1d_ln_lrelu(in_channels, out_channels, kernel_size, stride, padding, bias=False):
			conv1d_ln_lrelu_module = nn.Sequential(
				nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
				nn.BatchNorm1d(num_features=out_channels, affine=True),
				nn.LeakyReLU(negative_slope=0.2)
			)
			return conv1d_ln_lrelu_module
		super(GraphConvolutionalDiscriminator, self).__init__()
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.graph_size = graph_size
		self.feature_channels = feature_channels
		self.graph_convolutional_module = GATConv(in_channels=self.feature_channels, out_channels=self.feature_channels)
		self.temporal_attention_module = nn.TransformerEncoderLayer(d_model=self.graph_size * self.feature_channels, nhead=3)
		self.convolutional_module_3 = nn.Sequential(
			nn.Conv1d(in_channels=self.graph_size * self.feature_channels, out_channels=self.graph_size * self.feature_channels * 2, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2),
			_conv1d_ln_lrelu(self.graph_size * self.feature_channels * 2, self.graph_size * self.feature_channels * 4, 3, 2, 1),
			nn.Conv1d(in_channels=self.graph_size * self.feature_channels * 4, out_channels=self.graph_size * self.feature_channels * 8, kernel_size=3, stride=2, padding=1)
		)
		self.convolutional_module_4 = nn.Sequential(
			nn.Conv1d(in_channels=self.graph_size * self.feature_channels, out_channels=self.graph_size * self.feature_channels * 2, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2),
			_conv1d_ln_lrelu(self.graph_size * self.feature_channels * 2, self.graph_size * self.feature_channels * 4, 4, 2, 1),
			nn.Conv1d(in_channels=self.graph_size * self.feature_channels * 4, out_channels=self.graph_size * self.feature_channels * 8, kernel_size=4, stride=2, padding=1)
		)
		self.convolutional_module_5 = nn.Sequential(
			nn.Conv1d(in_channels=self.graph_size * self.feature_channels, out_channels=self.graph_size * self.feature_channels * 2, kernel_size=5, stride=2, padding=2),
			nn.LeakyReLU(negative_slope=0.2),
			_conv1d_ln_lrelu(self.graph_size * self.feature_channels * 2, self.graph_size * self.feature_channels * 4, 5, 2, 2),
			nn.Conv1d(in_channels=self.graph_size * self.feature_channels * 4, out_channels=self.graph_size * self.feature_channels * 8, kernel_size=5, stride=2, padding=2)
		)
		self.linear_module = nn.Sequential(
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(self.graph_size * self.feature_channels * 8, self.graph_size * self.feature_channels * 4),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(self.graph_size * self.feature_channels * 4, self.graph_size * self.feature_channels * 2),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(self.graph_size * self.feature_channels * 2, self.graph_size * self.feature_channels),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(self.graph_size * self.feature_channels, 128),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(128, 1)
		)
	def forward(self, model_input):
		graph_convolutional_module_output = self.graph_convolutional_module(model_input.x, model_input.edge_index)
		temporal_attention_module_input = graph_convolutional_module_output.view(self.batch_size, self.seq_len, self.graph_size * self.feature_channels)
		temporal_attention_module_output = self.temporal_attention_module(temporal_attention_module_input)
		convolutional_module_input = temporal_attention_module_output.transpose(2, 1)
		#convolutional_module_output = self.convolutional_module_3(convolutional_module_input)
		convolutional_module_output = self.convolutional_module_4(convolutional_module_input)
		#convolutional_module_output = self.convolutional_module_5(convolutional_module_input)
		linear_module_input = convolutional_module_output.view(self.batch_size, self.graph_size * self.feature_channels * 8)
		score = self.linear_module(linear_module_input)
		return score