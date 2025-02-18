# -*- coding=utf-8 -*-


import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from unet import Unet1d


def _add_mask_activation(self):
	self.sigmoid = nn.Sigmoid()
	self.hardtanh = nn.Hardtanh(min_val=0, max_val=1)
	self.activation = lambda x: self.hardtanh(self.sigmoid(x / 0.66) * 1.2 - 0.1)


def _add_data_activation(self):
	self.activation = nn.Tanh()


class LinearGenerator(nn.Module):
	def __init__(self, noise_size, batch_size, seq_len, graph_size, feature_channels):
		super(LinearGenerator, self).__init__()
		self.noise_size = noise_size
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.graph_size = graph_size
		self.feature_channels = feature_channels
		self.linear_module = nn.Sequential(
			nn.Linear(in_features=self.noise_size, out_features=256),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=256, out_features=512),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=512, out_features=self.seq_len * self.graph_size * self.feature_channels)
		)
	def forward(self, noise, edge_index):
		linear_module_output = self.linear_module(noise)
		activation_output = self.activation(linear_module_output)
		fake = Data(x=activation_output.view(self.batch_size * self.seq_len * self.graph_size, self.feature_channels), edge_index=edge_index)
		return fake


class GraphConvolutionalGenerator(nn.Module):
	def __init__(self, noise_size, batch_size, seq_len, graph_size, feature_channels):
		def _deconv1d_bn_lrelu(in_channels, out_channels, kernel_size, stride, padding, output_padding=0, bias=False):
			deconv1d_bn_lrelu_module = nn.Sequential(
				nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias),
				nn.BatchNorm1d(num_features=out_channels),
				nn.LeakyReLU(negative_slope=0.2)
			)
			return deconv1d_bn_lrelu_module
		super(GraphConvolutionalGenerator, self).__init__()
		self.noise_size = noise_size
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.graph_size = graph_size
		self.feature_channels = feature_channels
		self.linear_module = nn.Sequential(
			nn.Linear(in_features=self.noise_size, out_features=self.graph_size * self.feature_channels * 8 * 1, bias=False),
			nn.BatchNorm1d(num_features=self.graph_size * self.feature_channels * 8 * 1),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.convolutional_module_3 = nn.Sequential(
			_deconv1d_bn_lrelu(self.graph_size * self.feature_channels * 8, self.graph_size * self.feature_channels * 4, 3, 2, 1, 1),
			_deconv1d_bn_lrelu(self.graph_size * self.feature_channels * 4, self.graph_size * self.feature_channels * 2, 3, 2, 1, 1),
			nn.ConvTranspose1d(in_channels=self.graph_size * self.feature_channels * 2, out_channels=self.graph_size * self.feature_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		)
		self.convolutional_module_4 = nn.Sequential(
			_deconv1d_bn_lrelu(self.graph_size * self.feature_channels * 8, self.graph_size * self.feature_channels * 4, 4, 2, 1),
			_deconv1d_bn_lrelu(self.graph_size * self.feature_channels * 4, self.graph_size * self.feature_channels * 2, 4, 2, 1),
			nn.ConvTranspose1d(in_channels=self.graph_size * self.feature_channels * 2, out_channels=self.graph_size * self.feature_channels, kernel_size=4, stride=2, padding=1, bias=False)
		)
		self.convolutional_module_5 = nn.Sequential(
			_deconv1d_bn_lrelu(self.graph_size * self.feature_channels * 8, self.graph_size * self.feature_channels * 4, 5, 2, 2, 1),
			_deconv1d_bn_lrelu(self.graph_size * self.feature_channels * 4, self.graph_size * self.feature_channels * 2, 5, 2, 2, 1),
			nn.ConvTranspose1d(in_channels=self.graph_size * self.feature_channels * 2, out_channels=self.graph_size * self.feature_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
		)
		self.graph_convolutional_module = GATConv(in_channels=self.feature_channels, out_channels=self.feature_channels)
	def forward(self, noise, edge_index, edge_weight):
		linear_module_output = self.linear_module(noise)
		convolutional_module_input = linear_module_output.view(self.batch_size, self.graph_size * self.feature_channels * 8, 1)
		# convolutional_module_output = self.convolutional_module_3(convolutional_module_input)
		convolutional_module_output = self.convolutional_module_4(convolutional_module_input)
		# convolutional_module_output = self.convolutional_module_5(convolutional_module_input)
		graph_convolutional_module_input_x = convolutional_module_output.permute(0, 2, 1).reshape(self.batch_size * self.seq_len * self.graph_size, self.feature_channels)
		graph_convolutional_module_output = self.graph_convolutional_module(graph_convolutional_module_input_x, edge_index, edge_weight)
		activation_output = self.activation(graph_convolutional_module_output)
		fake = Data(x=activation_output, edge_index=edge_index, edge_weight=edge_weight)
		return fake


class LinearMaskGenerator(LinearGenerator):
	def __init__(self, noise_size, batch_size, seq_len, graph_size, feature_channels):
		super().__init__(noise_size, batch_size, seq_len, graph_size, feature_channels)
		_add_mask_activation(self)


class GraphConvolutionalMaskGenerator(GraphConvolutionalGenerator):
	def __init__(self, noise_size, batch_size, seq_len, graph_size, feature_channels):
		super().__init__(noise_size, batch_size, seq_len, graph_size, feature_channels)
		_add_mask_activation(self)


class LinearImputationGenerator(nn.Module):
	def __init__(self, batch_size, seq_len, graph_size, feature_channels):
		super(LinearImputationGenerator, self).__init__()
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.graph_size = graph_size
		self.feature_channels = feature_channels
		self.linear_module = nn.Sequential(
			nn.Linear(in_features=self.seq_len * self.graph_size * self.feature_channels, out_features=self.seq_len * self.graph_size * self.feature_channels),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=self.seq_len * self.graph_size * self.feature_channels, out_features=self.seq_len * self.graph_size * self.feature_channels),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=self.seq_len * self.graph_size * self.feature_channels, out_features=self.seq_len * self.graph_size * self.feature_channels),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(in_features=self.seq_len * self.graph_size * self.feature_channels, out_features=self.seq_len * self.graph_size * self.feature_channels),
			nn.Tanh()
		)
	def forward(self, data, mask, imputation_noise):
		linear_module_input = (data.x * mask.x + imputation_noise.x * (1 - mask.x)).view(self.batch_size, self.seq_len * self.graph_size * self.feature_channels)
		linear_module_output = self.linear_module(linear_module_input)
		imputation = Data(x=linear_module_output.view(self.batch_size * self.seq_len * self.graph_size, self.feature_channels), edge_index=data.edge_index)
		return imputation


class GraphConvolutionalImputationGenerator(nn.Module):
	def __init__(self, batch_size, seq_len, graph_size, feature_channels):
		super(GraphConvolutionalImputationGenerator, self).__init__()
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.graph_size = graph_size
		self.feature_channels = feature_channels
		self.graph_convolutional_encoder = GATConv(in_channels=self.feature_channels, out_channels=self.feature_channels)
		self.temporal_attention_module = nn.TransformerEncoderLayer(d_model=self.graph_size * self.feature_channels, nhead=3)
		self.convolutional_auto_encoder_module = Unet1d(self.graph_size * self.feature_channels, self.graph_size * self.feature_channels)
		self.graph_convolutional_decoder = GATConv(in_channels=self.feature_channels, out_channels=self.feature_channels)
		self.activation = nn.Tanh()
	def forward(self, data, mask, imputation_noise):
		graph_convolutional_encoder_input_x = data.x * mask.x + imputation_noise.x * (1 - mask.x)
		graph_convolutional_encoder_output = self.graph_convolutional_encoder(graph_convolutional_encoder_input_x, data.edge_index)
		temporal_attention_module_input = graph_convolutional_encoder_output.view(self.batch_size, self.seq_len, self.graph_size * self.feature_channels)
		temporal_attention_module_output = self.temporal_attention_module(temporal_attention_module_input)
		convolutional_auto_encoder_module_input = temporal_attention_module_output.transpose(2, 1)
		convolutional_auto_encoder_module_output = self.convolutional_auto_encoder_module(convolutional_auto_encoder_module_input)
		graph_convolutional_decoder_input_x = convolutional_auto_encoder_module_output.transpose(2, 1)
		graph_convolutional_decoder_output = self.graph_convolutional_decoder(graph_convolutional_decoder_input_x, data.edge_index, data.edge_weight)
		graph_convolutional_decoder_output = graph_convolutional_decoder_output.reshape(self.batch_size * self.seq_len * self.graph_size, self.feature_channels)
		activation_output = self.activation(graph_convolutional_decoder_output)
		imputation = Data(x=activation_output, edge_index=data.edge_index)
		return imputation