# -*- coding=utf-8 -*-


import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


# ============================== NetworkedTimeSeriesDataset.random_walk ==============================

def _depth_first_random_walk(self, num_missing_node):
	if self.random_walk_table == None:
		self.random_walk_table = [[] for _ in range(self.num_node)]
		for root_node in range(self.num_node):
			trajectory = np.array([False for _ in range(self.num_node)])
			current_node = root_node
			self.random_walk_table[root_node].append(current_node)
			trajectory[current_node] = True
			while True:
				current_node_probability = self.raw_adjacency[current_node] * (1 - trajectory)
				if current_node_probability.sum() > 0:
					next_node = current_node_probability.argmax()
					current_node = next_node
					self.random_walk_table[root_node].append(current_node)
					trajectory[current_node] = True
				else:
					break
	missing_index = np.array([False for _ in range(self.num_node)])
	num_selected_node = 0
	while num_selected_node < num_missing_node:
		current_node = np.random.choice(np.arange(self.num_node)[~missing_index])
		walk_length = min(len(self.random_walk_table[current_node]), num_missing_node - num_selected_node)
		missing_index[self.random_walk_table[current_node][:walk_length]] = True
		num_selected_node += walk_length
	return missing_index


def _breadth_first_random_walk(self, num_missing_node):
	if self.random_walk_table == None:
		self.random_walk_table = [[] for _ in range(self.num_node)]
		for root_node in range(self.num_node):
			trajectory = np.array([False for _ in range(self.num_node)])
			current_node = root_node
			current_index = 0
			self.random_walk_table[root_node].append(current_node)
			trajectory[current_node] = True
			while True:
				current_node_probability = self.raw_adjacency[current_node] * (1 - trajectory)
				if current_node_probability.sum() > 0:
					next_node = current_node_probability.argmax()
					self.random_walk_table[root_node].append(next_node)
					trajectory[next_node] = True
				elif current_index + 1 <= len(self.random_walk_table[root_node]) - 1:
					current_index += 1
					current_node = self.random_walk_table[root_node][current_index]
				else:
					break
	missing_index = np.array([False for _ in range(self.num_node)])
	num_selected_node = 0
	while num_selected_node < num_missing_node:
		current_node = np.random.choice(np.arange(self.num_node)[~missing_index])
		walk_length = min(len(self.random_walk_table[current_node]), num_missing_node - num_selected_node)
		missing_index[self.random_walk_table[current_node][:walk_length]] = True
		num_selected_node += walk_length
	return missing_index


# ============================== NetworkedTimeSeriesDataset.load_raw_data ==============================

def _load_train_raw_data(self):
	raw_data = np.load(self.raw_data_directory)
	num_train_timestamp = int(raw_data.shape[0] * self.train_proportion)
	raw_data = raw_data[:num_train_timestamp, :, :].reshape(num_train_timestamp, self.num_node * self.feature_channels)
	self.scaler = MinMaxScaler(feature_range=(-1, 1))
	self.scaler.fit(raw_data)
	if self.disruption_scale > 0:
		raw_data = torch.from_numpy(self.scaler.transform(raw_data)).float()
		raw_data += torch.normal(0, self.disruption_scale, raw_data.size())
		raw_data = self.scaler.inverse_transform(raw_data.numpy())
		self.scaler.fit(raw_data)
	raw_data = torch.from_numpy(self.scaler.transform(raw_data).reshape(num_train_timestamp, self.num_node, self.feature_channels)).float()
	return raw_data


def _load_validation_raw_data(self):
	raw_data = np.load(self.raw_data_directory)
	num_train_timestamp = int(raw_data.shape[0] * self.train_proportion)
	num_validation_timestamp = int(raw_data.shape[0] * self.validation_proportion)
	raw_data = raw_data[num_train_timestamp : num_train_timestamp + num_validation_timestamp, :, :].reshape(num_validation_timestamp, self.num_node * self.feature_channels)
	self.scaler = MinMaxScaler(feature_range=(-1, 1))
	self.scaler.fit(raw_data)
	if self.disruption_scale > 0:
		raw_data = torch.from_numpy(self.scaler.transform(raw_data)).float()
		raw_data += torch.normal(0, self.disruption_scale, raw_data.size())
		raw_data = self.scaler.inverse_transform(raw_data.numpy())
		self.scaler.fit(raw_data)
	raw_data = torch.from_numpy(self.scaler.transform(raw_data).reshape(num_validation_timestamp, self.num_node, self.feature_channels)).float()
	return raw_data


def _load_test_raw_data(self):
	raw_data = np.load(self.raw_data_directory)
	num_train_timestamp = int(raw_data.shape[0] * self.train_proportion)
	num_validation_timestamp = int(raw_data.shape[0] * self.validation_proportion)
	num_test_timestamp = raw_data.shape[0] - num_train_timestamp - num_validation_timestamp
	raw_data = raw_data[-num_test_timestamp:, :, :].reshape(num_test_timestamp, self.num_node * self.feature_channels)
	self.scaler = MinMaxScaler(feature_range=(-1, 1))
	self.scaler.fit(raw_data)
	if self.disruption_scale > 0:
		raw_data = torch.from_numpy(self.scaler.transform(raw_data)).float()
		raw_data += torch.normal(0, self.disruption_scale, raw_data.size())
		raw_data = self.scaler.inverse_transform(raw_data.numpy())
		self.scaler.fit(raw_data)
	raw_data = torch.from_numpy(self.scaler.transform(raw_data).reshape(num_test_timestamp, self.num_node, self.feature_channels)).float()
	return raw_data


# ============================== NetworkedTimeSeriesDataset.build_data ==============================

def _build_imputation_data(self, raw_data):
	raw_data = [Data(x=raw_data[t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in range(raw_data.size(0))]
	self.data = []
	temp_batch = Batch()
	for t in range(raw_data.size(0) - self.num_timestamp + 1):
		temp_batch = temp_batch.from_data_list(raw_data[t : t + self.num_timestamp])
		data_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
		self.data.append(data_sample)


def _build_prediction_data(self, raw_data):
	raw_data = [Data(x=raw_data[t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in range(raw_data.size(0))]
	self.data = []
	temp_batch = Batch()
	for t in range(raw_data.size(0) - self.num_timestamp * 2 + 1):
		temp_batch = temp_batch.from_data_list(raw_data[t : t + self.num_timestamp * 2])
		data_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
		self.data.append(data_sample)


# ============================== NetworkedTimeSeriesDataset.generate_raw_mask_sample ==============================

def _generate_random_raw_mask_sample(self):
	raw_mask_sample = torch.rand(self.num_timestamp, self.num_node, self.feature_channels)
	raw_mask_sample[raw_mask_sample < self.missing_rate] = 0
	raw_mask_sample[raw_mask_sample >= self.missing_rate] = 1
	return raw_mask_sample


def _generate_sg_block_raw_mask_sample(self):
	raw_mask_sample = torch.ones(self.num_timestamp, self.num_node, self.feature_channels)
	num_missing_node = int(self.num_node * self.missing_rate)
	missing_node_index = self.random_walk(self, num_missing_node)
	raw_mask_sample[:, missing_node_index, :] = 0
	return raw_mask_sample


def _generate_st_block_raw_mask_sample(self):
	raw_mask_sample = torch.ones(self.num_timestamp, self.num_node, self.feature_channels)
	num_missing_timestamp = int(self.num_timestamp * self.missing_rate)
	start_index = torch.randint(0, self.num_timestamp - num_missing_timestamp + 1, (1,)).item()
	raw_mask_sample[start_index : start_index + num_missing_timestamp, :, :] = 0
	return raw_mask_sample


def _generate_sf_block_raw_mask_sample(self):
	raw_mask_sample = torch.ones(self.num_timestamp, self.num_node, self.feature_channels)
	num_missing_node = int(self.num_node * (self.missing_rate ** 0.5))
	missing_node_index = self.random_walk(self, num_missing_node)
	num_missing_timestamp = int(self.num_timestamp * (self.missing_rate ** 0.5))
	start_index = torch.randint(0, self.num_timestamp - num_missing_timestamp + 1, (1,)).item()
	raw_mask_sample[start_index : start_index + num_missing_timestamp, missing_node_index, :] = 0
	return raw_mask_sample


def _generate_sv_block_raw_mask_sample(self):
	raw_mask_sample = torch.ones(self.num_timestamp, self.num_node, self.feature_channels)
	missing_node_proportion = torch.rand(1,).item() * 0.5 + 0.25
	num_missing_node = int(self.num_node * missing_node_proportion)
	missing_node_index = self.random_walk(self, num_missing_node)
	missing_timestamp_proportion = torch.rand(1,).item() * 0.5 + 0.25
	num_missing_timestamp = int(self.num_timestamp * missing_timestamp_proportion)
	start_index = torch.randint(0, self.num_timestamp - num_missing_timestamp + 1, (1,)).item()
	raw_mask_sample[start_index : start_index + num_missing_timestamp, missing_node_index, :] = 0
	return raw_mask_sample


def _generate_mv_block_raw_mask_sample(self, min_node=1, max_node=8, min_timestamp=1, max_timestamp=4):
	raw_mask_sample = torch.ones(self.num_timestamp, self.num_node, self.feature_channels)
	average_block_size = (min_node + max_node - 1) * (min_timestamp + max_timestamp - 1) / 4
	for i in range(int(self.num_timestamp * self.num_node * self.feature_channels * self.missing_rate / average_block_size)):
		num_missing_node = torch.randint(min_node, max_node, (1,)).item()
		missing_node_index = self.random_walk(self, num_missing_node)
		num_missing_timestamp = torch.randint(min_timestamp, max_timestamp, (1,)).item()
		start_index = torch.randint(0, self.num_timestamp - num_missing_timestamp + 1, (1,)).item()
		raw_mask_sample[start_index : start_index + num_missing_timestamp, missing_node_index, :] = 0
	return raw_mask_sample


# ============================== NetworkedTimeSeriesDataset.generate_raw_mask ==============================

def _generate_imputation_raw_mask(self):
	raw_mask = torch.stack(tuple([self.generate_raw_mask_sample(self) for _ in range(len(self))]), dim=0)
	return raw_mask


def _generate_prediction_raw_mask(self):
	raw_mask = torch.stack(tuple([torch.cat((self.generate_raw_mask_sample(self), self.generate_raw_mask_sample(self)), dim=0) for _ in range(len(self))]), dim=0)
	return raw_mask


# ============================== NetworkedTimeSeriesDataset.build_mask ==============================

def _build_imputation_mask(self, raw_mask):
	self.mask, self.mask_star = [], []
	temp_batch = Batch()
	for i in range(len(self)):
		raw_mask_sample = [Data(x=raw_mask[i][t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in range(self.num_timestamp)]
		temp_batch = temp_batch.from_data_list(raw_mask_sample)
		mask_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
		self.mask.append(mask_sample)
		self.mask_star.append(mask_sample)


def _build_prediction_mask(self, raw_mask):
	self.mask, self.mask_star = [], []
	temp_batch = Batch()
	for i in range(len(self)):
		raw_mask_sample = [Data(x=raw_mask[i][t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in range(self.num_timestamp * 2)]
		temp_batch = temp_batch.from_data_list(raw_mask_sample)
		mask_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
		self.mask.append(mask_sample)
		raw_mask_star_sample = [Data(x=raw_mask[i][t], edge_index=self.edge_index, edge_weight=self.edge_weight) for t in range(self.num_timestamp)]
		raw_mask_star_sample += [Data(x=torch.zeros(raw_mask[i][t].size()), edge_index=self.edge_index, edge_weight=self.edge_weight) for t in range(self.num_timestamp, self.num_timestamp * 2)]
		temp_batch = temp_batch.from_data_list(raw_mask_star_sample)
		mask_star_sample = Data(x=temp_batch.x, edge_index=temp_batch.edge_index, edge_weight=temp_batch.edge_weight)
		self.mask_star.append(mask_star_sample)


# ============================== NetworkedTimeSeriesDataset.__getitem__ ==============================

def _get_imputation_item(self, index):
	data_sample = self.data[index]
	mask_sample = self.mask[index]
	mask_star_sample = self.mask_star[index]
	return data_sample, mask_sample, mask_star_sample


def _get_prediction_item(self, index):
	data_sample = self.data[index]
	mask_sample = self.mask[index]
	mask_star_sample = self.mask_star[index]
	return data_sample, mask_sample, mask_star_sample


# ============================== NetworkedTimeSeriesDataset.__len__ ==============================

def _imputation_length(self):
	return len(self.data)


def _prediction_length(self):
	return len(self.data)


# ============================== NetworkedTimeSeriesDataset ==============================

class NetworkedTimeSeriesDataset(Dataset):

	def __init__(self, raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels):

		self.raw_data_directory = raw_data_directory
		self.raw_adjacency_directory = raw_adjacency_directory
		self.train_proportion = train_proportion
		self.validation_proportion = validation_proportion
		self.disruption_scale = disruption_scale
		self.missing_pattern = missing_pattern
		self.missing_rate = missing_rate
		self.num_timestamp = num_timestamp
		self.num_node = num_node
		self.feature_channels = feature_channels

		self.random_walk_table = None
		self.random_walk = _breadth_first_random_walk

		self.load_raw_adjacency()
		self.build_adjacency()

		raw_data = self.load_raw_data(self)
		self.build_data(self, raw_data)

		if self.missing_pattern == 'Random':
			self.generate_raw_mask_sample = _generate_random_raw_mask_sample
		elif self.missing_pattern == 'SG-Block':
			self.generate_raw_mask_sample = _generate_sg_block_raw_mask_sample
		elif self.missing_pattern == 'ST-Block':
			self.generate_raw_mask_sample = _generate_st_block_raw_mask_sample
		elif self.missing_pattern == 'SF-Block':
			self.generate_raw_mask_sample = _generate_sf_block_raw_mask_sample
		elif self.missing_pattern == 'SV-Block':
			self.generate_raw_mask_sample = _generate_sv_block_raw_mask_sample
		elif self.missing_pattern == 'MV-Block':
			self.generate_raw_mask_sample = _generate_mv_block_raw_mask_sample
		else:
			raise NotImplementedError
		raw_mask = self.generate_raw_mask(self)
		self.build_mask(self, raw_mask)

	def load_raw_adjacency(self):
		self.raw_adjacency = np.load(self.raw_adjacency_directory)

	def build_adjacency(self):
		start_node, end_node = [], []
		self.edge_weight = []
		for i in range(self.num_node):
			for j in range(self.num_node):
				if self.raw_adjacency[i][j] > 0:
					start_node.append(i)
					end_node.append(j)
					self.edge_weight.append(self.raw_adjacency[i][j])
		self.edge_index = torch.tensor([start_node, end_node], dtype=torch.int64)
		self.edge_weight = torch.tensor(self.edge_weight).float()

	def load_raw_data(self):
		raise NotImplementedError
	
	def build_data(self, raw_data):
		raise NotImplementedError

	def generate_raw_mask(self):
		raise NotImplementedError
	
	def build_mask(self, raw_mask):
		raise NotImplementedError

	def __getitem__(self, index):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError


class ImputationTrainSet(NetworkedTimeSeriesDataset):
	def __init__(self, raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels):
		self.load_raw_data = _load_train_raw_data
		self.build_data = _build_imputation_data
		self.generate_raw_mask = _generate_imputation_raw_mask
		self.build_mask = _build_imputation_mask
		super().__init__(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	__getitem__ = _get_imputation_item
	__len__ = _imputation_length


class ImputationValidationSet(NetworkedTimeSeriesDataset):
	def __init__(self, raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels):
		self.load_raw_data = _load_validation_raw_data
		self.build_data = _build_imputation_data
		self.generate_raw_mask = _generate_imputation_raw_mask
		self.build_mask = _build_imputation_mask
		super().__init__(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	__getitem__ = _get_imputation_item
	__len__ = _imputation_length


class ImputationTestSet(NetworkedTimeSeriesDataset):
	def __init__(self, raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels):
		self.load_raw_data = _load_test_raw_data
		self.generate_raw_mask = _generate_imputation_raw_mask
		self.build_data = _build_imputation_data
		self.build_mask = _build_imputation_mask
		super().__init__(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	__getitem__ = _get_imputation_item
	__len__ = _imputation_length


class PredictionTrainSet(NetworkedTimeSeriesDataset):
	def __init__(self, raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels):
		self.load_raw_data = _load_train_raw_data
		self.generate_raw_mask = _generate_prediction_raw_mask
		self.build_data = _build_prediction_data
		self.build_mask = _build_prediction_mask
		super().__init__(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	__getitem__ = _get_prediction_item
	__len__ = _prediction_length


class PredictionValidationSet(NetworkedTimeSeriesDataset):
	def __init__(self, raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels):
		self.load_raw_data = _load_validation_raw_data
		self.generate_raw_mask = _generate_prediction_raw_mask
		self.build_data = _build_prediction_data
		self.build_mask = _build_prediction_mask
		super().__init__(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	__getitem__ = _get_prediction_item
	__len__ = _prediction_length


class PredictionTestSet(NetworkedTimeSeriesDataset):
	def __init__(self, raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels):
		self.load_raw_data = _load_test_raw_data
		self.generate_raw_mask = _generate_prediction_raw_mask
		self.build_data = _build_prediction_data
		self.build_mask = _build_prediction_mask
		super().__init__(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	__getitem__ = _get_prediction_item
	__len__ = _prediction_length


def breadth_first_random_walk(random_walk_table, num_missing_node):
	missing_index = np.array([False for _ in range(81)])
	num_selected_node = 0
	while num_selected_node < num_missing_node:
		current_node = np.random.choice(np.arange(81)[~missing_index])
		walk_length = min(len(random_walk_table[current_node]), num_missing_node - num_selected_node)
		missing_index[random_walk_table[current_node][:walk_length]] = True
		num_selected_node += walk_length
	return missing_index


def generate_mv_block_raw_mask_sample(random_walk_table, min_node=1, max_node=8, min_timestamp=1, max_timestamp=4):
	raw_mask_sample = torch.ones(16, 81, 1)
	average_block_size = (min_node + max_node - 1) * (min_timestamp + max_timestamp - 1) / 4
	for i in range(int(16 * 81 * 1 * 0.25 / average_block_size)):
		num_missing_node = torch.randint(min_node, max_node, (1,)).item()
		missing_node_index = breadth_first_random_walk(random_walk_table, num_missing_node)
		num_missing_timestamp = torch.randint(min_timestamp, max_timestamp, (1,)).item()
		start_index = torch.randint(0, 16 - num_missing_timestamp + 1, (1,)).item()
		raw_mask_sample[start_index : start_index + num_missing_timestamp, missing_node_index, :] = 0
	return raw_mask_sample