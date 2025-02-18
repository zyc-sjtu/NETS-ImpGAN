# -*- coding=utf-8 -*-


import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader
import torch.optim as optim
from dataset import PredictionTrainSet, PredictionValidationSet, PredictionTestSet
from generator import LinearMaskGenerator, GraphConvolutionalMaskGenerator, LinearImputationGenerator, GraphConvolutionalImputationGenerator
from discriminator import LinearDiscriminator, GraphConvolutionalDiscriminator
from updater import MaskGeneratorUpdater, ImputationGeneratorUpdater, DiscriminatorUpdater
from metric import compute_mean_absolute_error
from utils import load_config, save_config, load_model, save_model, mask


config = load_config()
raw_data_directory = config['data']['raw_data_directory']
raw_adjacency_directory = config['data']['raw_adjacency_directory']
train_proportion = config['data']['train_proportion']
validation_proportion = config['data']['validation_proportion']
disruption_scale = config['data']['disruption_scale']
missing_pattern = config['data']['missing_pattern']
missing_rate = config['data']['missing_rate']
num_timestamp = config['data']['num_timestamp']
num_node = config['data']['num_node']
feature_channels = config['data']['feature_channels']
mask_generator_architecture = config['model']['mask_generator_architecture']
mask_discriminator_architecture = config['model']['mask_discriminator_architecture']
imputation_generator_architecture = config['model']['imputation_generator_architecture']
imputation_discriminator_architecture = config['model']['imputation_discriminator_architecture']
mask_noise_size = config['model']['mask_noise_size']
model_directory = config['model']['model_directory']
cuda_device = config['optimization']['cuda_device']
batch_size = config['optimization']['batch_size']
mask_learning_rate = config['optimization']['mask_learning_rate']
imputation_learning_rate = config['optimization']['imputation_learning_rate']
num_epoch = config['optimization']['num_epoch']
epoch_generator = config['optimization']['epoch_generator']
epoch_train = config['optimization']['epoch_train']
epoch_validation = config['optimization']['epoch_validation']
epoch_test = config['optimization']['epoch_test']
tau = config['optimization']['tau']
alpha = config['optimization']['alpha']
beta = config['optimization']['beta']
gamma = config['optimization']['gamma']
eta = config['optimization']['eta']

save_config(config, model_directory)

device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')


def train():

	train_set = PredictionTrainSet(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	validation_set = PredictionValidationSet(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	test_set = PredictionTestSet(raw_data_directory, raw_adjacency_directory, train_proportion, validation_proportion, disruption_scale, missing_pattern, missing_rate, num_timestamp, num_node, feature_channels)
	train_set_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
	validation_set_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, drop_last=True)
	test_set_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

	mask_noise = torch.FloatTensor(batch_size, mask_noise_size).to(device)
	imputation_noise_x = torch.FloatTensor(batch_size * num_timestamp * num_node, feature_channels)

	theta = torch.FloatTensor(batch_size, 1, 1, 1).to(device)
	grad_outputs = torch.ones(batch_size, 1).to(device)

	mask_generator = GraphConvolutionalMaskGenerator(mask_noise_size, batch_size, num_timestamp, num_node, feature_channels).to(device)
	mask_discriminator = GraphConvolutionalDiscriminator(batch_size, num_timestamp, num_node, feature_channels).to(device)
	imputation_generator = GraphConvolutionalImputationGenerator(batch_size, num_timestamp, num_node, feature_channels).to(device)
	imputation_discriminator = GraphConvolutionalDiscriminator(batch_size, num_timestamp, num_node, feature_channels).to(device)

	num_param = 0
	for param in mask_generator.parameters():
		if param.requires_grad:
			num_param += np.prod(param.size())
	print (num_param)
	num_param = 0
	for param in mask_discriminator.parameters():
		if param.requires_grad:
			num_param += np.prod(param.size())
	print (num_param)
	num_param = 0
	for param in imputation_generator.parameters():
		if param.requires_grad:
			num_param += np.prod(param.size())
	print (num_param)
	num_param = 0
	for param in imputation_discriminator.parameters():
		if param.requires_grad:
			num_param += np.prod(param.size())
	print (num_param)

	mask_generator_optimizer = optim.Adam(params=mask_generator.parameters(), lr=mask_learning_rate, betas=(.5, .9))
	mask_discriminator_optimizer = optim.Adam(params=mask_discriminator.parameters(), lr=mask_learning_rate, betas=(.5, .9))
	imputation_generator_optimizer = optim.Adam(params=imputation_generator.parameters(), lr=imputation_learning_rate, betas=(.5, .9))
	imputation_discriminator_optimizer = optim.Adam(params=imputation_discriminator.parameters(), lr=imputation_learning_rate, betas=(.5, .9))

	mask_generator_updater = MaskGeneratorUpdater(mask_discriminator, mask_generator_optimizer)
	mask_discriminator_updater = DiscriminatorUpdater(mask_discriminator, mask_discriminator_optimizer, batch_size, num_timestamp, num_node, feature_channels, theta, grad_outputs, eta)
	imputation_generator_updater = ImputationGeneratorUpdater(imputation_discriminator, imputation_generator_optimizer)
	imputation_discriminator_updater = DiscriminatorUpdater(imputation_discriminator, imputation_discriminator_optimizer, batch_size, num_timestamp, num_node, feature_channels, theta, grad_outputs, eta)

	mask_generator_loss_log = []
	mask_discriminator_loss_log = []
	imputation_generator_loss_log = []
	imputation_discriminator_loss_log = []
	train_mean_absolute_error_log = []
	validation_mean_absolute_error_log = []
	test_mean_absolute_error_log = []

	for epoch in range(num_epoch):

		print ('Epoch ' + str(epoch + 1))

		epoch_imputation_generator_loss = 0
		epoch_imputation_discriminator_loss = 0

		for real_data, real_mask, real_mask_star in train_set_loader:

			real_data = real_data.to(device)
			real_mask = real_mask.to(device)
			real_mask_star = real_mask_star.to(device)

			for p in mask_generator.parameters():
				p.requires_grad_(False)
			for p in imputation_generator.parameters():
				p.requires_grad_(False)

			mask_noise.normal_()
			fake_mask = mask_generator(mask_noise, real_mask.edge_index, real_mask.edge_weight)

			imputation_noise_x.uniform_()
			imputation_noise = Data(x=imputation_noise_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight).to(device)
			imputation = imputation_generator(real_data, real_mask_star, imputation_noise)
			imputed_data_x = mask(real_data.x, real_mask_star.x, imputation.x)
			imputed_data = Data(x=imputed_data_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight)

			real_masked_data_x = mask(real_data.x, real_mask.x, tau)
			real_masked_data = Data(x=real_masked_data_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight)
			imputed_masked_data_x = mask(imputed_data.x, fake_mask.x, tau)
			imputed_masked_data = Data(x=imputed_masked_data_x, edge_index=imputed_data.edge_index, edge_weight=imputed_data.edge_weight)

			mask_discriminator_loss = mask_discriminator_updater(real_mask, fake_mask)
			imputation_discriminator_loss = imputation_discriminator_updater(real_masked_data, imputed_masked_data)

			epoch_mask_discriminator_loss += mask_discriminator_loss
			epoch_imputation_discriminator_loss += imputation_discriminator_loss

			for p in mask_generator.parameters():
				p.requires_grad_(True)
			for p in imputation_generator.parameters():
				p.requires_grad_(True)

			if (epoch + 1) % epoch_generator == 0:

				for p in mask_discriminator.parameters():
					p.requires_grad_(False)
				for p in imputation_discriminator.parameters():
					p.requires_grad_(False)

				mask_noise.normal_()
				fake_mask = mask_generator(mask_noise, real_mask.edge_index, real_mask.edge_weight)

				imputation_noise_x.uniform_()
				imputation_noise = Data(x=imputation_noise_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight).to(device)
				imputation = imputation_generator(real_data, real_mask_star, imputation_noise)
				imputed_data_x = mask(real_data.x, real_mask_star.x, imputation.x)
				imputed_data = Data(x=imputed_data_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight)

				imputed_masked_data_x = mask(imputed_data.x, fake_mask.x, tau)
				imputed_masked_data = Data(x=imputed_masked_data_x, edge_index=imputed_data.edge_index, edge_weight=imputed_data.edge_weight)

				mask_generator_loss = mask_generator_updater(fake_mask)
				imputation_generator_loss = imputation_generator_updater(imputed_masked_data, real_data, real_mask_star, imputation)

				epoch_mask_generator_loss += mask_generator_loss
				epoch_imputation_generator_loss += imputation_generator_loss

				for p in mask_discriminator.parameters():
					p.requires_grad_(True)
				for p in imputation_discriminator.parameters():
					p.requires_grad_(True)

		mask_generator_loss_log.append(epoch_mask_generator_loss)
		mask_discriminator_loss_log.append(epoch_mask_discriminator_loss)
		imputation_generator_loss_log.append(epoch_imputation_generator_loss)
		imputation_discriminator_loss_log.append(epoch_imputation_discriminator_loss)
		print ('Mask Generator Loss:' + str(epoch_mask_generator_loss))
		print ('Mask Discriminator Loss: ' + str(epoch_mask_discriminator_loss))
		print ('Imputation Generator Loss:' + str(epoch_imputation_generator_loss))
		print ('Imputation Discriminator Loss: ' + str(epoch_imputation_discriminator_loss))

		sum_mean_absolute_error = 0
		if (epoch + 1) % epoch_train == 0:
			with torch.no_grad():
				imputation_generator.eval()
				for real_data, real_mask, real_mask_star in train_set_loader:
					real_data = real_data.to(device)
					real_mask = real_mask.to(device)
					real_mask_star = real_mask_star.to(device)
					imputation_noise_x.uniform_()
					imputation_noise = Data(x=imputation_noise_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight).to(device)
					imputation = imputation_generator(real_data, real_mask_star, imputation_noise)
					imputed_data_x = mask(real_data.x, real_mask_star.x, imputation.x)
					imputed_data = Data(x=imputed_data_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight)
					real_data_x = torch.tensor(train_set.scaler.inverse_transform(real_data.x.view(batch_size * num_timestamp * 2, num_node * feature_channels).detach().cpu().numpy()))
					imputed_data_x = torch.tensor(train_set.scaler.inverse_transform(imputed_data.x.view(batch_size * num_timestamp * 2, num_node * feature_channels).detach().cpu().numpy()))
					real_future_x = real_data_x.view(batch_size, num_timestamp * 2, num_node, feature_channels)[:, -num_timestamp:, :, :]
					imputed_future_x = imputed_data_x.view(batch_size, num_timestamp * 2, num_node, feature_channels)[:, -num_timestamp:, :, :]
					sum_mean_absolute_error += compute_mean_absolute_error(real_future_x, imputed_future_x)
				sum_mean_absolute_error /= len(train_set_loader)
				imputation_generator.train()
		train_mean_absolute_error_log.append(sum_mean_absolute_error)
		print ('Train Mean Absolute Error: ' + str(sum_mean_absolute_error))

		sum_mean_absolute_error = 0
		if (epoch + 1) % epoch_validation == 0:
			with torch.no_grad():
				imputation_generator.eval()
				for real_data, real_mask, real_mask_star in validation_set_loader:
					real_data = real_data.to(device)
					real_mask = real_mask.to(device)
					real_mask_star = real_mask_star.to(device)
					imputation_noise_x.uniform_()
					imputation_noise = Data(x=imputation_noise_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight).to(device)
					imputation = imputation_generator(real_data, real_mask_star, imputation_noise)
					imputed_data_x = mask(real_data.x, real_mask_star.x, imputation.x)
					imputed_data = Data(x=imputed_data_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight)
					real_data_x = torch.tensor(validation_set.scaler.inverse_transform(real_data.x.view(batch_size * num_timestamp * 2, num_node * feature_channels).detach().cpu().numpy()))
					imputed_data_x = torch.tensor(validation_set.scaler.inverse_transform(imputed_data.x.view(batch_size * num_timestamp * 2, num_node * feature_channels).detach().cpu().numpy()))
					real_future_x = real_data_x.view(batch_size, num_timestamp * 2, num_node, feature_channels)[:, -num_timestamp:, :, :]
					imputed_future_x = imputed_data_x.view(batch_size, num_timestamp * 2, num_node, feature_channels)[:, -num_timestamp:, :, :]
					sum_mean_absolute_error += compute_mean_absolute_error(real_future_x, imputed_future_x)
				sum_mean_absolute_error /= len(validation_set_loader)
				imputation_generator.train()
		validation_mean_absolute_error_log.append(sum_mean_absolute_error)
		print ('Validation Mean Absolute Error: ' + str(sum_mean_absolute_error))

		sum_mean_absolute_error = 0
		if (epoch + 1) % epoch_test == 0:
			with torch.no_grad():
				imputation_generator.eval()
				for real_data, real_mask, real_mask_star in test_set_loader:
					real_data = real_data.to(device)
					real_mask = real_mask.to(device)
					real_mask_star = real_mask_star.to(device)
					imputation_noise_x.uniform_()
					imputation_noise = Data(x=imputation_noise_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight).to(device)
					imputation = imputation_generator(real_data, real_mask_star, imputation_noise)
					imputed_data_x = mask(real_data.x, real_mask_star.x, imputation.x)
					imputed_data = Data(x=imputed_data_x, edge_index=real_data.edge_index, edge_weight=real_data.edge_weight)
					real_data_x = torch.tensor(test_set.scaler.inverse_transform(real_data.x.view(batch_size * num_timestamp * 2, num_node * feature_channels).detach().cpu().numpy()))
					imputed_data_x = torch.tensor(test_set.scaler.inverse_transform(imputed_data.x.view(batch_size * num_timestamp * 2, num_node * feature_channels).detach().cpu().numpy()))
					real_future_x = real_data_x.view(batch_size, num_timestamp * 2, num_node, feature_channels)[:, -num_timestamp:, :, :]
					imputed_future_x = imputed_data_x.view(batch_size, num_timestamp * 2, num_node, feature_channels)[:, -num_timestamp:, :, :]
					sum_mean_absolute_error += compute_mean_absolute_error(real_future_x, imputed_future_x)
				sum_mean_absolute_error /= len(test_set_loader)
				imputation_generator.train()
		test_mean_absolute_error_log.append(sum_mean_absolute_error)
		print ('Test Mean Absolute Error: ' + str(sum_mean_absolute_error))

		if (epoch + 1) % epoch_generator == 0:
			save_model({
				'mask_generator': mask_generator.state_dict(),
				'mask_discriminator': mask_discriminator.state_dict(),
				'imputation_generator': imputation_generator.state_dict(),
				'imputation_discriminator': imputation_discriminator.state_dict(),
				'mask_generator_optimizer': mask_generator_optimizer.state_dict(),
				'mask_discriminator_optimizer': mask_discriminator_optimizer.state_dict(),
				'imputation_generator_optimizer': imputation_generator_optimizer.state_dict(),
				'imputation_discriminator_optimizer': imputation_discriminator_optimizer.state_dict(),
				'mask_generator_loss_log': np.array(mask_generator_loss_log),
				'mask_discriminator_loss_log': np.array(mask_discriminator_loss_log),
				'imputation_generator_loss_log': np.array(imputation_generator_loss_log),
				'imputation_discriminator_loss_log': np.array(imputation_discriminator_loss_log),
				'train_mean_absolute_error_log': np.array(train_mean_absolute_error_log),
				'validation_mean_absolute_error_log': np.array(validation_mean_absolute_error_log),
				'test_mean_absolute_error_log': np.array(test_mean_absolute_error_log)
			}, model_directory, 'model_' + str(int((epoch + 1) / epoch_generator)) + '.pth')


if __name__ == '__main__':
	train()