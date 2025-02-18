# -*- coding=utf-8 -*-


import argparse
import yaml
import torch
import matplotlib.pyplot as plt


def load_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('config_dir')
	args = parser.parse_args()
	config = yaml.safe_load(open(args.config_dir, 'r'))
	return config


def save_config(config, path):
	yaml.dump(config, open(path + 'config.yaml', 'w'), sort_keys=False)


def load_model(model_dir, model_name):
	model = torch.load(model_dir + model_name, map_location='cpu')
	return model


def save_model(model_dict, model_dir, model_name):
	torch.save(model_dict, model_dir + model_name)


def mask(data, mask, imputation):
	masked_data = data * mask + imputation * (1 - mask)
	return masked_data


def save_model_mask_data(mask_generator, mask_discriminator, data_generator, data_discriminator,
	mask_generator_optimizer, mask_discriminator_optimizer, data_generator_optimizer, data_discriminator_optimizer,
	mask_generator_loss_log, mask_discriminator_loss_log, data_generator_loss_log, data_discriminator_loss_log,
	missing_rate_log, model_dir, model_name):
	torch.save({
		'mask_generator': mask_generator.state_dict(),
		'mask_discriminator': mask_discriminator.state_dict(),
		'data_generator': data_generator.state_dict(),
		'data_discriminator': data_discriminator.state_dict(),
		'mask_generator_optimizer': mask_generator_optimizer.state_dict(),
		'mask_discriminator_optimizer': mask_discriminator_optimizer.state_dict(),
		'data_generator_optimizer': data_generator_optimizer.state_dict(),
		'data_discriminator_optimizer': data_discriminator_optimizer.state_dict(),
		'mask_generator_loss_log': mask_generator_loss_log,
		'mask_discriminator_loss_log': mask_discriminator_loss_log,
		'data_generator_loss_log': data_generator_loss_log,
		'data_discriminator_loss_log': data_discriminator_loss_log,
		'missing_rate_log': missing_rate_log
		}, model_dir + model_name)


def plot_mask(mask, plot_dir):
	for t in range(mask.shape[1]):
		plt.imshow(mask[0, t, :, :] * 255, cmap='gist_gray', vmin=0, vmax=255)
		plt.xticks([])
		plt.yticks([])
		plt.savefig(plot_dir + '_' + str(t+1) + '.pdf', bbox_inches='tight')
		plt.close('all')