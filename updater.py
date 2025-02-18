# -*- coding=utf-8 -*-


from torch_geometric.data import Data
from torch.autograd import grad


class MaskGeneratorUpdater:
	def __init__(self, mask_discriminator, mask_generator_optimizer, data_discriminator=None, alpha=None):
		self.mask_discriminator = mask_discriminator
		self.data_discriminator = data_discriminator
		self.mask_generator_optimizer = mask_generator_optimizer
		self.alpha = alpha
	def __call__(self, fake_mask, fake_masked_data=None):
		self.mask_generator_optimizer.zero_grad()
		mask_generator_loss = -self.mask_discriminator(fake_mask).mean()
		if self.alpha:
			mask_generator_loss += -self.alpha * self.data_discriminator(fake_masked_data).mean()
		mask_generator_loss.backward(retain_graph=True)
		self.mask_generator_optimizer.step()
		return mask_generator_loss.item()


class DataGeneratorUpdater:
	def __init__(self, data_discriminator, data_generator_optimizer, imputation_discriminator=None, beta=None):
		self.data_discriminator = data_discriminator
		self.data_generator_optimizer = data_generator_optimizer
		self.imputation_discriminator = imputation_discriminator
		self.beta = beta
	def __call__(self, fake_masked_data, fake_data=None):
		self.data_generator_optimizer.zero_grad()
		data_generator_loss = -self.data_discriminator(fake_masked_data).mean()
		if self.beta:
			data_generator_loss += self.beta * self.imputation_discriminator(fake_data).mean()
		data_generator_loss.backward(retain_graph=True)
		self.data_generator_optimizer.step()
		return data_generator_loss.item()


class ImputationGeneratorUpdater:
	def __init__(self, imputation_discriminator, imputation_generator_optimizer, gamma=None):
		self.imputation_discriminator = imputation_discriminator
		self.imputation_generator_optimizer = imputation_generator_optimizer
		self.gamma = gamma
	def __call__(self, imputed_data, real_data=None, real_mask=None, imputation=None):
		self.imputation_generator_optimizer.zero_grad()
		imputation_generator_loss = -self.imputation_discriminator(imputed_data).mean()
		if self.gamma:
			reconstruction_loss = ((imputation.x - real_data.x) * real_mask.x).abs_().sum() / real_mask.x.sum()
			imputation_generator_loss += self.gamma * reconstruction_loss
		imputation_generator_loss.backward(retain_graph=True)
		self.imputation_generator_optimizer.step()
		return imputation_generator_loss.item()


class DiscriminatorUpdater:
	def __init__(self, discriminator, discriminator_optimizer, batch_size, seq_len, graph_size, feature_channels, theta, grad_outputs, eta):
		self.discriminator = discriminator
		self.discriminator_optimizer = discriminator_optimizer
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.graph_size = graph_size
		self.feature_channels = feature_channels
		self.theta = theta
		self.grad_outputs = grad_outputs
		self.eta = eta
	def __call__(self, real, fake):
		self.discriminator_optimizer.zero_grad()
		wasserstein_distance = self.discriminator(fake).mean() - self.discriminator(real).mean()
		self.theta.uniform_()
		real_x = real.x.view(self.batch_size, self.seq_len, self.graph_size, self.feature_channels)
		fake_x = fake.x.view(self.batch_size, self.seq_len, self.graph_size, self.feature_channels)
		interpolation = (self.theta * real_x + (1 - self.theta) * fake_x).requires_grad_()
		interpolation = interpolation.view(self.batch_size * self.seq_len * self.graph_size, self.feature_channels)
		interpolation = Data(x=interpolation, edge_index=real.edge_index, edge_weight=real.edge_weight)
		gradient = grad(self.discriminator(interpolation), interpolation.x, grad_outputs=self.grad_outputs, create_graph=True)[0]
		gradient = gradient.view(self.batch_size, -1)
		gradient_penalty = ((gradient.norm(dim=1) - 1) ** 2).mean()
		discriminator_loss = wasserstein_distance + self.eta * gradient_penalty
		discriminator_loss.backward(retain_graph=True)
		self.discriminator_optimizer.step()
		return discriminator_loss.item()