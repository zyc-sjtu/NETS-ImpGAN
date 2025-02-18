# -*- coding=utf-8 -*-


import torch
import torch.nn as nn


class UnetSkipConnectionBlock1d(nn.Module):
	def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm1d):
		super(UnetSkipConnectionBlock1d, self).__init__()
		self.outermost = outermost
		use_bias = norm_layer == nn.InstanceNorm1d
		if input_nc is None:
			input_nc = outer_nc
		#downconv = nn.Conv1d(input_nc, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
		downconv = nn.Conv1d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
		#downconv = nn.Conv1d(input_nc, inner_nc, kernel_size=5, stride=2, padding=2, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)
		if outermost:
			#upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
			upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
			#upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc, kernel_size=5, stride=2, padding=2, output_padding=1)
			down = [downconv]
			up = [uprelu, upconv] # [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			#upconv = nn.ConvTranspose1d(inner_nc, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
			upconv = nn.ConvTranspose1d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
			#upconv = nn.ConvTranspose1d(inner_nc, outer_nc, kernel_size=5, stride=2, padding=2, output_padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			#upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
			upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
			#upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc, kernel_size=5, stride=2, padding=2, output_padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]
			model = down + [submodule] + up
		self.model = nn.Sequential(*model)
	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([x, self.model(x)], 1)


class Unet1d(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, layers=3):
		super(Unet1d, self).__init__()
		mid_layers = layers - 2
		fact = 2 ** mid_layers
		unet_block = UnetSkipConnectionBlock1d(ngf * fact, ngf * fact, innermost=True)
		for _ in range(mid_layers):
			half_fact = fact // 2
			unet_block = UnetSkipConnectionBlock1d(ngf * half_fact, ngf * fact, submodule=unet_block)
			fact = half_fact
		unet_block = UnetSkipConnectionBlock1d(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)
		self.model = unet_block
	def forward(self, input):
		return self.model(input)