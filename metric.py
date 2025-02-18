# -*- coding=utf-8 -*-


import torch


def compute_mean_absolute_error(real_data, imputed_data):
	mean_abosulte_error = (real_data - imputed_data).abs_().sum().item() / torch.numel(real_data)
	return mean_abosulte_error