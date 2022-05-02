#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import time, pdb, numpy
from utils import accuracy


class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
		super(LossFunction, self).__init__()

		self.test_normalize = True
	    
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.fc 		= nn.Linear(nOut,nClasses)

		self.class_wise_learnable_norm = True # czy potrzebne?

		if self.class_wise_learnable_norm:
			WeightNorm.apply(self.fc, 'weight', dim=0) #split the weight update component to direction and norm

		if nOut <=200: # dobrać odpoweidni scale factor? nOut -> out_dim
			self.scale_factor = 2
		else:
			self.scale_factor = 10

		print('Initialised Cosine Distance Loss')

	def forward(self, x, label=None):
		x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
		x_normalized = x.div(x_norm  + 0.00001)
		if not self.class_wise_learnable_norm:
			L_norm = torch.norm(self.fc.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.fc.weight.data)
			self.fc.weight.data = self.fc.weight.data.div(L_norm + 0.00001)
		cos_dist = self.fc(x_normalized)
		scores = self.scale_factor * (cos_dist)

		nloss = self.criterion(scores, label)
		prec1 = accuracy(scores.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1