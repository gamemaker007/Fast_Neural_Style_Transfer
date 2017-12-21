import torch
import torch.nn as nn

class FeatureReconstructionLoss(nn.Module):
	def forward (self, inp, target):
		b,c,h,w = inp.size()
		out = nn.MSELoss()(inp, target)
		out = out/(c*h*w)
		return out

class GramMatrix(nn.Module):
	def forward(self, inp):
		b,c,h,w = inp.size()
		F = inp.view(b, c, h*w)
		G = torch.bmm(F, F.transpose(1,2)) 
		G = torch.div(G,c*h*w)
		return G

class StyleReconstructionLoss(nn.Module):
	def forward(self, inp, target):
		i1 = GramMatrix()(inp)
		i2 = GramMatrix()(target)
		return nn.MSELoss()(i1, i2.expand_as(i1))
		