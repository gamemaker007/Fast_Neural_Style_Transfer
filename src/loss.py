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
		G.div_(c*h*w)
		return G

class StyleReconstructionLoss(nn.Module):
	def forward(self, inp, target):
		i1 = GramMatrix()(inp)
		i2 = GramMatrix()(target)
		return nn.MSELoss()(i1, i2.expand_as(i1))
		

# class IdentityUnit(nn.Module):
# 	def forward(self, inp):
# 		self.output = inp.clone()
# 		return self.output

# 	def backward(self):
# 		self.output.backward(retain_graph=True)
# 		return self.output


# class ContentLoss(nn.Module):

# 	def __init__(self, target, weight):
# 		super(ContentLoss, self).__init__()
# 		# we 'detach' the target content from the tree used
# 		self.target = target.detach() * weight
# 		# to dynamically compute the gradient: this is a stated value,
# 		# not a variable. Otherwise the forward method of the criterion
# 		# will throw an error.
# 		self.weight = weight
# 		self.criterion = nn.MSELoss()

# 	def forward(self, inp):
# 		self.output = inp.clone()
# 		return self.output

# 	def backward(self, retain_graph=True):
# 		# self.loss.backward(retain_graph=retain_graph)
# 		self.loss = self.criterion(self.output * self.weight, self.target)
# 		return self.loss




# # class GramMatrix(nn.Module):

# #     def forward(self, inp):
# #         a, b, c, d = inp.size()  # a=batch size(=1)
# #         # b=number of feature maps
# #         # (c,d)=dimensions of a f. map (N=c*d)

# #         features = inp.view(a * b, c * d)  # resise F_XL into \hat F_XL

# #         G = torch.mm(features, features.t())  # compute the gram product

# #         # we 'normalize' the values of the gram matrix
# #         # by dividing by the number of element in each feature maps.
# #         return G.div(a * b * c * d)


# class StyleLoss(nn.Module):

# 	def __init__(self, target, weight):
# 		super(StyleLoss, self).__init__()
# 		self.target = target.detach() * weight
# 		self.weight = weight
# 		self.gram = GramMatrix()
# 		self.criterion = nn.MSELoss()

# 	def forward(self, inp):
# 		self.output = inp.clone()
# 		self.G = self.gram(self.output)
# 		self.G.mul_(self.weight)
# 		# self.loss = self.criterion(self.G, self.target)
# 		return self.output

# 	def backward(self, retain_graph=True):
# 		# self.loss.backward(retain_graph=retain_graph)
# 		self.loss = self.criterion(self.G, self.target)
# 		return self.loss
