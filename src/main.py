import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets, models
import os
import sys
from PIL import Image
from model import ImageTransformationNetwork, TrainDataset, VGG16
import tqdm
import copy
import numpy as np
from loss import FeatureReconstructionLoss, GramMatrix, StyleReconstructionLoss


if torch.cuda.is_available():
	gpu=True
	print('Running on GPU!')
else:
	gpu=False

# torch.manual_seed(1)

img_size = 256 if gpu else 128  # use small size if no gpu

RGB = transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])
BGR = transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])

load_content = transforms.Compose([transforms.Resize((img_size, img_size)),
		transforms.CenterCrop(img_size),
							transforms.ToTensor(),
							BGR,
							transforms.Lambda(lambda x: x.mul(255))
						 ])

load_style = transforms.Compose([ transforms.ToTensor(),
							BGR,
							 transforms.Normalize(mean=[0.406, 0.456, 0.485], #subtract imagenet mean
												std=[0.225,0.224,0.229]),
						   ])

image_dir = '../Images/'	

def imshow( data,filename,direc=image_dir+'run/'):
	img = data.clone().clamp(0, 255).view(3,img_size,img_size)
	img = RGB(img)
	img = img.numpy()
	img = img.transpose(1, 2, 0).astype("uint8")
	img = Image.fromarray(img)
	img.save(direc+filename+'.jpg')

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

direc = image_dir+'content_images/'
content_images = datasets.ImageFolder(direc, transform=load_content)

direc = image_dir+'style_images/'
style_images=datasets.ImageFolder(direc, transform = load_style)


style_images = style_images[0][0].unsqueeze(0)
vgg = VGG16()
model = ImageTransformationNetwork(img_size)

if gpu:
	style_images_var = Variable(style_images.cuda(),requires_grad=False)
	vgg = vgg.cuda()
	model = model.cuda()
	style_loss_fns = [StyleReconstructionLoss().cuda()] * 4
	content_loss_fns = [FeatureReconstructionLoss().cuda()]
else:			
	style_images_var = Variable(style_images, requires_grad=False)
	style_loss_fns = [StyleReconstructionLoss()] * 4
	content_loss_fns = [FeatureReconstructionLoss()] 

for param in vgg.parameters():
	param.requires_grad = False

style_weights = [1e5,1e5,1e5,1e5]
# style_weights = [1e3/n**2 for n in [64,128,256,512]]
content_weights = [1e5]
tv_weight=[1e-9]



style_targets = [sl.detach() for sl in vgg(style_images_var)]

show_batch = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 4
content_images_loader = torch.utils.data.DataLoader(content_images,
	batch_size=batch_size, shuffle=True, num_workers=1)

def transform(batch):
	# normalize using imagenet mean and std
	mean = batch.data.new(batch.data.size())
	std = batch.data.new(batch.data.size())
	mean[:, 2, :, :] = 0.485
	mean[:, 1, :, :] = 0.456
	mean[:, 0, :, :] = 0.406
	std[:, 2, :, :] = 0.229
	std[:, 1, :, :] = 0.224
	std[:, 0, :, :] = 0.225
	batch = torch.div(batch, 255.0)
	batch = batch - Variable(mean)
	batch = batch / Variable(std)
	return batch

# def rev_transform(x):
# 	x.data.div_(255)
# 	x.data[:,0].mul_(0.225).add_(0.406)
# 	x.data[:,1].mul_(0.224).add_(0.456)
# 	x.data[:,2].mul_(0.229).add_(0.485)
# 	x.data.mul_(255)
# 	return x


def train(epoch):
	model.train()
	content_losses = 0
	total_losses =0
	style_losses =0
	tv_losses = 0
	for batch_idx,(content_images_batch,_) in enumerate(content_images_loader):
		optimizer.zero_grad()

		input_images_batch = content_images_batch.clone()
		
		if gpu:
			input_images_batch = Variable(input_images_batch.cuda())
			content_images_batch_var = Variable(content_images_batch.cuda(), 
				requires_grad=False)
		else:
			input_images_batch = Variable(input_images_batch)
			content_images_batch_var = Variable(content_images_batch, 
				requires_grad=False)
		content_images_batch_var = transform(content_images_batch_var)
		content_targets = [cl.detach() for cl in vgg(content_images_batch_var)]
		
		
		output_images = model(input_images_batch)		
		output_images = transform(output_images)
		output = vgg(output_images)
		
		content_loss = content_weights[0]*content_loss_fns[0](output[2], 
					content_targets[2])
		
		style_loss=[]
		for i in range(len(style_loss_fns)):
			loss = style_weights[i]*style_loss_fns[i](output[i], style_targets[i])
			style_loss.append(loss)		
		
		# tv_loss = tv_weight[0] * (torch.sum(torch.abs(output_images[:, :, :, :-1] - output_images[:, :, :, 1:])) + 
		# torch.sum(torch.abs(output_images[:, :, :-1, :] - output_images[:, :, 1:, :])))
		
		style_loss = sum(style_loss)
		total_loss = content_loss + style_loss #+ tv_loss		
		total_loss.backward()
		optimizer.step()
		content_losses+=content_loss.data[0]		
		total_losses += total_loss.data[0]
		style_losses+=style_loss.data[0]
		# tv_losses +=tv_loss.data[0]
		
		if batch_idx%show_batch == 0:
			print('Epoch: %d, Iteration: %d, loss: %f, c:%f, s:%f, tv:%f'%(epoch, batch_idx, total_losses/show_batch, 
				content_losses/show_batch, style_losses/show_batch, tv_losses/show_batch
				))
			content_losses = 0
			total_losses =0
			style_losses =0
			tv_losses = 0
			predict(epoch,batch_idx)
			# print(list(model.parameters())[1].data)

def predict(e=0,b=0):
	model.eval()
	test_data = load_content(pil_loader(image_dir + 'test_image.jpg')).unsqueeze(0)
	imshow(test_data, 'test')
	if gpu:
		test_data = Variable(test_data.cuda(), requires_grad = False)
	else:
		test_data = Variable(test_data, requires_grad = False)
	output = model(test_data).detach().cpu()
	# output= rev_transform(output)
	imshow(output.data, 'm55_'+str(e)+'_'+str(b))

for epoch in range(0,2,1):
	train(epoch) 


predict(20,20) 


print('Its done.')
