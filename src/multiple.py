import os
import sys
import tqdm
import copy
import numpy as np
from PIL import Image
import logging

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


from model import ImageTransformationNetwork, VGG16
from loss import FeatureReconstructionLoss, GramMatrix, StyleReconstructionLoss

script_name = os.path.basename(__file__)
log_file_name = script_name + '_3.out'
# log_file_name = 'two_face.out'

logging.basicConfig(filename=log_file_name,level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("\n\n\n############ 55 Multiple Images on Validation Set Part 3 F #################\n\n\n")
if torch.cuda.is_available():
	gpu=True
	logger.info('Running on GPU!')
	torch.cuda.manual_seed(5)
else:
	gpu=False
	logger.info('Running on CPU!')
	torch.manual_seed(5)


img_size = 256 if gpu else 128  # use small size if no gpu



load_content = transforms.Compose([
							transforms.Resize((img_size, img_size)),
							transforms.CenterCrop(img_size),
							transforms.ToTensor(),
							transforms.Lambda(lambda x: x.mul(255))
							])

load_style = transforms.Compose([ transforms.ToTensor(),
							transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
							])

def denormalize(img):
	mean = img.new(img.size())
	std = img.new(img.size())
	mean[ 0, :, :] = 0.485
	mean[ 1, :, :] = 0.456
	mean[ 2, :, :] = 0.406
	std[ 0, :, :] = 0.229
	std[ 1, :, :] = 0.224
	std[ 2, :, :] = 0.225
	img = (img*std)+mean
	return img


image_dir = '../Images/'	

def imshow( img,filename,direc=image_dir+'run/'):
	img = img.clamp(0, 255)
	img = img.numpy()
	img = img.transpose(1, 2, 0).astype("uint8")
	img = Image.fromarray(img)
	img.save(direc+filename+'_3.jpg')

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

direc = image_dir+'content-images/'
content_images = datasets.ImageFolder(direc, transform=load_content)

direc = image_dir+'multi/'
style_images=datasets.ImageFolder(direc, transform = load_style)
style_img_arr = []
style_imgs_count = 0

for img in style_images:
	style_img_arr.append(img[0].unsqueeze(0))
	style_imgs_count += 1
print('appended')

vgg = VGG16()
model = ImageTransformationNetwork(img_size)

style_images_var = []
if gpu:
	vgg = vgg.cuda()
	model = model.cuda()
	for i in range(style_imgs_count):
		style_images_var.append(Variable(style_img_arr[i].cuda(),volatile=True))
	style_loss_fns = [StyleReconstructionLoss().cuda()] * 4
	content_loss_fns = [FeatureReconstructionLoss().cuda()]
else:			
	for i in range(style_imgs_count):
		style_images_var.append(Variable(style_img_arr[i],volatile=True))
	# style_images_var = Variable(style_images, volatile=True)
	style_loss_fns = [StyleReconstructionLoss()] * 4
	content_loss_fns = [FeatureReconstructionLoss()] 

for param in vgg.parameters():
	param.requires_grad = False

style_weights = []
for i in range(style_imgs_count):
	style_weights.append(5e3)
content_weights = [1e5]
tv_weight=[1e-7]

style_targets = []
for i in range(style_imgs_count):
	style_targets.append([sl.detach()  for sl in vgg(style_images_var[i])])

show_batch = 400
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 4
content_images_loader = torch.utils.data.DataLoader(content_images,
	batch_size=batch_size, shuffle=True, num_workers=1)

def scale_down(batch):
	batch = torch.div(batch, 255.0)
	return batch

def scale_up(batch):
	batch = torch.mul(batch, 255.0)
	return batch

def normalize(batch):
	# normalize using imagenet mean and std
	mean = batch.data.new(batch.data.size())
	std = batch.data.new(batch.data.size())
	mean[:, 0, :, :] = 0.485
	mean[:, 1, :, :] = 0.456
	mean[:, 2, :, :] = 0.406
	std[:, 0, :, :] = 0.229
	std[:, 1, :, :] = 0.224
	std[:, 2, :, :] = 0.225
	batch = batch - Variable(mean, requires_grad=False)
	batch = batch / Variable(std, requires_grad=False)
	return batch

def train(epoch):
	model.train()
	content_losses = 0.0
	total_losses =0.0
	style_losses = []
	for i in range(style_imgs_count):
		style_losses.append(0.0)

	tv_losses = 0.0
	save_model = 5000
	for batch_idx,(content_images_batch,_) in enumerate(content_images_loader):
		optimizer.zero_grad()
		model.zero_grad()

		input_images_batch = content_images_batch.clone()
		
		if gpu:
			input_images_batch_var = Variable(input_images_batch.cuda())
			content_images_batch_var = Variable(content_images_batch.cuda(), 
				requires_grad=False)
		else:
			input_images_batch_var = Variable(input_images_batch)
			content_images_batch_var = Variable(content_images_batch, 
				requires_grad=False)
		
		content_images_batch_var = normalize(scale_down(content_images_batch_var))
		content_targets = [cl.detach() for cl in vgg(content_images_batch_var)]
		
		
		output_images = model(input_images_batch_var)		
		output_images = normalize(scale_down(output_images))
		output = vgg(output_images)
		
		content_loss = content_weights[0]*content_loss_fns[0](output[2], 
					content_targets[2])
		
		style_loss_arr=[]
		for j in range(style_imgs_count):
			style_loss = []
			for i in range(len(style_loss_fns)):
				loss = style_weights[j]*style_loss_fns[i](output[i], style_targets[j][i])
				style_loss.append(loss)		
			style_loss = sum(style_loss)
			style_loss_arr.append(style_loss)

		style_loss = sum(style_loss_arr)

		tv_loss = tv_weight[0] * (torch.sum(torch.abs(output_images[:, :, :, :-1] - output_images[:, :, :, 1:])) + 
		torch.sum(torch.abs(output_images[:, :, :-1, :] - output_images[:, :, 1:, :])))
		
		total_loss = content_loss + style_loss + tv_loss		
		total_loss.backward()
		optimizer.step()
		total_loss = total_loss.detach()
		total_losses += total_loss.data[0]
		content_losses+=content_loss.data[0]
		# style_losses+=style_loss.data[0]
		
		for j in range(style_imgs_count):
				style_losses[j]+=style_loss_arr[j].data[0]
		
		tv_losses +=tv_loss.data[0]
		
		if batch_idx%show_batch == 0:
			logger.info('Epoch: %d, Iteration: %d, loss: %f, c:%f, tv:%f'%(epoch, batch_idx, total_losses/show_batch, 
				content_losses/show_batch, tv_losses/show_batch
				))
			print_str = 'Style Losses:\n'
			for i in range(style_imgs_count):
				print_str += str(i+1) + ': ' + str(style_losses[i]/show_batch) + '\t'

			logger.info(print_str)

			content_losses = 0.0
			total_losses = 0.0
			# style_losses = 0.0
			for i in range(style_imgs_count):
				style_losses[i] = 0.0
			tv_losses = 0.0
			predict(epoch,batch_idx)

		if batch_idx%save_model:
			direc =  image_dir + 'models/'
			save_model_path = direc+"multi_intermediate_2.pth"
			torch.save(model.state_dict(), save_model_path)


def predict(e=0,b=0):
	model.eval()
	test_data = load_content(pil_loader(image_dir + 'test_image_old.jpg')).unsqueeze(0)

	if gpu:
		test_data = Variable(test_data.cuda(), volatile=True)
	else:
		test_data = Variable(test_data, volatile=True)
	output = model(test_data).detach().cpu()

	imshow(output.data[0], 'multiple'+str(e)+'_'+str(b))


for epoch in range(0,2,1):
	train(epoch) 


predict(20,20) 
direc = image_dir + '/models/'
save_model_path = direc+"multi.pth"
torch.save(model.state_dict(), save_model_path)

logger.info('Its done.')
