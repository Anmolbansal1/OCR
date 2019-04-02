import os
import cv2
import string
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable

from dataset.collate_fn import text_collate
from dataset.data_transform import Resize, Rotation, Translation, Scale
from model_loader import load_model
from torchvision.transforms import Compose

import string

import editdistance
def give_output(imgs):
	# img_path = '../../east-text-detection/crop'
	# imgs = os.listdir(img_path)
	label = ''
	backend = 'resnet18'
	snapshot = './snaps/crnn_best'
	input_size = [320, 32]
	seq_proj = [10, 20]
	
	print('Creating transform')
	transform = Compose([
		# Rotation(),
		Resize(size=(input_size[0], input_size[1]))
	])
	
	abc = string.digits+string.ascii_letters
	print('Possibilities - ' + abc)
	# load model
	net = load_model(abc, seq_proj, backend, snapshot, cuda=True).eval()
	out_arr = []
	for imgx in imgs:
	
		# prepare image
		# img = cv2.imread(os.path.join(img_path, imgx))
		img = imgx
		print('Readed image of shape - ' + str(img.shape))
		img = transform(img)
		img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0)
		
		img = Variable(img)
		print('Image fed to model of shape - ', str(img.shape))
		out = net(img, decode=True)
		out_arr.append(out)
	
	return out_arr