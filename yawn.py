import time
import cv2
import os
import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET 
import matplotlib.image as mpimg
import tarfile
import re
from PIL import Image, ImageDraw, ImageOps
from collections import OrderedDict
from skimage import io, transform
from zipfile import ZipFile

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

!pip install facenet-pytorch
from facenet_pytorch import MTCNN

from .facial_landmark import detect_facial_landmark
from .model import Network
from .util import print_overwrite, train_network

yawn_data_train_transform = transforms.Compose([
	transforms.Resize((96,96), interpolation = Image.LANCZOS),
	transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
	transforms.RandomRotation(degrees = 10, resample = Image.BILINEAR),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

yawn_data_test_transform = transforms.Compose([					   
	transforms.Resize((96,96), interpolation = Image.LANCZOS),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

class YawnDataset(Dataset):

	def __init__(self, mode ,transform=None):
		self.image_filenames = []
		self.label = []
		self.transform = transform
		self.root_dir = '/data/dataset_new'
		
		for filename in os.listdir(os.path.join(self.root_dir, mode, 'no_yawn')):
			self.image_filenames.append(os.path.join(self.root_dir, mode, 'no_yawn', filename))
			self.label.append(0)
		for filename in os.listdir(os.path.join(self.root_dir, mode, 'yawn')):
			self.image_filenames.append(os.path.join(self.root_dir, mode, 'yawn', filename))
			self.label.append(1) 

		assert len(self.image_filenames) == len(self.label)

	def __len__(self):
		return len(self.image_filenames)

	def __getitem__(self, index):
		image = Image.open(self.image_filenames[index]).convert('RGB')
		label = self.label[index]
		
		if self.transform:
			image = self.transform(image)

		return image, label

class YawnTransforms():
	def __init__(self, transforms):
		self.transforms = transforms
	
	def pad(self, image):
		w, h = image.size
		length = max(w, h)
		delta_w = length - w
		delta_h = length - h
		border = (delta_w // 2, delta_h//2 , delta_w - delta_w//2, delta_h - delta_h//2)
		image = ImageOps.expand(image, border = border)
		return image

	def crop_mouth(self, image):
		landmarks = detect_facial_landmark(image)
		mouth = landmarks[49 : ]
		x = min(mouth[:,0])
		z = max(mouth[:,0])
		y = min(mouth[:,1])
		t = max(mouth[:,1])
		w = int(abs(z - x) * 0.2)
		h = int(abs(t - y) * 0.2)

		image = image.crop((x - w // 2, y - h // 2, z + w // 2, t + h // 2))
		return image

	def __call__(self, image):
		image = self.crop_mouth(image)
		image = self.pad(image)
		image = self.transforms(image)
		return image

def load_pretrained_yawn(network):
	network.load_state_dict(torch.load('/pretrained/yawn.pt'))
	return network

def train():
	yawn_dataset_train = YawnDataset('train', YawnTransforms(yawn_data_train_transform))
	yawn_dataset_test = YawnDataset('test', YawnTransforms(yawn_data_test_transform))
	yawn_train_loader = torch.utils.data.DataLoader(yawn_dataset_train, batch_size = 64, shuffle = True, drop_last = True)
	yawn_valid_loader = torch.utils.data.DataLoader(yawn_dataset_test, batch_size = 64, shuffle = False, drop_last = False)

	torch.autograd.set_detect_anomaly(True)
	yawn_network = Network(1)
	yawn_network.cuda()	

	yawn_criterion = nn.BCEWithLogitsLoss()
	yawn_optimizer = optim.Adam(yawn_network.parameters(), lr=0.001)
	yawn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(yawn_optimizer, mode='max',
																	 factor=0.5, patience=1, threshold=0.0001, threshold_mode='abs',
																	 cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)

	yawn_num_epochs = 10
	train_network(yawn_network, yawn_train_loader, yawn_valid_loader, yawn_optimizer, yawn_criterion, yawn_scheduler, yawn_num_epochs, 'pretrained/yawn.pt')

#test yawn detector
def test_yawn_detector(image_path):
	image = Image.open(image_path)
	with torch.no_grad():
		best_network = Network(1)
		best_network.to(device)
		load_pretrained_yawn(best_network)
		best_network.eval()
				
		image = image.to(device)
		prediction = torch.sigmoid(best_network(image.unsqueeze(0)).squeeze()).item()
		prediction = 1 if prediction >= 0.5 else 0
		
		return prediction