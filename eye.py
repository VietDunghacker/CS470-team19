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

from .facial_landmark import detect_facial_landmark
from .model import Network
from .util import print_overwrite, train_network

eye_transform = transforms.Compose([
	transforms.Resize(96, interpolation = Image.LANCZOS),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

def download_data():
	if not os.path.exists('/data/mrlEyes_2018_01'):
		urllib.request.urlretrieve('http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip', '/data/mrlEyes_2018_01.zip')
	    with ZipFile('/data/mrlEyes_2018_01.zip', 'r') as zip_ref:
	        zip_ref.extractall('/data/')
	    os.remove('/data/mrlEyes_2018_01.zip')


class EyeDataset(Dataset):

	def __init__(self, transform=None):
		self.image_filenames = []
		self.label = []
		self.transform = transform
		self.root_dir = '/data/mrlEyes_2018_01'
		
		for folder in os.listdir(self.root_dir):
			if folder == 'annotation.txt' or folder == 'stats_2018_01.ods':
				continue
			for image_filename in os.listdir(os.path.join(self.root_dir, folder)):
				self.image_filenames.append(os.path.join(self.root_dir, folder, image_filename))
				info = image_filename[ : image_filename.find('.')].split('_')
				state = info[4]
				self.label.append(int(info[4]))

		assert len(self.image_filenames) == len(self.label)

	def __len__(self):
		return len(self.image_filenames)

	def __getitem__(self, index):
		image = Image.open(self.image_filenames[index]).convert('RGB')
		label = self.label[index]
		
		if self.transform:
			image = self.transform(image)

		return image, label


def train():
	download_data()
	eye_dataset = EyeDataset(eye_transform)

	len_eye_valid_set = int(0.1*len(eye_dataset))
	len_eye_train_set = len(eye_dataset) - len_eye_valid_set

	eye_train_dataset , eye_valid_dataset = torch.utils.data.random_split(eye_dataset , [len_eye_train_set, len_eye_valid_set])

	eye_train_loader = torch.utils.data.DataLoader(eye_train_dataset, batch_size = 128, shuffle = True, drop_last = True, num_workers = 4)
	eye_valid_loader = torch.utils.data.DataLoader(eye_valid_dataset, batch_size = 128, shuffle = False, drop_last = False, num_workers = 4)

	torch.autograd.set_detect_anomaly(True)
	eye_network = Network(1)
	eye_network.cuda()

	eye_criterion = nn.BCEWithLogitsLoss()
	eye_optimizer = optim.Adam(eye_network.parameters(), lr=0.001)
	eye_scheduler = optim.lr_scheduler.ReduceLROnPlateau(eye_optimizer, mode='max',
																	 factor=0.5, patience=1, threshold=0.0001, threshold_mode='abs',
																	 cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)

	eye_num_epochs = 10

	train_yawn(eye_network, eye_train_loader, eye_valid_loader, eye_optimizer, eye_criterion, eye_scheduler, eye_num_epochs, 'pretrained/eye.pt')

#test eye close detector
def test_eye_close_detector(image_path):
	image = Image.open(image_path)
	with torch.no_grad():
		best_network = Network(1)
		best_network.to(device)
		load_pretrained_eye(best_network)
		best_network.eval()
				
		image = image.to(device)
		prediction = torch.sigmoid(best_network(image.unsqueeze(0)).squeeze()).item()
		prediction = 1 if prediction >= 0.5 else 0
		
		return prediction