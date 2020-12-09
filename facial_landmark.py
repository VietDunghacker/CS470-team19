import time
import cv2
import os
import random
import math
import sys
import urllib.request
import imutils
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET 
import matplotlib.image as mpimg
import tarfile
from PIL import Image, ImageDraw, ImageOps
from collections import OrderedDict
from skimage import io, transform

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

from .model import Network
from .util import print_overwrite

mtcnn = MTCNN(image_size = 224, margin = 24)

best_network = None

def download_data():
	if not os.path.exists('/data/ibug_300W_large_face_landmark_dataset'):
		urllib.request.urlretrieve('http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz',  '/data/ibug_300W_large_face_landmark_dataset.tar.gz')
		with tarfile.open('/data/ibug_300W_large_face_landmark_dataset.tar.gz', "r:gz") as tar:
			tar.extractall('/data/ibug_300W_large_face_landmark_dataset/')
		os.remove('/data/ibug_300W_large_face_landmark_dataset.tar.gz')

#transform for facial landmark dataset only
class Transforms():
	def __init__(self):
		pass
	
	def rotate(self, image, landmarks, angle):
		angle = random.uniform(-angle, +angle)

		transformation_matrix = torch.tensor([
			[+math.cos(math.radians(angle)), -math.sin(math.radians(angle))], 
			[+math.sin(math.radians(angle)), +math.cos(math.radians(angle))]
		])

		image = imutils.rotate(np.array(image), angle)

		landmarks = landmarks - 0.5
		new_landmarks = np.matmul(landmarks, transformation_matrix)
		new_landmarks = new_landmarks + 0.5
		return Image.fromarray(image), new_landmarks

	def resize(self, image, landmarks, img_size):
		image = TF.resize(image, img_size)
		return image, landmarks

	def color_jitter(self, image, landmarks):
		color_jitter = transforms.ColorJitter(brightness=0.3, 
											  contrast=0.3,
											  saturation=0.3, 
											  hue=0.1)
		image = color_jitter(image)
		return image, landmarks

	def crop_face(self, image, landmarks, crops):
		left = int(crops['left'])
		top = int(crops['top'])
		width = int(crops['width'])
		height = int(crops['height'])

		image = TF.crop(image, top, left, height, width)

		img_shape = np.array(image).shape
		landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
		landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
		return image, landmarks

	def __call__(self, image, landmarks, crops):
		image = Image.fromarray(image)
		image, landmarks = self.crop_face(image, landmarks, crops)
		image, landmarks = self.resize(image, landmarks, (224, 224))
		image, landmarks = self.color_jitter(image, landmarks)
		image, landmarks = self.rotate(image, landmarks, angle=10)
		
		image = TF.to_tensor(image)
		image = TF.normalize(image, [0.5], [0.5])
		return image, landmarks

#Dataset for Facial Landmark
class FaceLandmarksDataset(Dataset):

	def __init__(self, transform=None):

		tree = ET.parse('/data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
		root = tree.getroot()

		self.image_filenames = []
		self.landmarks = []
		self.crops = []
		self.transform = transform
		self.root_dir = '/data/ibug_300W_large_face_landmark_dataset/'
		
		for filename in root[2]:
			self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

			self.crops.append(filename[0].attrib)

			landmark = []
			for num in range(68):
				x_coordinate = int(filename[0][num].attrib['x'])
				y_coordinate = int(filename[0][num].attrib['y'])
				landmark.append([x_coordinate, y_coordinate])
			self.landmarks.append(landmark)

		self.landmarks = np.array(self.landmarks).astype('float32')	 

		assert len(self.image_filenames) == len(self.landmarks)

	def __len__(self):
		return len(self.image_filenames)

	def __getitem__(self, index):
		image = np.array(Image.open(self.image_filenames[index]).convert('RGB'))
		landmarks = self.landmarks[index]
		
		if self.transform:
			image, landmarks = self.transform(image, landmarks, self.crops[index])

		landmarks = landmarks - 0.5

		return image, landmarks

def load_pretrained_facial_landmark(network):
	network.load_state_dict(torch.load('/pretrained/landmarks.pt'))
	return network

def train_facial_landmark(network, train_loader, valid_loader, optimizer, criterion, scheduler, num_epochs, data_dir, loss_min = np.inf):
	start_time = time.time()
	for epoch in range(1 , num_epochs + 1):
		loss_train = 0
		running_loss = 0
		train_num_data = 0
		
		network.train()
		for step, (images, landmarks) in enumerate(train_loader):
			images = images.cuda()
			landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
			
			predictions = network(images)
			
			# clear all the gradients before calculating them
			optimizer.zero_grad()
			
			# find the loss for the current step
			loss_train_step = criterion(predictions, landmarks)
			
			# calculate the gradients
			loss_train_step.backward()
			
			# update the parameters
			optimizer.step()
			
			train_num_data += images.size(0)
			loss_train += loss_train_step.item() * images.size(0)
			running_loss = loss_train/train_num_data
			
			print_overwrite(step + 1, len(train_loader), running_loss, 'train')

		loss_train /= train_num_data
		print('\n--------------------------------------------------')
		print('Epoch: {}'.format(epoch))
		print('Train Loss: {}'.format(loss_train))

		loss_valid = validate_facial_landmark(network, valid_loader, criterion, epoch)
		if loss_valid < loss_min:
			loss_min = loss_valid
			torch.save(network.state_dict(), data_dir) 
			print("\nMinimum Validation Loss of {} at epoch {}/{}".format(loss_min, epoch, num_epochs))
			print('Model Saved\n')
		scheduler.step(loss_valid)

def validate_facial_landmark(network, valid_loader, criterion, epoch):
	
	network.eval()
	with torch.no_grad():
		loss_valid = 0
		valid_num_data = 0
		
		for step, (images, landmarks) in enumerate(valid_loader):
					
			images = images.cuda()
			landmarks = landmarks.view(landmarks.size(0),-1).cuda()
		
			predictions = network(images)

			# find the loss for the current step
			loss_valid_step = criterion(predictions, landmarks)

			valid_num_data += images.size(0)

			loss_valid += loss_valid_step.item() * images.size(0)
			running_loss = loss_valid/valid_num_data

			print_overwrite(step + 1, len(valid_loader), running_loss, 'valid')
	
	loss_valid /= valid_num_data
	
	print('\n--------------------------------------------------')
	print('Valid Loss: {}'.format(loss_valid))
	print('--------------------------------------------------')

	return loss_valid


def train():
	download_data()
	facial_landmark_dataset = FaceLandmarksDataset(Transforms())
	# split the dataset into validation and test sets
	len_valid_set = int(0.1*len(facial_landmark_dataset))
	len_train_set = len(facial_landmark_dataset) - len_valid_set

	print("The length of Train set is {}".format(len_train_set))
	print("The length of Valid set is {}".format(len_valid_set))

	facial_landmark_train_dataset , facial_landmark_valid_dataset,  = torch.utils.data.random_split(facial_landmark_dataset , [len_train_set, len_valid_set])

	# shuffle and batch the datasets
	facial_landmark_train_loader = torch.utils.data.DataLoader(facial_landmark_train_dataset, batch_size = 64, shuffle = True, drop_last = True, num_workers = 4)
	facial_landmark_valid_loader = torch.utils.data.DataLoader(facial_landmark_valid_dataset, batch_size = 64, shuffle = False, drop_last = False, num_workers = 4)

	torch.autograd.set_detect_anomaly(True)
	facial_landmark_network = Network(136)
	facial_landmark_network.cuda()	

	facial_landmark_criterion = nn.MSELoss()
	facial_landmark_optimizer = optim.Adam(facial_landmark_network.parameters(), lr=0.001)
	facial_landmark_scheduler = optim.lr_scheduler.ReduceLROnPlateau(facial_landmark_optimizer, mode='min',
																	 factor=0.5, patience=3, threshold=0.0001, threshold_mode='rel',
																	 cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)

	facial_landmark_num_epochs = 50


	train_facial_landmark(facial_landmark_network, facial_landmark_train_loader, facial_landmark_valid_loader, facial_landmark_optimizer, facial_landmark_criterion, facial_landmark_scheduler, facial_landmark_num_epochs, '/pretrained/facial_landmarks.pt')

#Test facial landmark detection
#argument image_path: path to image
def detect_facial_landmark(image_path):
	image = Image.open(image_path)

	box = tuple(mtcnn.detect(image)[0][0].tolist())
	image = image.crop(box)
	image = pad_image(image)
	image = image.resize((224,224))

	with torch.no_grad():
		image = transforms.ToTensor()(image)
		global best_network
		if best_network is None:
			best_network = Network(136)
			best_network.to(device)
			load_pretrained_facial_landmark(best_network)
			best_network.eval()
				
		image = image.to(device)
		prediction = (best_network(image.unsqueeze(0)).cpu() + 0.5) * 224
		prediction = prediction.view(68, 2)
	return prediction