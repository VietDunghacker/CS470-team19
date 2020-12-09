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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_overwrite(step, total_step, loss, operation):
	sys.stdout.write('\r')
	if operation == 'train':
		sys.stdout.write("Train Steps: %d/%d  Loss: %f " % (step, total_step, loss))   
	else:
		sys.stdout.write("Valid Steps: %d/%d  Loss: %f " % (step, total_step, loss))
		
	sys.stdout.flush()

def pad_image(image):
	w, h = image.size
	length = max(w, h)
	delta_w = length - w
	delta_h = length - h
	border = (delta_w // 2, delta_h//2 , delta_w - delta_w//2, delta_h - delta_h//2)
	image = ImageOps.expand(image, border = border)
	return image

def train_network(network, train_loader, valid_loader, optimizer, criterion, scheduler, num_epochs, data_dir, accuracy_max = 0):
	start_time = time.time()
	for epoch in range(1 , num_epochs + 1):
		loss_train = 0
		accuracy_train = 0
		running_loss = 0
		running_accuracy = 0
		train_num_data = 0
		
		network.train()
		for step, (images, labels) in enumerate(train_loader):
			images = images.to(device)
			labels = labels.float().to(device) 
			
			predictions = network(images).squeeze()
			
			# clear all the gradients before calculating them
			optimizer.zero_grad()
			
			# find the loss for the current step
			loss_train_step = criterion(predictions, labels)
			
			# calculate the gradients
			loss_train_step.backward()
			
			# update the parameters
			optimizer.step()

			predictions[predictions >= 0.5] = 1
			predictions[predictions < 0.5] = 0
			accuracy_train_step = (predictions == labels).float().cpu().detach().mean()

			train_num_data += images.size(0)
			loss_train += loss_train_step.item() * images.size(0)
			accuracy_train += accuracy_train_step.item() * images.size(0)
			running_loss = loss_train / train_num_data
			running_accuracy = accuracy_train / train_num_data
			
			print_overwrite(step + 1, len(train_loader), running_loss, 'train')

		loss_train /= train_num_data
		accuracy_train /= train_num_data
		print('\n--------------------------------------------------')
		print('Epoch: {}'.format(epoch))
		print('Train Loss: {}'.format(loss_train))
		print("Train Accuracy: {}%".format(accuracy_train * 100))

		accuracy_valid = validate_network(network, valid_loader, criterion, epoch)
		if accuracy_max < accuracy_valid:
			accuracy_max = accuracy_valid
			torch.save(network.state_dict(), data_dir) 
			print("\nMaximum Validation Accuracy of {} at epoch {}/{}".format(accuracy_valid, epoch, num_epochs))
			print('Model Saved\n')
		scheduler.step(accuracy_valid)

def validate_network(network, valid_loader, criterion, epoch):
	
	network.eval()
	with torch.no_grad():
		loss_valid = 0
		accuracy_valid = 0
		running_loss = 0
		running_accuracy = 0
		valid_num_data = 0
		
		for step, (images, labels) in enumerate(valid_loader):
					
			images = images.to(device)
			labels = labels.float().to(device)
		
			predictions = network(images).squeeze()

			# find the loss for the current step
			
			loss_valid_step = criterion(predictions, labels)
			predictions[predictions >= 0.5] = 1
			predictions[predictions < 0.5] = 0
			accuracy_valid_step = (predictions == labels).float().cpu().detach().mean()

			valid_num_data += images.size(0)

			loss_valid += loss_valid_step.item() * images.size(0)
			accuracy_valid += accuracy_valid_step.item() * images.size(0)
			running_loss = loss_valid/valid_num_data
			running_accuracy = accuracy_valid / valid_num_data

			print_overwrite(step + 1, len(valid_loader), running_loss, 'valid')
	
	loss_valid /= valid_num_data
	accuracy_valid /= valid_num_data
	
	print('\n--------------------------------------------------')
	print('Valid Loss: {}'.format(loss_valid))
	print('Valid Accuracy: {}%'.format(accuracy_valid * 100))
	print('--------------------------------------------------')

	return accuracy_valid

def pad_image_cv2(image):
    h, w = image.shape[:2]
    length = max(w, h)
    delta_w = length - w
    delta_h = length - h
    top, bottom = delta_h//2, delta_h - delta_h//2
    left, right = delta_w//2, delta_w - delta_w//2
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image