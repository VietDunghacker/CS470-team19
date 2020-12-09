import time
import cv2
import os
import random
import math
import sys
import shutil
import urllib.request
import imutils
from imutils.video import FileVideoStream, VideoStream
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET 
import imutils
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
mtcnn = MTCNN(image_size = 224, margin = 24)

from .model import Network
from .facial_landmark import load_pretrained_facial_landmark
from .yawn import load_pretrained_yawn
from .eye import load_pretrained_eye
from .util import pad_image_cv2

facial_landmark_network = Network(136)
eye_network = Network(1)
yawn_network = Network(1)

facial_landmark_network = load_pretrained_facial_landmark(facial_landmark_network)
eye_network = load_pretrained_eye(eye_network)
yawn_network = load_pretrained_yawn(yawn_network)

facial_landmark_network.eval()
eye_network.eval()
yawn_network.eval()

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

#face detector using MTCNN or Haar Cascade Classifier
def face_detector(image):
	box = mtcnn.detect(image)[0]
	if box is not None:
		return tuple(mtcnn.detect(image)[0][0].tolist())
	else:
		boxes = face_cascade.detectMultiScale(image, 1.3, 5)
		if len(boxes) >= 1:
			area = []
			for box in boxes:
				x, y, z, t = box
				area.append(z * t)
			x, y, z, t = tuple(boxes[np.argmax(area)])
			return x, y, x + z, y + t
		else:
			return "Not Found"

def facial_landmark_predictor(image, network, box):
	x, y, z, t = box
	image = image[int(y) : int(t), int(x) : int(z)]
	h, w = image.shape[:2]
	image = pad_image_cv2(image)
	image = cv2.resize(image, (224,224))

	with torch.no_grad():	
		image = transforms.ToTensor()(image)
		start = time.time()
		landmarks = (network(image.unsqueeze(0)) + 0.5) * max(w,h)
		landmarks = landmarks.view(68, 2).numpy()

		if w < h:
			for i in range(68):
				landmarks[i][0] -= (h - w) // 2
		else:
			for i in range(68):
				landmarks[i][1] -= (w - h) // 2
		
		for i in range(68):
			landmarks[i][0] += x
			landmarks[i][1] += y

	return landmarks

#Yawn detector on cv2 image
#Input image: cv2 image
#Input network: yawn detector neural network
def yawn_detector(image, network):
	with torch.no_grad():
		image = cv2.resize(image, (96, 96))
		image = transforms.ToTensor()(image)
		prediction = network(image.unsqueeze(0)).squeeze()
		return torch.sigmoid(prediction).item()

#Eye close detector on cv2 image
#Input image: cv2 image
#Input network: eye close detector neural network
def eye_close_detector(image, network):
	with torch.no_grad():
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (96, 96))
		image = np.stack((image,)*3, axis=-1)
		image = transforms.ToTensor()(image)
		prediction = network(image.unsqueeze(0)).squeeze()
		return torch.sigmoid(prediction).item()

#Test the model on a video stream
#If streaming, 
def test(video_path):
	vs = FileVideoStream(video_path).start()
	time.sleep(1.0)

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	writer = None
	(height, width) = (None, None)
	zeros = None
	count = 0
	total = 0
	undetected_frame = 0
	eye_close_count = 0
	mouth_close_count = 0

	while True:
		start = time.time()

		frame = vs.read()

		#break if end of video
		if not vs.more():
			break

		#resize the frame
		frame = imutils.resize(frame, width=360)
		box = face_detector(frame)

		if box != "Not Found":
			undetected_frame = 0

			landmarks = facial_landmark_predictor(frame, facial_landmark_network, box)

			leftEye = landmarks[42 : 48]
			rightEye = landmarks[36 : 42]
			mouth = landmarks[48 : 68]

			x = int(min(leftEye[:, 0]))
			y = int(min(leftEye[:, 1]))
			z = int(max(leftEye[:, 0]))
			t = int(max(leftEye[:, 1]))
			delta = (z - x) - (t - y)
			x, y, z, t = x,y - delta//2,z,t + (delta - delta//2)
			left_eye_image = frame[y : t, x : z]
			left_eye_box = (x, y, z, t)
			cv2.rectangle(frame,(x,y),(z,t),(0,255,0),2)

			x = int(min(rightEye[:, 0]))
			y = int(min(rightEye[:, 1]))
			z = int(max(rightEye[:, 0]))
			t = int(max(rightEye[:, 1]))
			delta = (z - x) - (t - y)
			x, y, z, t = x,y - delta//2,z,t + (delta - delta//2)
			right_eye_image = frame[y : t, x : z]
			right_eye_box = (x, y, z, t)
			cv2.rectangle(frame,(x,y),(z,t),(0,255,0),2)

			x = int(min(mouth[:, 0]))
			y = int(min(mouth[:, 1]))
			z = int(max(mouth[:, 0]))
			t = int(max(mouth[:, 1]))
			ratio = (z - x) / (t - y)
			if ratio > 1.5:
				delta = int((z - x) * 2/3 - (t - y))
				y = y - delta//2
				t = t + delta - delta//2
			else:
				delta = int((t - y) * 3/2 - (z - x))
				x = x - delta//2
				z = z + delta - delta//2
			mouth_image = frame[y : t, x : z]
			mouth_box = (x, y, z, t)
			cv2.rectangle(frame,(x,y),(z,t),(0,0,255),2)

			left_eye_close = eye_close_predictor(left_eye_image, eye_network)
			right_eye_close = eye_close_predictor(right_eye_image, eye_network)
			mouth_close = yawn_predictor(mouth_image, yawn_network)

			if (left_eye_close + right_eye_close) >= 1:
				eye_close_count += 1
			else:
				eye_close_count = 0
			if (mouth_close) >= 0.5:
				mouth_close_count += 1
			else:
				mouth_close_count = 0
			cv2.putText(frame, "Eyes: {:.2f}".format((right_eye_close + left_eye_close)/2),
						(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Yawn: {:.2f}".format(mouth_close),
						(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


			if (eye_close_count >= 10) or (mouth_close_count >= 5):
				cv2.putText(frame, "DRIVER DROWSINESS ALERT!", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			undetected_frame += 1
			if undetected_frame >= 10:
				cv2.putText(frame, "FACE UNDETECTED!", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if writer is None:
			# Get the space size (width and height) of the frame, and instantiate the video stream writer
			(height, width) = frame.shape[:2]
			writer = cv2.VideoWriter('output.avi', fourcc, 20, (width, height), True)

		# Write frame to video
		writer.write(frame)

	vs.stop()
	writer.release()