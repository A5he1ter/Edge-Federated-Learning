import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from models.Resnets import *

device = None
"""
如果使用cuda的话，把该段代码改为
if torch.cuda.is_available():
	device = torch.device('cuda')
else device = torch.device('cpu')
"""
if torch.backends.mps.is_available():
	device = torch.device('mps')
else:
	device = torch.device('cpu')

class Mnist_2NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 100)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, inputs):
		tensor = inputs.view(-1, 784)
		tensor = F.relu(self.fc1(tensor))
		tensor = self.fc2(tensor)
		return tensor

class EMnist_CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 30, 3, 1, 1)
		self.pool1 = nn.MaxPool2d(2, 2, 0)
		self.conv2 = nn.Conv2d(30, 5, 3, 1, 1)
		self.pool2 = nn.MaxPool2d(2, 2, 0)
		self.fc1 = nn.Linear(7 * 7 * 5, 100)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, inputs):
		tensor = inputs.view(-1, 1, 28, 28)
		tensor = F.relu(self.conv1(tensor))
		tensor = self.pool1(tensor)
		tensor = F.relu(self.conv2(tensor))
		tensor = self.pool2(tensor)
		tensor = tensor.view(-1, 7 * 7 * 5)
		tensor = F.relu(self.fc1(tensor))
		tensor = self.fc2(tensor)
		return tensor

def get_model(name="vgg16", pretrained=True):
	if name == "resnet18":
		model = ResNet18_cifar10(num_classes=10) # batch_size = 128
	elif name == "2NN":
		model = Mnist_2NN() # batch_size = 64
	elif name == "CNN":
		model = EMnist_CNN() # batch_size = 256

	return model.to(device)
		
	# if torch.cuda.is_available():
	# 	return model.cuda()
	# else:
	# 	return model