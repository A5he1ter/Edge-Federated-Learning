import torch
from torchvision import datasets, transforms

def get_dataset(dir, name):

	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	elif name=='cifar10':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
		
	elif name=='emnist':
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1736,), (0.3317,))
		])

		train_dataset = datasets.EMNIST(root=dir, split='byclass', train=True, download=True, transform=transform)
		eval_dataset = datasets.EMNIST(root=dir, split='byclass', train=False, download=True, transform=transform)
	
	return train_dataset, eval_dataset