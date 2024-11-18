from copy import deepcopy

import torch

from models import models
from edge_server import *
from utils.utils import Adding_Trigger

device = None
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Server(object):
	
	def __init__(self, conf, eval_dataset):
		self.conf = conf
		self.global_model = models.get_model(self.conf["model_name"])
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

	# def model_aggregate(self, weight_accumulator):
	# 	for name, data in self.global_model.state_dict().items():
	#
	# 		update_per_layer = weight_accumulator[name] * self.conf["lambda"]
	#
	# 		if data.type() != update_per_layer.type():
	# 			data.add_(update_per_layer.to(torch.int64))
	# 		else:
	# 			data.add_(update_per_layer)

	def set_edge_servers(self, edge_servers):
		self.edge_servers = edge_servers

	# def model_aggregate(self, weight_accumulator_list):
	# 	for i in range(self.conf["num_edge_servers"]):
	# 		for name, data in self.global_model.state_dict().items():
	# 			data.add_((weight_accumulator_list[i][name] * self.conf["lambda"]).long())

	def model_aggregate(self, weight_accumulator_list):
		grads = []
		global_params = {}
		for i in range(len(weight_accumulator_list)):
			sub_global_params_flatten = weight_accumulator_list[i]
			grads = sub_global_params_flatten[None, :] if len(grads) == 0 else torch.cat(
				(grads, sub_global_params_flatten[None, :]), 0
			)

		global_params_flatten = torch.mean(grads, dim=0)

		idx = 0
		for key, val in self.global_model.state_dict().items():
			param = global_params_flatten[idx: idx + len(val.data.view(-1))].reshape(val.data.shape)
			idx = idx + len(val.data.view(-1))
			global_params[key] = deepcopy(param)

		return global_params

	def model_eval(self):
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]

			data = data.to(device)
			target = target.to(device)
			
			# if torch.cuda.is_available():
			# 	data = data.cuda()
			# 	target = target.cuda()
				
			
			output = self.global_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc =  float(correct) / float(dataset_size)
		total_l = total_loss / dataset_size

		return acc, total_l

	def TACC(self):
		loss_collecter = []

		self.global_model.eval()
		sum_accu = 0
		num = 0
		loss_collecter.clear()
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			data, target = data.to(device), target.to(device)
			output = self.global_model(data)
			loss = torch.nn.functional.cross_entropy(output, target.long())
			loss_collecter.append(loss.item())
			output = torch.argmax(output, dim=1)
			sum_accu += (output == target).float().mean()
			num += 1

		acc = sum_accu / num
		avg_loss = sum(loss_collecter) / len(loss_collecter)

		return acc, avg_loss

	def ASR(self, model):
		sum_ASR = 0
		count = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch

			data = data.to(device)
			target = target.to(device)

			for example_id in range(data.shape[0]):
				data[example_id] = Adding_Trigger(data[example_id])

			output = model(data)
			output = torch.argmax(output, dim=1)

			for i, v in enumerate(output):
				if v != target[i] and v == 0:
					count += 1

			sum_ASR += data.shape[0]

		asr = count / sum_ASR
		return asr

	def eval(self, model):
		model.eval()

		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			dataset_size += data.size()[0]

			data = data.to(device)
			target = target.to(device)

			# if torch.cuda.is_available():
			# 	data = data.cuda()
			# 	target = target.cuda()

			output = model(data)

			total_loss += torch.nn.functional.cross_entropy(output, target,
															reduction='sum').item()  # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		return acc, total_l