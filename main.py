import argparse, json
import datetime
import os
import logging
from collections import OrderedDict
from copy import deepcopy
from curses.ascii import isxdigit

import numpy as np
import torch, random
import matplotlib.pyplot as plt
from sympy.codegen.ast import float32

from cloud_server import *
from edge_server import *
from client import *
import models, datasets

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, "r") as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

	edge_servers = []

	server = Server(conf, eval_datasets)
	clients = []
	num_client_groups = conf["num_edge_servers"]
	client_groups = []

	# 创建客户端，并标记恶意客户端
	num_malicious_clients = int(conf["malicious_ratio"] * conf["num_models"])
	print("恶意客户端数量", num_malicious_clients)
	malicious_clients = random.sample(range(conf["num_models"]), num_malicious_clients)
	print("恶意客户端", malicious_clients)

	for c in range(conf["num_models"]):
		if c in malicious_clients:
			# TODO 如有需要，给恶意客户端分配特定数据集
			clients.append(Client(conf, server.global_model, train_datasets, True, c))
		else:
			# 良性客户端
			clients.append(Client(conf, server.global_model, train_datasets, False, c))


	# for c in range(conf["num_models"]):
	# 	clients.append(Client(conf, server.global_model, train_datasets, c))



	# 随机选择客户端进行分组
	candidates = random.sample(clients, conf["k"])
	group_size = len(candidates) // num_client_groups
	for i in range(num_client_groups):
		client_groups.append(candidates[i * group_size:(i + 1) * group_size])

	# 将分组好的客户端分配至边缘服务器下
	for i in range(conf["num_edge_servers"]):
		edge_servers.append(EdgeServer(conf, i, server.global_model, client_groups[i]))

	# 将边缘服务器分配至云服务器下
	server.set_edge_servers(edge_servers)

	# for i in range(len(edge_servers)):
	# 	print("边缘服务器id", edge_servers[i].server_id, "该边缘服务器下客户端数量", len(edge_servers[i].clients))

	# 收集全局迭代过程中全局模型的测试准确率与损失，以便后续画图
	global_acc_list = []
	global_loss_list = []
	global_asr_list = []

	# 进行全局训练
	print("\n\n")
	for e in range(conf["global_epochs"]):
	
		# candidates = random.sample(clients, conf["k"])

		# 设置并初始化一系列列表与数组收集模型参数
		edge_weights_accumulator = dict()
		edge_weights_accumulator_list = []
		edge_aggregate_result = {}
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		if conf["attack_type"] == "a little enough attack":
			z_values = {3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891, 28: 0.48}
			malicious_params = []
			clients_params = {}
			diff = dict()
			diff_dic = dict()
			for i in range(conf["num_edge_servers"]):
				for c in edge_servers[i].clients:
					diff, local_params = c.local_train(edge_servers[i].global_model, c.client_id)
					diff_dic[c] = diff
					local_params_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_params.items()], dim=0)
					clients_params[c.client_id] = deepcopy(local_params_flatten.cpu())

			global_params = server.global_model.state_dict()
			original_params = torch.cat([param.data.clone().view(-1) for key, param in global_params.items()], dim=0).cpu()
			user_grads = []
			learning_rate = conf["lr"]

			for i in range(conf["num_edge_servers"]):
				for c in edge_servers[i].clients:
					if c.is_malicious:
						local_params = clients_params[c.client_id]
						local_grads = (original_params - local_params) / learning_rate
						user_grads =  local_grads[None, :] if len(user_grads) == 0 else torch.cat(
							(user_grads, local_grads[None, :]), 0)

			grads_mean = torch.mean(user_grads, dim=0)
			grads_stdev = torch.std(user_grads, dim=0)

			params_mean = original_params - learning_rate * grads_mean

			global_temp_params = OrderedDict()
			start_idx = 0

			for name, data in server.global_model.state_dict().items():
				param = params_mean[start_idx: start_idx + len(data.data.view(-1))].reshape(data.data.shape)
				start_idx = start_idx + len(data.data.view(-1))
				global_temp_params[name] = deepcopy(param)

			mal_client_params = {}
			user_params = []
			alpha = 0.2
			for i in range(conf["num_edge_servers"]):
				for c in edge_servers[i].clients:
					if c.is_malicious:
						local_temp_params = c.backdoor_attack_train(edge_servers[i].global_model, c.client_id, alpha)
						local_temp_params_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_temp_params.items()], dim=0).cpu()
						clients_params[c.client_id] = deepcopy(local_temp_params_flatten)
						user_params = local_temp_params_flatten[None, :] if len(user_params) == 0 else torch.cat(
							(user_params, local_temp_params_flatten[None, :]), 0
						)
			params_temp_mean = torch.mean(user_params, dim=0)

			new_params = params_temp_mean + learning_rate * grads_mean
			new_grads = (params_mean - new_params) / learning_rate

			num_std = z_values[28]
			new_user_grads = np.clip(new_grads, grads_mean - num_std * grads_stdev,
									 grads_mean + num_std * grads_stdev)

			mal_params = original_params - learning_rate * new_user_grads

			diff_temp = dict()
			mal_state_dict = dict()
			for i in range(conf["num_edge_servers"]):
				for name, params in server.global_model.state_dict().items():
					edge_weights_accumulator[name] = torch.zeros_like(params)
				for c in edge_servers[i].clients:
					if c.is_malicious:
						start_idx = 0
						for name, data in server.global_model.state_dict().items():
							param = mal_params[start_idx: start_idx + len(data.data.view(-1))].reshape(data.data.shape)
							param = param.to(device)
							start_idx = start_idx + len(data.data.view(-1))
							mal_state_dict[name] = deepcopy(param)
						c.local_model.load_state_dict(mal_state_dict, strict=True)

						for name, params in server.global_model.state_dict().items():
							diff_temp[name] = params - c.local_model.state_dict()[name]
							edge_weights_accumulator[name].add_(diff_temp[name])
						edge_weights_accumulator_list.append(edge_weights_accumulator)
					else:
						for name, params in server.global_model.state_dict().items():
							diff_temp[name] = params - c.local_model.state_dict()[name]
							edge_weights_accumulator[name].add_(diff_temp[name])
						edge_weights_accumulator_list.append(edge_weights_accumulator)
		else:
			for i in range(conf["num_edge_servers"]):
				for name, params in server.global_model.state_dict().items():
					edge_weights_accumulator[name] = torch.zeros_like(params)

				# 本地训练与攻击
				for c in edge_servers[i].clients:
					diff = dict()
					if conf["attack_type"] == "scaling attack":
						if c.is_malicious:
							diff = c.scaling_attack_train(edge_servers[i].global_model, c.client_id,
													  conf["num_models"], num_malicious_clients)
						else:
							diff, _ = c.local_train(edge_servers[i].global_model, c.client_id)

					elif conf["attack_type"] == "label flipping attack":
						if c.is_malicious:
							diff = c.label_flipping_attack_train(edge_servers[i].global_model, c.client_id,
																 conf["num_models"], num_malicious_clients)
						else:
							diff, _ = c.local_train(edge_servers[i].global_model, c.client_id)

					elif conf["attack_type"] == "gaussian attack":
						if c.is_malicious:
							diff = c.gaussian_attack_train(edge_servers[i].global_model, c.client_id,
														   conf["num_models"], num_malicious_clients)
						else:
							diff, _ = c.local_train(edge_servers[i].global_model, c.client_id)

					else:
						# conf["attack_type"] == "no attack":
						diff, _ = c.local_train(edge_servers[i].global_model, c.client_id)

					for name, params in server.global_model.state_dict().items():
						edge_weights_accumulator[name].add_(diff[name])

					edge_weights_accumulator_list.append(edge_weights_accumulator)

		# for c in candidates:
		# 	diff = c.local_train(server.global_model, c.client_id)
		#
		# 	for name, params in server.global_model.state_dict().items():
		# 		edge_weights_accumulator[name] = torch.zeros_like(params)
		#
		# 	for name, params in server.global_model.state_dict().items():
		# 		edge_weights_accumulator[name].add_(diff[name])
			
			# for name, params in server.global_model.state_dict().items():
			# 	weight_accumulator[name].add_(diff[name])

		weights_accumulator_list = []

		for i in range(conf["num_edge_servers"]):
			for name, params in server.global_model.state_dict().items():
				edge_aggregate_result[name] = torch.zeros_like(params)

			# print("边缘服务器", i, "聚合开始")
			weights_accumulator_list.append(edge_servers[i].edge_model_aggregate(edge_weights_accumulator_list[i], edge_aggregate_result))
			# print("边缘服务器", i, "聚合完成")
		
		server.model_aggregate(weights_accumulator_list)

		acc, loss = server.model_eval()

		global_acc_list.append(acc)
		global_loss_list.append(loss)
		

		if conf["attack_type"] == "scaling attack" or conf["attack_type"] == "a little enough attack":
			asr = server.ASR()
			global_asr_list.append(asr)
			print("Epoch %d, acc: %f, loss: %f, asr: %f\n" % (e, acc, loss, asr))
		else:
			print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

	# global model accuracy fig
	plt.figure()
	plt.plot(range(0, conf["global_epochs"]), global_acc_list, 'r', label='Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Global Model Accuracy')
	plt.legend()
	plt.savefig('./fig/acc_'+str(conf["type"])+"_"+str(conf["attack_type"])+".png")

	# global model loss fig
	plt.figure()
	plt.plot(range(0, conf["global_epochs"]), global_loss_list, 'b', label='Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Global Model Loss')
	plt.legend()
	plt.savefig('./fig/loss_' + str(conf["type"]) + "_" + str(conf["attack_type"]) + ".png")
	plt.show()

	# global asr fig
	if conf["attack_type"] == "scaling attack" or conf["attack_type"] == "a little enough attack":
		plt.figure()
		plt.plot(range(0, conf["global_epochs"]), global_asr_list, 'g', label='ASR')
		plt.xlabel('Epoch')
		plt.ylabel('ASR')
		plt.title('ASR')
		plt.legend()
		plt.savefig('./fig/asr_' + str(conf["type"]) + "_" + str(conf["attack_type"]) + ".png")
		plt.show()