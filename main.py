import argparse, json
import datetime
import os
import logging
import torch, random
import matplotlib.pyplot as plt

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

	for c in range(conf["num_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))

	candidates = random.sample(clients, conf["k"])

	# group_size = len(clients) // num_client_groups
	group_size = len(candidates) // num_client_groups

	for i in range(num_client_groups):
		client_groups.append(candidates[i * group_size:(i + 1) * group_size])

	for i in range(conf["num_edge_servers"]):
		edge_servers.append(EdgeServer(conf, i, server.global_model, client_groups[i]))

	server.set_edge_servers(edge_servers)

	# for i in range(len(edge_servers)):
	# 	print("边缘服务器id", edge_servers[i].server_id, "该边缘服务器下客户端数量", len(edge_servers[i].clients))

	global_acc_list = []
	global_loss_list = []

	print("\n\n")
	for e in range(conf["global_epochs"]):
	
		# candidates = random.sample(clients, conf["k"])

		edge_weights_accumulator = {}
		edge_weights_accumulator_list = []
		edge_aggregate_result = {}
		weight_accumulator = {}

		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		for i in range(conf["num_edge_servers"]):
			for name, params in server.global_model.state_dict().items():
				edge_weights_accumulator[name] = torch.zeros_like(params)

			for c in edge_servers[i].clients:
				diff = c.local_train(edge_servers[i].global_model, c.client_id)

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
			print("边缘服务器", i, "聚合完成")
		
		server.model_aggregate(weights_accumulator_list)

		acc, loss = server.model_eval()

		global_acc_list.append(acc)
		global_loss_list.append(loss)
		
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

	# global model accuracy fig
	plt.figure()
	plt.plot(range(0, conf["global_epochs"]), global_acc_list, 'r', label='Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Global Model Accuracy')
	plt.legend()
	plt.savefig('./fig/acc_femnist.png')

	# global model loss fig
	plt.figure()
	plt.plot(range(0, conf["global_epochs"]), global_loss_list, 'b', label='Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Global Model Loss')
	plt.legend()
	plt.savefig('./fig/loss_femnist.png')
	plt.show()