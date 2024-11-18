from copy import deepcopy

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sympy.codegen.ast import float32

from LIE_attack import LIE_attack
from cloud_server import *
from edge_server import *
from client import *
import models, datasets
from tqdm import tqdm
from detect import multi_krum
from utils.utils import eval_defense_acc

device = None
if torch.backends.mps.is_available():
	device = torch.device('mps')
elif torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

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
	num_malicious_clients = 0

	print("num of edge servers:", num_client_groups)
	print("num of clients:", conf["num_models"])
	print("num of global epochs:", conf["global_epochs"])
	print("device:", device)
	print("dataset:", conf["type"])
	# 创建客户端，并标记恶意客户端
	if conf["attack_type"] == "no attack":
		for c in range(conf["num_models"]):
			clients.append(Client(conf, server.global_model, train_datasets, eval_datasets, False, c))
	else:
		num_malicious_clients = int(conf["malicious_ratio"] * conf["num_models"])
		print("num of malicious clients:", num_malicious_clients)
		malicious_clients = random.sample(range(conf["num_models"]), num_malicious_clients)
		print("malicious clients:", malicious_clients)
		print("attack type:", conf["attack_type"])
		print("detect type:", conf["detect_type"])
		for c in range(conf["num_models"]):
			if c in malicious_clients:
				# TODO 如有需要，给恶意客户端分配特定数据集
				clients.append(Client(conf, server.global_model, train_datasets, eval_datasets, True, c))
			else:
				# 良性客户端
				clients.append(Client(conf, server.global_model, train_datasets, eval_datasets, False, c))

		malicious_clients_list = []
		for c in clients:
			if c.is_malicious:
				malicious_clients_list.append(c)

	# 随机选择客户端进行分组
	candidates = random.sample(clients, conf["k"])
	group_size = len(candidates) // num_client_groups
	for i in range(num_client_groups):
		client_groups.append(candidates[i * group_size:(i + 1) * group_size])

	# 将分组好的客户端分配至边缘服务器下
	for i in range(conf["num_edge_servers"]):
		edge_servers.append(EdgeServer(conf, i, server.global_model, client_groups[i], eval_datasets))

	for i in range(conf["num_edge_servers"]):
		edge_servers[i].know_num_malicious()

	# 将边缘服务器分配至云服务器下
	server.set_edge_servers(edge_servers)

	# 收集全局迭代过程中全局模型的测试准确率与损失，以便后续画图
	global_acc_list = []
	global_loss_list = []
	global_asr_list = []

	defense_acc_list = []
	malicious_precision_list = []
	malicious_recall_list = []

	# 进行全局训练
	print("\n\n")
	for e in range(conf["global_epochs"]):
		detect_malicious_client = []

		# 设置并初始化一系列列表与数组收集模型参数
		edge_weights_accumulator = dict()
		edge_weights_accumulator_list = []
		weights_accumulator_list = []

		for i in range(conf["num_edge_servers"]):
			# 本地训练与攻击
			for c in edge_servers[i].clients:
				local_params = None
				if conf["attack_type"] == "scaling attack":
					if c.is_malicious:
						local_params = c.scaling_attack_train(edge_servers[i].global_model, c.client_id,
												  conf["num_models"], num_malicious_clients)
					else:
						local_params = c.local_train(edge_servers[i].global_model, c.client_id)

				elif conf["attack_type"] == "label flipping attack":
					if c.is_malicious:
						local_params = c.label_flipping_attack_train(edge_servers[i].global_model, c.client_id)
					else:
						local_params = c.local_train(edge_servers[i].global_model, c.client_id)

				elif conf["attack_type"] == "random label flipping attack":
					if c.is_malicious:
						local_params = c.random_label_flipping_attack_train(edge_servers[i].global_model, c.client_id)
					else:
						local_params = c.local_train(edge_servers[i].global_model, c.client_id)

				elif conf["attack_type"] == "mixed attack":
					if c.is_malicious:
						attack_func = [c.label_flipping_attack_train, c.random_label_flipping_attack_train]
						local_params = random.choice(attack_func)(edge_servers[i].global_model, c.client_id,
																  conf["num_models"], num_malicious_clients)

					else:
						local_params = c.local_train(edge_servers[i].global_model, c.client_id)

				elif conf["attack_type"] == "gaussian attack":
					if c.is_malicious:
						local_params = c.gaussian_attack_train(edge_servers[i].global_model, c.client_id,
													   conf["num_models"], num_malicious_clients)
					else:
						local_params = c.local_train(edge_servers[i].global_model, c.client_id)

				else:
					local_params = c.local_train(edge_servers[i].global_model, c.client_id)

				local_params_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_params.items()], dim=0)
				edge_servers[i].set_local_params(local_params_flatten.cpu(), c.client_id)

			if conf["detect_type"] == "multi krum":
				sub_global_params_flatten, detected_malicious = multi_krum(edge_servers[i])
				detect_malicious_client += detected_malicious
			else:
				sub_global_params_flatten = edge_servers[i].edge_model_aggregate()
				print("edge", i, "acc: %f, loss: %f" % (edge_servers[i].TACC()))

			weights_accumulator_list.append(sub_global_params_flatten)
			edge_servers[i].local_params_list.clear()

		server.global_model.load_state_dict(server.model_aggregate(weights_accumulator_list), strict=True)

		for i in range(conf["num_edge_servers"]):
			edge_servers[i].set_global_model(server.global_model)

		acc, loss = server.TACC()

		global_acc_list.append(acc)
		global_loss_list.append(loss)


		if conf["attack_type"] == "scaling attack" or conf["attack_type"] == "a little enough attack" or conf["attack_type"] == "mixed attack":
			asr = server.ASR(server.global_model)
			global_asr_list.append(asr)
			print("Epoch %d acc: %f loss: %f asr: %f" % (e, acc, loss, asr))
		else:
			print("Epoch %d acc: %f loss: %f" % (e, acc, loss))

		if conf["attack_type"] != "no attack":
			defense_acc, malicious_precision, malicious_recall = eval_defense_acc(clients, malicious_clients_list, detect_malicious_client)
			defense_acc_list.append(defense_acc)
			malicious_precision_list.append(malicious_precision)
			malicious_recall_list.append(malicious_recall)
			print("defense acc: %f malicious precision: %f malicious recall: %f" % (defense_acc, malicious_precision, malicious_recall))

		for i in range(conf["num_edge_servers"]):
			edge_servers[i].set_global_model(server.global_model)

	fig, axes = plt.subplots(2, 3, figsize=(18, 12))

	highlight_indices = np.array([0, 24, 49, 74, 99])
	epoch_list = np.array(range(len(global_acc_list)))
	global_acc_list = np.array(global_acc_list)
	global_loss_list = np.array(global_loss_list)
	global_asr_list = np.array(global_asr_list)
	defense_acc_list = np.array(defense_acc_list)
	malicious_precision_list = np.array(malicious_precision_list)
	malicious_recall_list = np.array(malicious_recall_list)

	# global model accuracy fig
	highlight_y = global_acc_list[highlight_indices]
	highlight_x = epoch_list[highlight_indices]
	axes[0, 0].plot(epoch_list, global_acc_list, "red", label='Accuracy')
	axes[0, 0].scatter(highlight_x, highlight_y, color="red", label='Global Accuracy')
	axes[0, 0].set_xlabel('Epoch')
	axes[0, 0].set_ylabel('Accuracy')
	axes[0, 0].set_title('Global Model Accuracy')
	axes[0, 0].legend()

	# global model loss fig
	highlight_y = global_loss_list[highlight_indices]
	highlight_x = epoch_list[highlight_indices]
	axes[0, 1].plot(epoch_list, global_loss_list, "blue", label='Loss')
	axes[0, 1].scatter(highlight_x, highlight_y, color="blue", label='Global Loss')
	axes[0, 1].set_xlabel('Epoch')
	axes[0, 1].set_ylabel('Loss')
	axes[0, 1].set_title('Global Model Loss')
	axes[0, 1].legend()

	# global asr fig
	if conf["attack_type"] == "scaling attack" or conf["attack_type"] == "a little enough attack" or conf["attack_type"] == "mixed attack":
		highlight_y = global_asr_list[highlight_indices]
		highlight_x = epoch_list[highlight_indices]
		axes[0, 2].plot(epoch_list, global_asr_list, "green", label='ASR')
		axes[0, 2].scatter(highlight_x, highlight_y, color="green", label='ASR')
		axes[0, 2].set_xlabel('Epoch')
		axes[0, 2].set_ylabel('ASR')
		axes[0, 2].set_title('ASR')
		axes[0, 2].legend()

	if conf["attack_type"] != "no attack":
		highlight_y = defense_acc_list[highlight_indices]
		highlight_x = epoch_list[highlight_indices]
		axes[1, 0].plot(epoch_list, defense_acc_list, "yellow", label='Defense Accuracy')
		axes[1, 0].scatter(highlight_x, highlight_y, color="yellow", label='Defense Accuracy')
		axes[1, 0].set_xlabel('Epoch')
		axes[1, 0].set_ylabel('Defense Accuracy')
		axes[1, 0].set_title('Defense Accuracy')
		axes[1, 0].legend()

		highlight_y = malicious_precision_list[highlight_indices]
		highlight_x = epoch_list[highlight_indices]
		axes[1, 1].plot(epoch_list, malicious_precision_list, "black", label='Malicious Precision')
		axes[1, 1].scatter(highlight_x, highlight_y, color="black", label='Malicious Precision')
		axes[1, 1].set_xlabel('Epoch')
		axes[1, 1].set_ylabel('Malicious Precision')
		axes[1, 1].set_title('Malicious Precision')
		axes[1, 1].legend()

		highlight_y = malicious_recall_list[highlight_indices]
		highlight_x = epoch_list[highlight_indices]
		axes[1, 2].plot(epoch_list, malicious_recall_list, "purple", label='Malicious Recall')
		axes[1, 2].scatter(highlight_x, highlight_y, color="purple", label='Malicious Recall')
		axes[1, 2].set_xlabel('Epoch')
		axes[1, 2].set_ylabel('Malicious Recall')
		axes[1, 2].set_title('Malicious Recall')
		axes[1, 2].legend()

	plt.tight_layout(pad=0.1)

	plt.savefig('./fig/' + conf["type"] + " " + conf["attack_type"] + " " + conf["detect_type"] + ".png")

	plt.show()