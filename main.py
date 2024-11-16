from copy import deepcopy

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

	# 创建客户端，并标记恶意客户端
	if conf["attack_type"] == "no attack":
		print("num of edge servers:", num_client_groups)
		print("num of clients:", conf["num_models"])
		print("num of global epochs:", conf["global_epochs"])
		print("device:", device)
		print("attack type:", conf["attack_type"])
		print("detect type:", conf["detect_type"])
		print("dataset:", conf["type"])
		for c in range(conf["num_models"]):
			clients.append(Client(conf, server.global_model, train_datasets, eval_datasets, False, c))
	else:
		num_malicious_clients = int(conf["malicious_ratio"] * conf["num_models"])
		print("num of edge servers:", num_client_groups)
		print("num of clients:", conf["num_models"])
		print("num of malicious clients:", num_malicious_clients)
		malicious_clients = random.sample(range(conf["num_models"]), num_malicious_clients)
		print("malicious clients:", malicious_clients)
		print("num of global epochs:", conf["global_epochs"])
		print("device:", device)
		print("attack type:", conf["attack_type"])
		print("detect type:", conf["detect_type"])
		print("dataset:", conf["type"])
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

	for i in range(conf["num_edge_servers"]):
		edge_servers[i].know_num_malicious()

	# 将边缘服务器分配至云服务器下
	server.set_edge_servers(edge_servers)

	# for i in range(len(edge_servers)):
	# 	print("边缘服务器id", edge_servers[i].server_id, "该边缘服务器下客户端数量", len(edge_servers[i].clients))



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
		# candidates = random.sample(clients, conf["k"])

		# 设置并初始化一系列列表与数组收集模型参数
		edge_weights_accumulator = dict()
		edge_weights_accumulator_list = []
		weights_accumulator_list = []

		if conf["attack_type"] == "a little enough attack":
			client_params = LIE_attack(edge_servers, conf["lr"], server.global_model)
			mal_state_dict = {}
			benign_state_dict = {}
			for i in range(conf["num_edge_servers"]):
				for name, params in server.global_model.state_dict().items():
					edge_weights_accumulator[name] = torch.zeros_like(params)

				for c in edge_servers[i].clients:
					diff_temp = dict()
					if c.is_malicious:
						mal_params = client_params[c.client_id]
						start_idx = 0
						for name, data in c.local_model.state_dict().items():
							param = mal_params[start_idx: start_idx + len(data.data.view(-1))].reshape(data.data.shape)
							start_idx = start_idx + len(data.data.view(-1))
							mal_state_dict[name] = deepcopy(param)

						c.local_model.load_state_dict(mal_state_dict, strict=True)

						# local_acc, local_loss = server.eval(c.local_model)
						# local_asr = server.ASR(c.local_model)
						# print('[LIT_attack %s] accuracy: %f  loss: %f asr: %f' % (c.client_id, local_acc, local_loss, local_asr))

						for name, params in edge_servers[i].global_model.state_dict().items():
							diff_temp[name] = (c.local_model.state_dict()[name] - params)
							edge_weights_accumulator[name].add_(diff_temp[name])
					else:
						benign_params = client_params[c.client_id]
						start_idx = 0
						for name, data in c.local_model.state_dict().items():
							param = benign_params[start_idx: start_idx + len(data.data.view(-1))].reshape(data.data.shape)
							start_idx = start_idx + len(data.data.view(-1))
							benign_state_dict[name] = deepcopy(param)
						c.local_model.load_state_dict(benign_state_dict, strict=True)

						for name, params in edge_servers[i].global_model.state_dict().items():
							diff_temp[name] = (c.local_model.state_dict()[name] - params)
							edge_weights_accumulator[name].add_(diff_temp[name])

				edge_weights_accumulator_list.append(edge_weights_accumulator)
		else:
			for i in range(conf["num_edge_servers"]):
				# with tqdm(edge_servers[i].clients, total=len(edge_servers[i].clients), desc=f"Edge Server {i+1}/{conf['num_edge_servers']}", ncols=120) as pbar:
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
							local_params = c.label_flipping_attack_train(edge_servers[i].global_model, c.client_id,
																 conf["num_models"], num_malicious_clients)
						else:
							local_params = c.local_train(edge_servers[i].global_model, c.client_id)

					elif conf["attack_type"] == "random label flipping attack":
						if c.is_malicious:
							local_params = c.random_label_flipping_attack_train(edge_servers[i].global_model, c.client_id)
						else:
							local_params = c.local_train(edge_servers[i].global_model, c.client_id)

					elif conf["attack_type"] == "gaussian attack":
						if c.is_malicious:
							local_params = c.gaussian_attack_train(edge_servers[i].global_model, c.client_id,
														   conf["num_models"], num_malicious_clients)
						else:
							local_params = c.local_train(edge_servers[i].global_model, c.client_id)

					else:
						# conf["attack_type"] == "no attack":
						local_params = c.local_train(edge_servers[i].global_model, c.client_id)

					local_params_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_params.items()], dim=0)
					edge_servers[i].set_local_params(local_params_flatten.cpu(), c.client_id)

				if conf["detect_type"] == "multi krum":
					sub_global_params_flatten, detected_malicious = multi_krum(edge_servers[i])
					detect_malicious_client += detected_malicious
				else:
					sub_global_params_flatten = edge_servers[i].edge_model_aggregate()
				weights_accumulator_list.append(sub_global_params_flatten)
				edge_servers[i].local_params_list = {}

		server.global_model.load_state_dict(server.model_aggregate(weights_accumulator_list))

		for i in range(conf["num_edge_servers"]):
			edge_servers[i].set_global_model(server.global_model)

		acc, loss = server.model_eval()

		global_acc_list.append(acc)
		global_loss_list.append(loss)


		if conf["attack_type"] == "scaling attack" or conf["attack_type"] == "a little enough attack":
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

	# global model accuracy fig
	axes[0, 0].plot(range(len(global_acc_list)), global_acc_list, 'r-o', label='Accuracy')
	axes[0, 0].set_xlabel('Epoch')
	axes[0, 0].set_ylabel('Accuracy')
	axes[0, 0].set_title('Global Model Accuracy')
	axes[0, 0].legend()
	# plt.savefig('./fig/acc '+str(conf["type"])+" "+str(conf["attack_type"])+" "+str(conf["detect_type"])+".png")

	# global model loss fig
	axes[0, 1].plot(range(len(global_loss_list)), global_loss_list, 'b-o', label='Loss')
	axes[0, 1].set_xlabel('Epoch')
	axes[0, 1].set_ylabel('Loss')
	axes[0, 1].set_title('Global Model Loss')
	axes[0, 1].legend()
	# plt.savefig('./fig/loss ' + str(conf["type"]) + " " + str(conf["attack_type"]) + str(conf["detect_type"]) + ".png")

	# global asr fig
	if conf["attack_type"] == "scaling attack" or conf["attack_type"] == "a little enough attack":
		axes[0, 2].plot(range(len(global_asr_list)), global_asr_list, 'g-o', label='ASR')
		axes[0, 2].set_xlabel('Epoch')
		axes[0, 2].set_ylabel('ASR')
		axes[0, 2].set_title('ASR')
		axes[0, 2].legend()
		# plt.savefig('./fig/asr ' + str(conf["type"]) + " " + str(conf["attack_type"]) + str(conf["detect_type"])+ ".png")

	if conf["attack_type"] != "no attack":
		axes[1, 0].plot(range(len(defense_acc_list)), defense_acc_list, 'y-o', label='Defense Accuracy')
		axes[1, 0].set_xlabel('Epoch')
		axes[1, 0].set_ylabel('Defense Accuracy')
		axes[1, 0].set_title('Defense Accuracy')
		axes[1, 0].legend()

		axes[1, 1].plot(range(len(malicious_precision_list)), malicious_precision_list, 'k-o', label='Malicious Precision')
		axes[1, 1].set_xlabel('Epoch')
		axes[1, 1].set_ylabel('Malicious Precision')
		axes[1, 1].set_title('Malicious Precision')
		axes[1, 1].legend()

		axes[1, 2].plot(range(len(malicious_recall_list)), malicious_recall_list, 'm-o', label='Malicious Recall')
		axes[1, 2].set_xlabel('Epoch')
		axes[1, 2].set_ylabel('Malicious Recall')
		axes[1, 2].set_title('Malicious Recall')
		axes[1, 2].legend()

	plt.tight_layout(pad=0.1)

	plt.savefig('./fig/' + conf["type"] + " " + conf["attack_type"] + " " + conf["detect_type"] + ".png")

	plt.show()