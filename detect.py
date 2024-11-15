import numpy as np
import torch
from sklearn.cluster import KMeans

from utils.utils import euclidean_clients


def multi_krum(edge_server):
    user_grads = []
    clients = []
    diff_list = edge_server.get_clients_diff_list()
    clients_list = edge_server.clients
    num_malicious = edge_server.num_malicious

    for i in clients_list:
        clients.append(i)
        local_params_update = diff_list[i.client_id]
        local_flatten_params = torch.cat([param.data.clone().view(-1) for key, param in local_params_update.items()], dim=0).cpu()
        user_grads = local_flatten_params[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_flatten_params[None, :]), 0
        )

    euclidean_matrix = euclidean_clients(user_grads)

    scores = []
    for list in euclidean_matrix:
        client_dis = sorted(list)
        client_dis1 = client_dis[1: len(clients_list) - num_malicious]
        score = np.sum(np.array(client_dis1))
        scores.append(score)
    client_scores = dict(zip(clients, scores))
    client_scores = sorted(client_scores.items(), key=lambda d: d[1], reverse=False)

    benign_client = client_scores[: len(clients_list) - num_malicious]
    benign_client = [i for i, val in benign_client]

    malicious_client = client_scores[len(clients_list) - num_malicious: ]
    malicious_client = [i for i, val in malicious_client]

    benign_client_params = {}
    for i in clients_list:
        if i in benign_client:
            benign_client_params[i.client_id] = diff_list[i.client_id]

    return benign_client_params, malicious_client