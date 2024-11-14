import torch

class EdgeServer:
    """
    初始化边缘服务器，接收从云服务器传来的初始全局模型。

    :param server_id: 边缘服务器的ID
    :param clients: 客户端列表
    :param initial_model: 从云服务器获取的初始全局模型
    """
    def __init__(self, conf, server_id, global_model, clients):
        self.conf = conf
        self.server_id = server_id
        self.global_model = global_model
        self.clients = clients
        self.clients_diff_list = {}
        self.num_malicious = 0

    def set_global_model(self, global_model):
        self.global_model = global_model

    def edge_model_aggregate(self, edge_weight_accumulator, edge_aggregate_result):
        for key, diff in self.clients_diff_list.items():
            for name, params in diff.items():
                edge_aggregate_result[name].add_(diff[name] * (1 / len(self.clients_diff_list)))

        return edge_aggregate_result

    def collect_clients_diff(self, diff, client_id):
        self.clients_diff_list[client_id] = diff

    def get_clients_diff_list(self):
        return self.clients_diff_list

    def know_num_malicious(self):
        for c in self.clients:
            if c.is_malicious:
                self.num_malicious += 1