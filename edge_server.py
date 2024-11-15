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
        self.local_params_list = {}
        self.num_malicious = 0

    def set_global_model(self, global_model):
        self.global_model.load_state_dict(global_model.state_dict())

    def edge_model_aggregate(self):
        user_grads = []
        for c in self.local_params_list:
            local_params = self.local_params_list[c]
            local_params_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_params.items()],
                                             dim=0)
            user_grads = local_params_flatten[None, :] if len(user_grads) == 0 else torch.cat(
                (user_grads, local_params_flatten[None, :]), dim=0
            )

        avg_params = torch.mean(user_grads, dim=0)

        return avg_params

    def set_local_params(self, local_params, client_id):
        self.local_params_list[client_id] = local_params

    def get_local_params(self):
        return self.local_params_list

    def know_num_malicious(self):
        for c in self.clients:
            if c.is_malicious:
                self.num_malicious += 1