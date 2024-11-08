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

    def set_global_model(self, global_model):
        self.global_model = global_model

    # def model_aggregate(self, weight_accumulator):
    #     for name, data in self.global_model.state_dict().items():
    #
    #         update_per_layer = weight_accumulator[name] * self.conf["lambda"]
    #
    #         if data.type() != update_per_layer.type():
    #             data.add_(update_per_layer.to(torch.int64))
    #         else:
    #             data.add_(update_per_layer)
    """
    初始化边缘服务器，接收从云服务器传来的初始全局模型。

    :param edge_weight_accumulator: 边缘设备权重
    :param edge_aggregate_result: 边缘服务器局部聚合结果
    """
    def edge_model_aggregate(self, edge_weight_accumulator, edge_aggregate_result):
        for name, data in edge_weight_accumulator.items():
            edge_aggregate_result[name].add_((edge_weight_accumulator[name] * self.conf["edge_lambda"]).to(torch.int64))

        return edge_aggregate_result