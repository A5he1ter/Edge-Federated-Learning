import torch
import copy

device = None
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class EdgeServer:
    """
    初始化边缘服务器，接收从云服务器传来的初始全局模型。

    :param server_id: 边缘服务器的ID
    :param clients: 客户端列表
    :param initial_model: 从云服务器获取的初始全局模型
    """
    def __init__(self, conf, server_id, global_model, clients, eval_dataset):
        self.conf = conf
        self.server_id = server_id
        self.global_model = global_model
        self.clients = clients
        self.local_params_list = {}
        self.num_malicious = 0
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def set_global_model(self, global_model):
        self.global_model.load_state_dict(global_model.state_dict())

    def edge_model_aggregate(self):
        user_grads = []
        edge_global_params = {}
        for c in self.local_params_list:
            local_params = self.local_params_list[c]
            user_grads = local_params[None, :] if len(user_grads) == 0 else torch.cat(
                (user_grads, local_params[None, :]), dim=0
            )

        avg_params = torch.mean(user_grads, dim=0)

        idx = 0
        for key, val in self.global_model.state_dict().items():
            param = avg_params[idx: idx + len(val.data.view(-1))].reshape(val.data.shape)
            idx = idx + len(val.data.view(-1))
            edge_global_params[key] = copy.deepcopy(param)

        return avg_params

    def set_local_params(self, local_params, client_id):
        self.local_params_list[client_id] = local_params

    def get_local_params(self):
        return self.local_params_list

    def know_num_malicious(self):
        for c in self.clients:
            if c.is_malicious:
                self.num_malicious += 1

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