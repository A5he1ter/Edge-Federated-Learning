import torch
from copy import deepcopy
from collections import OrderedDict
import numpy as np

def LIE_attack(edge_servers, lr, global_model):
    z_values = {3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891, 28: 0.48}
    clients_params = {}
    for i in range(len(edge_servers)):
        for c in edge_servers[i].clients:
            _, local_params = c.local_train(edge_servers[i].global_model, c.client_id)
            local_params_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_params.items()],
                                             dim=0)
            clients_params[c.client_id] = deepcopy(local_params_flatten.cpu())

    global_params = global_model.state_dict()
    original_params = torch.cat([param.data.clone().view(-1) for key, param in global_params.items()], dim=0).cpu()
    user_grads = []
    learning_rate = lr

    for i in range(len(edge_servers)):
        for c in edge_servers[i].clients:
            if c.is_malicious:
                local_params = clients_params[c.client_id]
                local_grads = (original_params - local_params) / learning_rate
                user_grads = local_grads[None, :] if len(user_grads) == 0 else torch.cat(
                    (user_grads, local_grads[None, :]), 0)

    grads_mean = torch.mean(user_grads, dim=0)
    grads_stdev = torch.std(user_grads, dim=0)

    params_mean = original_params - learning_rate * grads_mean

    mal_model_params = LIE_model_train(global_model, edge_servers, global_params)

    new_params = mal_model_params + learning_rate * grads_mean
    new_grads = (params_mean - new_params) / learning_rate

    num_std = z_values[28]
    new_user_grads = np.clip(new_grads, grads_mean - num_std * grads_stdev,
                             grads_mean + num_std * grads_stdev)

    mal_params = original_params - learning_rate * new_user_grads

    for i in range(len(edge_servers)):
        for c in edge_servers[i].clients:
            if c.is_malicious:
                clients_params[c.client_id] = deepcopy(mal_params)

    return clients_params

def LIE_model_train(global_model, edge_servers, params_mean):
    # global_params = OrderedDict()
    start_idx = 0

    # for name, data in global_model.state_dict().items():
    #     param = params_mean[start_idx: start_idx + len(data.data.view(-1))].reshape(data.data.shape)
    #     start_idx = start_idx + len(data.data.view(-1))
    #     global_params[name] = deepcopy(param)

    clients_params = {}
    user_params = []
    alpha = 0.2
    for i in range(len(edge_servers)):
        for c in edge_servers[i].clients:
            if c.is_malicious:
                local_params = c.backdoor_attack_train(params_mean, c.client_id, alpha)
                local_params_flatten = torch.cat(
                    [param.data.clone().view(-1) for key, param in local_params.items()], dim=0).cpu()
                clients_params[c.client_id] = deepcopy(local_params_flatten)
                user_params = local_params_flatten[None, :] if len(user_params) == 0 else torch.cat(
                    (user_params, local_params_flatten[None, :]), 0
                )
    params_mean = torch.mean(user_params, dim=0)
    return params_mean