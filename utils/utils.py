import torch

device = None
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def Adding_Trigger(data):
    if data.shape[0] == 3:
        for i in range(3):
            data[i][1][28] = 1
            data[i][1][29] = 1
            data[i][1][30] = 1
            data[i][2][29] = 1
            data[i][3][28] = 1
            data[i][4][29] = 1
            data[i][5][28] = 1
            data[i][5][29] = 1
            data[i][5][30] = 1

    if data.shape[0] == 1:
        data[0][1][24] = 1
        data[0][1][25] = 1
        data[0][1][26] = 1
        data[0][2][24] = 1
        data[0][3][25] = 1
        data[0][4][26] = 1
        data[0][5][24] = 1
        data[0][5][25] = 1
        data[0][5][26] = 1
    return data

def euclidean_clients(param_matrix):
    dev = device
    param_tf = torch.FloatTensor(param_matrix).to(dev)
    output = torch.cdist(param_tf, param_tf, p=2)

    return output.tolist()

def eval_defense_acc(clients, malicious_clients, detect_malicious_client):
    # d_m_c = []
    # m_c = []
    #
    # for c in clients:
    #     d_m_c.append(1) if c in detect_malicious_client else d_m_c.append(0)
    #     m_c.append(1) if c in malicious_clients else m_c.append(0)
    #
    # count = 0
    # for key, val in enumerate(d_m_c):
    #     if m_c[key] == val:
    #         count += 1

    count = 0
    for c in clients:
        if (c in detect_malicious_client) == (c in malicious_clients):
            count += 1

    defense_acc = count / len(clients)

    count1 = 0
    for c in malicious_clients:
        if c in detect_malicious_client:
            count1 += 1

    if len(malicious_clients) != 0:
        malicious_precision = count1 / len(malicious_clients)
    else:
        malicious_precision = 1

    count2 = 0
    for c in detect_malicious_client:
        if c in malicious_clients:
            count2 += 1

    if len(detect_malicious_client) != 0:
        malicious_recall = count2 / len(detect_malicious_client)
    else:
        malicious_recall = 0

    return defense_acc, malicious_precision, malicious_recall