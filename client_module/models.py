__author__ = 'yang.xu'

import re

import torch.nn.functional as F
import torch.nn as nn

# <--For Sent140
'''
an automatically generated sentiment analysis dataset that annotates tweets
based on the emoticons present in them. Each device is a different twitter user
'''


class Sent140_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        pass

    def forward(self, x):
        pass
        return None


# <--For CIFAR10

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CIFAR10_Deep_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Deep_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# <--For FashionMNIST & MNIST
class MNIST_Small_Net(nn.Module):
    def __init__(self):
        super(MNIST_Small_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 32, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 32)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNIST_LR_Net(nn.Module):
    def __init__(self):
        super(MNIST_LR_Net, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x), inplace=True)
        x = F.relu(self.hidden2(x), inplace=True)
        x = self.out(x)
        return F.log_softmax(x, dim=1)


def create_model_instance(dataset_type, model_type, device):
    if dataset_type == 'FashionMNIST':
        if model_type == 'LR':
            model = MNIST_LR_Net().to(device)
        else:
            model = MNIST_Net().to(device)

    elif dataset_type == 'MNIST':
        if model_type == 'LR':
            model = MNIST_LR_Net().to(device)
        else:
            model = MNIST_Small_Net().to(device)

    elif dataset_type == 'CIFAR10':

        if model_type == 'Deep':
            model = CIFAR10_Deep_Net().to(device)
            decay_rate = 0.98
        else:
            model = CIFAR10_Net().to(device)
            decay_rate = 0.98

    elif dataset_type == 'Sent140':

        if model_type == 'LSTM':
            model = Sent140_Net().to(device)
            decay_rate = 0.99
        else:
            model = Sent140_Net().to(device)
            decay_rate = 0.99

    return model


# 从模型到列表
def Seq2Tup(sequen):
    lst = []
    for tup in sequen._modules.items():
        lst.append(tup)
    return lst


def ExtractParam(string, num):
    str_num_list = re.findall(r'[0-9]+\.?[0-9]*', string[string.find('('):])
    if (num == 1):
        return float(str_num_list[0])
    num_list = [0] * num
    idx1 = 0
    while (idx1 < num):
        if (num == 5 and idx1 == 2):
            if (str_num_list[2] == str_num_list[3]):
                num_list[2] = int(str_num_list[2])
            else:
                num_list[2] = int(str_num_list[2]), int(str_num_list[3])
            if (str_num_list[4] == str_num_list[5]):
                num_list[3] = int(str_num_list[4])
            else:
                num_list[3] = int(str_num_list[4]), int(str_num_list[5])
            if (len(str_num_list) == 8):
                if (str_num_list[6] == str_num_list[7]):
                    num_list[4] = int(str_num_list[7])
                else:
                    num_list[4] = int(str_num_list[6]), int(str_num_list[7])
            idx1 = 5
        else:
            num_list[idx1] = int(str_num_list[idx1])
            idx1 += 1
    return tuple(num_list)


def FunType(string):
    if string.find('BatchNorm') != -1:
        return 'Batchnorm'
    elif string.find('ReLU') != -1:
        return 'Relu'
    elif string.find('Sigmoid') != -1:
        return 'Sigmoid'
    elif string.find('MaxPool') != -1:
        return ExtractParam(string, 3)
    elif string.find('Dropout') != -1:
        return ExtractParam(string, 1)
    elif string.find('Softmax') != -1:
        return 'Softmax'


def Net2Tuple(net):
    tmp = nn.Sequential()
    net_list = []
    net_param_list = []
    for item in net._modules.items():
        if (isinstance(item[1], type(tmp))):
            net_list += Seq2Tup(item[1])
        else:
            net_list.append(item)

    idx = 0
    while (idx < len(net_list)):
        layer = []
        layer.append(net_list[idx][0])
        tostr = str(net_list[idx][1])
        if (tostr.find('Conv') != -1):
            layer.append('Conv')
            idx += 1
            layer.append(ExtractParam(tostr, 5))
            while (idx < len(net_list) and str(net_list[idx][1]).find('Linear') == -1 and str(net_list[idx][1]).find(
                    'Conv') == -1):
                if (str(net_list[idx][1]).find('MaxPool') != -1):
                    layer.append('Maxpool')
                layer.append(FunType(str(net_list[idx][1])))
                idx += 1
            net_param_list.append(tuple(layer))
        else:
            layer.append('FC')
            idx += 1
            layer.append(ExtractParam(tostr, 2))
            while (idx < len(net_list) and str(net_list[idx][1]).find('Linear') == -1):
                if (str(net_list[idx][1]).find('Dropout') != -1):
                    layer.append('Dropout')
                layer.append(FunType(str(net_list[idx][1])))
                idx += 1
            net_param_list.append(tuple(layer))
    return net_param_list
