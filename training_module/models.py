__author__ = 'yang.xu'

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