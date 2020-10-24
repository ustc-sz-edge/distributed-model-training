import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist

from functools import singledispatch
import numpy as np
from multiprocessing import (
    Process,
    Pool
)
import os
import paramiko

import fl_utils
from config_module.config import *

import rpyc

def main():
    # Init global paramter
    config = LocalConfig()

    device = torch.device("cuda" if config.common.use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if config.common.use_cuda else {}

    if config.common.training_structure == 'local':
        config = LocalConfig()

    """Create model, dataset, etc


    """

    # Create federated model instance
    model = fl_models.create_model_instance(
        global_config.dataset_type, global_config.model_type, device)

    # Create dataset instance
    train_dataset, test_dataset = fl_datasets.load_datasets(
        global_config.dataset_type)
    test_loader = fl_utils.create_server_test_loader(
        global_config, kwargs, test_dataset)

    training_client_list = get_client_list(config)

    # Training
    for epoch_idx in range(1, 1 + global_config.epoch):
        
        pass


    
    process_list = [Process(target=send_para, args=()) for _ in range(global_config.worker_num)]


def send_para():
    pass


if __name__ == "__main__":
    main()


def send_client_scrpts():

    pass

@singledispatch
def get_client_list(config):
    pass

@get_client_list.register(LocalTrainingConfig)
def _(config):
        print("This is local")

@get_client_list.register(RemoteTrainingConfig)
def _(config):
        print("This is remote")