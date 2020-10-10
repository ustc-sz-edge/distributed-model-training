import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist

from dataclasses import dataclass
import numpy as np
from multiprocessing import Process, Pool
import os
import paramiko

import fl_models
import fl_datasets
import fl_utils


@dataclass
class GlobalConfig:
    dataset_type: str = 'MNIST'
    model_type: str = 'LR'
    use_cuda: bool = True
    training_structure: str = 'local'
    writer: SummaryWriter = SummaryWriter() # global_config.writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    worker_num : int = 10

    epoch_start: int = 0
    global_update_number: int = 10
    test_batch_size: int = 1000


@dataclass
class LocalTrainingConfig:
    available_gpu: list = ["0", "1"]
    pass


@dataclass
class RemoteTrainingConfig:
    pass


@dataclass
class DockerTrainingConfig:
    pass

def send_client_scrpts():
    
    pass

def main():
    # Init global paramter
    global_config = GlobalConfig()

    device = torch.device("cuda" if global_config.use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if global_config.use_cuda else {}

    if global_config.training_structure == 'local':
        config = LocalTrainingConfig()

    """Create model, dataset, etc


    """

    # Create federated model instance
    model = fl_models.create_model_instance(global_config.dataset_type, global_config.model_type, device)
    
    # Create dataset instance
    train_dataset, test_dataset = fl_datasets.load_datasets(global_config.dataset_type)
    test_loader = fl_utils.create_server_test_loader(global_config, kwargs, test_dataset)
    
    """ Initial worker process
    """
    # Send worker client scripts and start worker process
    for worker_idx in range(global_config.worker_num):
        if global_config.training_structure == 'local':
            
            dst_url = "localhost"

    process_list = [Process(target=send_para, args=()) for _ in range(global_config.worker_num)]

def send_para():
    pass



if __name__ == "__main__":
    main()
