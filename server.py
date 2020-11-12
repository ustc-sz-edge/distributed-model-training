# from torch.utils.tensorboard import SummaryWriter

import asyncio
import functools
import socket
import pickle
from functools import singledispatch
import asyncio
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F

from config import *
from communication_module.comm_utils import *
from training_module import datasets, models, utils
from training_module.action import ServerAction


def recode_state(paras, result):
    if paras[1] is None:
        pass
    else:
        paras[0].add_scalar('Accuracy/worker_' + str(paras[1].config.idx), paras[1].config.acc,
                            paras[1].config.epoch_num)


def main():
    # Init global parameter
    common_config = CommonConfig()
    # assert common_config.worker_num >= len(WORKER_IP_LIST)
    device = torch.device("cuda" if common_config.use_cuda and torch.cuda.is_available() else "cpu")
    
    worker_idx_list = [0, 1, 3, 6]

    client_port=[47001, 47002, 47003, 47004,47005,47006,47007,47008,47009,47010]
    master_port=[57001, 57002, 57003, 57004,57005,57006,57007,57008,57009,57010]
    # adjacency_matrix = [[0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1],
    #                     [1, 0, 0, 0]]
    # adjacency_matrix = np.array(adjacency_matrix)

    # for worker_idx in range(common_config.worker_num):
    for idx, worker_idx in enumerate(worker_idx_list):
        custom = dict()
        # custom["neighbors"] = list()
        # for neighbor_idx, link in enumerate(adjacency_matrix[worker_idx]):
        #     if link == 1:
        #         custom["neighbors"].append(WORKER_IP_LIST[neighbor_idx])

        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       master_ip_addr=socket.gethostbyname(socket.gethostname()),
                                       action=ClientAction.LOCAL_TRAINING,
                                       custom=custom),
                   ip_addr=WORKER_IP_LIST[idx],
                   master_port=master_port[worker_idx],
                   client_port=client_port[worker_idx]
                   )
        )

    # Create federated model instance
    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    global_model = global_model.to(device)
    # print("init", list(global_model.named_parameters()))

    # print(models.Net2Tuple(global_model))
    init_para = dict(global_model.named_parameters())
    model_tuple = models.Net2Tuple(global_model)

    train_dataset, test_dataset = datasets.load_datasets(
                                    common_config.dataset_type)
    partition_sizes = [1.0 / len(WORKER_IP_LIST) for _ in WORKER_IP_LIST]
    train_data_partition = utils.RandomPartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = utils.RandomPartitioner(test_dataset, partition_sizes=partition_sizes)

    for worker_idx, worker in enumerate(common_config.worker_list):
            worker.config.para = init_para
            worker.config.model = model_tuple
            worker.config.custom["dataset_type"] = common_config.dataset_type
            worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
            worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # Create dataset instance
    test_loader = utils.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    # TODO: Add thread to listen new client

    global_para = dict(global_model.named_parameters())


    action_queue = Queue()

    # Or you can add all action ad once
    

    for epoch_idx in range(1, 1 + common_config.epoch):
        # Execute action
        print("before send")
        action_queue.put(ServerAction.SEND_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)
        print("after send")

        print("before get")
        action_queue.put(ServerAction.GET_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)
        print("after get")


        # Do somethings for recoder

        for worker in common_config.worker_list:
            common_config.recoder.add_scalar('Accuracy/worker_' + str(worker.config.idx), worker.config.acc,
                                             epoch_idx)
            print('Accuracy/worker_' + str(worker.config.idx) + ':' + str(worker.config.acc) + " Epoch_" +
                  str(epoch_idx))

        # Do something for global model
        workers_para = dict()
        vm_weight = 1.0 / len(common_config.worker_list)
        # print("before", common_config.worker_list[0].config.para)
        for idx, worker in enumerate(common_config.worker_list):
            if idx == 0:
                for key, values in worker.config.para.items():
                    # print(values)
                    workers_para[key] = values * vm_weight
            else:
                for key, values in worker.config.para.items():
                    workers_para[key] += values * vm_weight
        # print("worker", workers_para)
        global_model.load_state_dict(workers_para)
        # print("global", list(global_model.named_parameters()))

        global_model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if common_config.dataset_type == 'FashionMNIST' or common_config.dataset_type == 'MNIST':
                    if common_config.model_type == 'LR':
                        data = data.squeeze(1) 
                        data = data.view(-1, 28 * 28)
                
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()

                correct += batch_correct

        test_loss /= len(test_loader.dataset)
        test_accuracy = np.float(1.0 * correct / len(test_loader.dataset))

        print('Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_idx, test_loss, correct, len(test_loader.dataset), 100. * test_accuracy))

        
        for idx, worker in enumerate(common_config.worker_list):
            worker.config.para = dict(global_model.named_parameters())
            worker.config.model = models.Net2Tuple(global_model)


if __name__ == "__main__":
    main()
