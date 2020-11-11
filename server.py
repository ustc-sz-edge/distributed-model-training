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
    assert common_config.worker_num >= len(WORKER_IP_LIST)
    for worker_idx in range(common_config.worker_num):
        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       master_ip_addr=socket.gethostbyname(socket.gethostname()),
                                       action=ClientAction.LOCAL_TRAINING),
                   ip_addr=WORKER_IP_LIST[worker_idx],
                   master_port=common_config.master_listen_port_base + worker_idx,
                   client_port=common_config.client_listen_port_base + worker_idx
                   )
        )

    device = torch.device("cuda" if common_config.use_cuda and torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if common_config.use_cuda else {}

    """Create model, dataset, etc


        """
    # Create federated model instance
    global_model = models.create_model_instance(
        common_config.dataset_type, common_config.model_type, device)
    # print("init", list(global_model.named_parameters()))

    # print(models.Net2Tuple(global_model))
    for idx, worker in enumerate(common_config.worker_list):
            worker.config.para = dict(global_model.named_parameters())
            worker.config.model = models.Net2Tuple(global_model)

    # Create dataset instance
    train_dataset, test_dataset = datasets.load_datasets(
        common_config.dataset_type)
    # test_loader = utils.create_server_test_loader(
    #     common_config, kwargs, test_dataset)
    is_train = False
    test_loader = utils.create_segment_loader(
            common_config, kwargs, 1, 1, is_train, test_dataset)

    # if not args.epoch_start == 0:
    #     global_model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    # TODO: Add thread to listen new client

    global_para = dict(global_model.named_parameters())

    # for worker in common_config.worker_list:
    #     worker.config.model = models.Net2Tuple(global_model)
    #     worker.config.para = global_para

    action_queue = Queue()

    # Or you can add all action ad once
    action_queue.put(ServerAction.LOCAL_TRAINING)

    for epoch_idx in range(1, 1 + common_config.epoch):
        # Execute action
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)

        # Add next action, may be conditionally
        action_queue.put(ServerAction.LOCAL_TRAINING)

        # Do somethings for recoder

        for worker in common_config.worker_list:
            common_config.recoder.add_scalar('Accuracy/worker_' + str(worker.config.idx), worker.config.acc,
                                             epoch_idx)
            print('Accuracy/worker_' + str(worker.config.idx) + str(worker.config.acc) + "Epoch" +
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
