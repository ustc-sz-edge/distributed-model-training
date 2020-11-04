# from torch.utils.tensorboard import SummaryWriter

import asyncio
import functools
import socket
import pickle
from functools import singledispatch
import asyncio
from queue import Queue

import torch

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

    device = torch.device("cuda" if common_config.use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if common_config.use_cuda else {}

    """Create model, dataset, etc


        """
    # Create federated model instance
    global_model = models.create_model_instance(
        common_config.dataset_type, common_config.model_type, device)

    # Create dataset instance
    train_dataset, test_dataset = datasets.load_datasets(
        common_config.dataset_type)
    test_loader = utils.create_server_test_loader(
        common_config, kwargs, test_dataset)

    # if not args.epoch_start == 0:
    #     global_model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    # TODO: Add thread to listen new client

    global_para = dict(global_model.named_parameters())

    # for worker in common_config.worker_list:
    #     worker.config.model = models.Net2Tuple(global_model)
    #     worker.config.para = global_para

    action_queue = Queue()
    action_queue.put(ServerAction.LOCAL_TRAINING)
    for epoch_idx in range(1, 1 + common_config.epoch):
        ServerAction().execute_action(action_queue.get())
        action_queue.put(ServerAction.LOCAL_TRAINING)

        # Methods you want to do every epoch

        for worker in common_config.worker_list:
            common_config.recoder.add_scalar('Accuracy/worker_' + str(worker.config.idx), worker.config.acc,
                                             worker.config.epoch_num)


if __name__ == "__main__":
    main()
