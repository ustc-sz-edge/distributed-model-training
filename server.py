# from torch.utils.tensorboard import SummaryWriter

import asyncio
import socket
import pickle
from functools import singledispatch
import asyncio

from config import *
from communication_module.comm_utils import *


def recv_basic(conn):
    total_data = b''
    while True:
        data = conn.recv(20480)
        if not data:
            break
        total_data = total_data + data
    return total_data


def main():
    # Init global parameter
    config = RemoteConfig()
    assert config.common.worker_num == len(WORKER_IP_LIST)

    for worker_idx in range(config.common.worker_num):
        config.work_list.append(
            Worker(config=ClientConfig(worker_idx, socket.gethostname(), action=Action.LOCAL_TRAINING),
                   ip_addr=WORKER_IP_LIST[worker_idx],
                   master_port=config.common.master_listen_port_base + worker_idx,
                   client_port=config.common.client_listen_port_base + worker_idx
                   )
        )

    model = 1

    # device = torch.device("cuda" if config.common.use_cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if config.common.use_cuda else {}
    loop = asyncio.get_event_loop()
    while True:
        tasks = []
        for worker in config.work_list:
            tasks.append(loop.create_task(worker.get_state()))
        loop.run_until_complete(asyncio.wait(tasks))

        for task in tasks:
            print(task.result())
    loop.close()

    """Create model, dataset, etc


    """

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

    """


def send_para():
    pass


if __name__ == "__main__":
    main()
