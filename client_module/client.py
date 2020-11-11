import socket
import pickle
import argparse
import asyncio
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from config import ClientConfig
from client_comm_utils import *
from training_utils import MyNet, train, test
import datasets, models, utils

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--listen_port', type=int, default=47000, metavar='N',
                    help='Port used to listen msg from master')
parser.add_argument('--master_listen_port', type=int, default=57000, metavar='N',
                    help='')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--min_lr', type=float, default=0.0001)
parser.add_argument('--decay_rate', type=float, default=0.98)
parser.add_argument('--local_iters', type=int, default=1)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--enable_vm_test', action="store_true", default=True)
parser.add_argument('--no_cuda', action="store_false", default=False)
parser.add_argument('--dataset_type', type=str, default='MNIST')
parser.add_argument('--model_type', type=str, default='LR')
parser.add_argument('--pattern_idx', type=int, default=0)
parser.add_argument('--tx_num', type=int, default=1)

args = parser.parse_args()

MASTER_IP = args.master_ip
LISTEN_PORT = args.listen_port
MASTER_LISTEN_PORT = args.master_listen_port


def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip_addr=args.master_ip,
        action=""
    )

    # model = None

    # Init dataset
    # train_dataset = None
    # test_dataset = None

    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training(client_config)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        for task in tasks:
            print(task.result())
        loop.close()


async def local_training(config):
    config = await get_data(LISTEN_PORT, MASTER_IP)
    # print(config.para)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    LOG_ROOT_PATH = '/data/test/'
    utils.create_dir(LOG_ROOT_PATH)
    LOG_PATH = LOG_ROOT_PATH + 'model_acc_loss.txt'
    log_out = open(LOG_PATH, 'w+')

    # Update model
    # print(config.model)
    model = MyNet(config.model)

    if config.para is not None:
        model.load_state_dict(config.para)
    # print("loaded", list(model.named_parameters()))
    args.lr = np.max((args.decay_rate ** (config.epoch_num - 1) * args.lr, args.min_lr))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    tx2_train_loader, tx2_test_loader = create_dataloaders(args, kwargs, config)
    train(args, config, model, device, tx2_train_loader, tx2_test_loader, optimizer, config.epoch_num, log_out)
    
    config.model = models.Net2Tuple(model)
    config.para = dict(model.named_parameters())
    # print("trained", config.para)

    await send_data(config, MASTER_IP, MASTER_LISTEN_PORT)

def create_dataloaders(args, kwargs, config):
    # <--Load datasets
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)

    # <--Create federated train/test loaders
    if args.pattern_idx == 0:  # random data (IID)
        is_train = True
        tx2_train_loader = utils.create_segment_loader(
            args, kwargs, args.tx_num, config.idx, is_train, train_dataset)
        is_train = False
        tx2_test_loader = utils.create_segment_loader(
            args, kwargs, args.tx_num, config.idx, is_train, test_dataset)
    del train_dataset
    del test_dataset
    return tx2_train_loader, tx2_test_loader

if __name__ == '__main__':
    main()


