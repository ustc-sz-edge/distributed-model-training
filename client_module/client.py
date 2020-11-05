import socket
import pickle
import argparse
import asyncio

from config import ClientConfig
from client_comm_utils import *
from training_utils import MyNet

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="192.168.1.104",
                    help='IP address for controller or ps')
parser.add_argument('--listen_port', type=int, default=47000, metavar='N',
                    help='Port used to listen msg from master')
parser.add_argument('--master_listen_port', type=int, default=57000, metavar='N',
                    help='')

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


    # Init dataset
    train_dataset = None 
    test_dataset = None
    
    while True:
        loop = asyncio.get_event_loop()
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
    config = await get_data(LISTEN_PORT)
    # Update model

    model = MyNet(config.model)
    if config.para is not None:
        model.load_state_dict(config.para)

    # Do something for training


    await send_data(config, MASTER_IP, MASTER_LISTEN_PORT)

if __name__ == '__main__':
    main()


