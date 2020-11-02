import socket
import pickle
import argparse
import asyncio

from config import ClientConfig
from client_comm_utils import *


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

async def local_training(config):
    config = await get_data(LISTEN_PORT)
    config.acc = config.acc + 0.001
    config.epoch_num = config.epoch_num + 1
    await send_data(config, MASTER_IP, MASTER_LISTEN_PORT)

def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip_addr=args.master_ip,
        action=""
    )

    loop = asyncio.get_event_loop()
    while True:
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

if __name__ == '__main__':
    main()


