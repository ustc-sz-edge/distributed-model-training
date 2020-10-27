import socket
import pickle
import argparse
import asyncio

from config import *
from comm_utils import *


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

def 

def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip_addr=args.master_ip,
        action="",
        state=ClientState()
    )

    loop = asyncio.get_event_loop()
    while True:
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                # get_worker_state(LISTEN_PORT)
                send_worker_state(client_config, MASTER_IP, MASTER_LISTEN_PORT)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        for task in tasks:
            print(task.result())
    loop.close()

if __name__ == '__main__':
    main()


