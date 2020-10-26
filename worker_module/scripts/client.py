import socket
import pickle
import argparse
import asyncio

from config import *
from communication_module.comm_utils import *

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--listen_port', type=int, default=47000, metavar='N',
                    help='Port used to listen msg from master')
parser.add_argument('--master_listen_port', type=int, default=57000, metavar='N',
                    help='')

args = parser.parse_args()


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
                # send_worker_state(worker.config, worker.ip_addr, worker.listen_port)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        for task in tasks:
            print(task.result())
        loop.close()

        if __name__ == '__main__':
            main()
