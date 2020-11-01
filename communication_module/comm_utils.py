import asyncio
import socket
import pickle
from time import sleep
from functools import singledispatch
import asyncio


async def send_worker_state(config, dst_ip, dst_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while s.connect_ex((dst_ip, dst_port)) != 0:
            sleep(1)
        data = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        s.send(data)


async def get_worker_state(listen_port, listen_ip=socket.gethostbyname(socket.gethostname())):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((listen_ip, listen_port))
        s.listen(1)
        conn, _ = s.accept()
        data = recv_basic(conn)
        config = pickle.loads(data)
        return config


def recv_basic(conn):
    total_data = b''
    while True:
        data = conn.recv(20480)
        if not data:
            break
        total_data = total_data + data
    return total_data
