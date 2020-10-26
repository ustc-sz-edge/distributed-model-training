import asyncio
import socket
import pickle
from functools import singledispatch
import asyncio

from config import *


@asyncio.coroutine
def send_worker_state(config, dst_ip, dst_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((dst_ip, dst_port))
        data = pickle.dumps(config)
        s.send(data)


@asyncio.coroutine
def get_worker_state(listen_port, listen_ip=socket.gethostname()):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((listen_ip, listen_port))
        s.listen(1)
        conn, addr = s.accept()
        data = recv_basic(conn)
        yield pickle.loads(data)


def recv_basic(conn):
    total_data = b''
    while True:
        data = conn.recv(20480)
        if not data:
            break
        total_data = total_data + data
    return total_data
