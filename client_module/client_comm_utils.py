import socket
import pickle
import asyncio

from time import sleep
from config import *


async def send_data(config, dst_ip, dst_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while s.connect_ex((dst_ip, dst_port)) != 0:
            sleep(1)
        data = pickle.dumps(config)
        s.send(data)


async def get_data(listen_port, listen_ip="192.168.1.105"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((listen_ip, listen_port))
        s.listen(1)
        conn, _ = s.accept()
        data = recv_basic(conn)
        recv_config = pickle.loads(data)
        return recv_config


def recv_basic(conn):
    total_data = b''
    while True:
        data = conn.recv(20480)
        if not data:
            break
        total_data = total_data + data
    return total_data
