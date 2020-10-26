

from config import ClientConfig, ClientState
from communication_module.comm_utils import *

class Worker:
    ip_addr = ""
    listen_port = 0
    master_port = 0

    __local_script_path = "./client.py"
    __remote_script_path = "~/client.py"

    def __init__(self,
                 config,
                 ip_addr,
                 master_port,
                 client_port
                 ):
        self.config = config
        self.work_thread = None
        self.idx = config.idx
        self.ip_addr = ip_addr
        self.listen_port = client_port
        self.master_port = master_port

        # Start remote process
        while not self.__check_worker_script_exist():
            self.__start_remote_worker_process()
            break
        else:
            self.__send_scripts()

    def __check_worker_script_exist(self):
        if not len(self.__local_script_path) is 0:
            return True
        else:
            return False

    def __send_scripts(self):
        pass

    def __start_remote_worker_process(self):

        pass

    def send_client_state(self):
        return asyncio.ensure_future(send_worker_state(self.config, self.ip_addr, self.listen_port))

    def get_client_state(self):
        return asyncio.ensure_future(get_worker_state(listen_port=self.master_port))




