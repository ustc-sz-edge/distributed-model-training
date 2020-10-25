import paramiko
from functools import singledispatch

from config_module.config import *


class Worker:
    __ip_addr = ""
    __port = ""
    __key_path = ""
    __auth_phrase = ""

    __local_script_path = "./client.py"
    __remote_script_path = "~/client.py"

    model = None
    para = dict()

    def __init__(self,
                 config: ClientConfig,
                 ip_addr: str,
                 port: int
                 ):
        self.config = config
        self.work_thread = None

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

    @singledispatch
    def get_client_list(self, config):
        pass

    @get_client_list.register(LocalConfig)
    def _(self, config):
        print("This is local")

    @get_client_list.register(RemoteConfig)
    def _(self, config):
        print("This is remote")

    def pull(self, order: list = None):
        pass

    def push(self, state_enable=False):
        pass
