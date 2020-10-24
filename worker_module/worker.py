import paramiko

from config_module.config import ClientConfig


class Worker:
    _ip_addr = ""
    _port = ""
    _key_path = ""
    _auth_phrase = ""

    _local_script_path = "./client.py"
    _remote_script_path = "~/client.py"

    def __init__(self,
                 config: ClientConfig,
                 ip_addr: str,
                 port: int
                 ):
        self.config = config

        self.work_thread = ProcessLookupError

    def _check_worker_script_exist(self):
        if not len(self._local_script_path) is 0:
            return True
        else:
            return False

    def _sent_scripts(self):
        pass

    def start_remote_worker_process(self):

        pass

    def send_para(self, order: list = None):
        pass

    def get_para(self, state_enable=False):
        pass
