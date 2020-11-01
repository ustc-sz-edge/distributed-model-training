from typing import List

from torch.utils.tensorboard import SummaryWriter
from communication_module.comm_utils import *

WORKER_IP_LIST = [
    "192.168.1.105"
]


class ClientAction:
    LOCAL_TRAINING = "local_training"


class ServerAction:
    LOCAL_TRAINING = "local_training"


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

    def send_config(self):
        return send_worker_state(self.config, self.ip_addr, self.listen_port)

    def get_config(self):
        return get_worker_state(listen_port=self.master_port)

    async def local_training(self):
        self.config.action = ClientAction.LOCAL_TRAINING
        await self.send_config()
        recv_config = await self.get_config()
        self.config = recv_config


class CommonConfig:
    def __init__(self):
        # global_config.writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
        self.recoder: SummaryWriter = SummaryWriter()

        self.dataset_type = 'MNIST'
        self.model_type = 'LR'
        self.use_cuda = True
        self.training_mode = 'local'

        self.worker_num = 1

        self.epoch_start = 0
        self.epoch = 50

        self.test_batch_size = 64

        self.master_listen_port_base = 57000
        self.client_listen_port_base = 47000

        self.available_gpu: list = ['0']

        self.worker_list: List[Worker] = list()

    # master_listen_port_for_client<idx> = master_listen_port_base + <client_idx>


class ClientConfig:
    def __init__(self,
                 idx: int,
                 master_ip_addr: str,
                 action: str
                 ):
        self.idx = idx
        self.master_ip_addr = master_ip_addr
        self.action = action
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.resource = {"CPU": "1"}
        self.custom = dict()
        self.acc: float = 0
        self.loss: float = 1
        self.running_time: int = 0
