from typing import List
import paramiko
from scp import SCPClient

from torch.utils.tensorboard import SummaryWriter
from communication_module.comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    ip_addr = ""
    listen_port = 0
    master_port = 0

    user_name = ""
    pass_wd = ""

    local_scripts_path = ""
    remote_scripts_path = ""

    def __init__(self,
                 config,
                 user_name,
                 pass_wd,
                 local_scripts_path,
                 remote_scripts_path,
                 ip_addr,
                 master_port,
                 client_port
                 ):
        self.config = config
        self.work_thread = None
        self.idx = config.idx
        self.ip_addr = ip_addr
        self.user_name = user_name
        self.pass_wd = pass_wd
        self.local_scripts_path = local_scripts_path
        self.remote_scripts_path = remote_scripts_path
        self.listen_port = client_port
        self.master_port = master_port

        # Start remote process
        # while not self.__check_worker_script_exist():
        #     self.__send_scripts()
        #     break

        self.__send_scripts()
        self.__start_remote_worker_process()

    def __check_worker_script_exist(self):
        if not len(self.local_scripts_path) is 0:
            return True
        else:
            return False

    def __send_scripts(self):
        s = paramiko.SSHClient()
        s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        s.connect(self.ip_addr, username=self.user_name, password=self.pass_wd)

        scp_client = SCPClient(s.get_transport(), socket_timeout=15.0)
        try:
            scp_client.put(self.local_scripts_path, self.remote_scripts_path, True)
        except FileNotFoundError as e:
            print(e)
            print("file not found " + self.local_scripts_path)
        else:
            print("file was uploaded to", self.user_name, ": ", self.ip_addr)
        scp_client.close()
        s.close()

    def __start_remote_worker_process(self):
        s = paramiko.SSHClient()
        s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        s.connect(self.ip_addr, username=self.user_name, password=self.pass_wd)
        stdin, stdout, stderr = s.exec_command('cd ' + self.remote_scripts_path + ';ls')
        print(stdout.read().decode('utf-8'))
        s.exec_command('cd ' + self.remote_scripts_path + '/client_module' + ';nohup python3 client.py --listen_port ' + str(self.listen_port) + ' --master_listen_port ' + str(
            self.master_port) + ' --idx ' + str(self.idx) + '&')

        print("start process at ", self.user_name, ": ", self.ip_addr)

    async def send_config(self):
        print("before send", self.idx, self.ip_addr, self.listen_port, self.master_port)
        await send_worker_state(self.config, self.ip_addr, self.listen_port)

    async def get_config(self):
        self.config = await get_worker_state(listen_port=self.master_port, listen_ip="192.168.1.104"
                                                                                     "")

    async def local_training(self):
        self.config.action = ClientAction.LOCAL_TRAINING
        print("before send", self.idx, self.ip_addr, self.listen_port, self.master_port)
        await self.send_config()
        print("after send", self.idx)
        recv_config = await self.get_config()
        print("after get", self.idx)
        self.config = recv_config


class CommonConfig:
    def __init__(self):
        # global_config.writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
        self.recoder: SummaryWriter = SummaryWriter()

        self.dataset_type = 'CIFAR10'
        self.model_type = 'AlexNet'
        self.use_cuda = True
        self.training_mode = 'local'

        self.epoch_start = 0
        self.epoch = 500

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
                 action: str,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.master_ip_addr = master_ip_addr
        self.action = action
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 0
        self.loss: float = 1
        self.running_time: int = 0
