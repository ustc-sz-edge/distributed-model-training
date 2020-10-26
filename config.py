from dataclasses import (
    dataclass,
    field
)
from typing import List

from worker_module.worker import Worker

from torch.utils.tensorboard import SummaryWriter

WORKER_IP_LIST = [
    "local_host"
]


class Action:
    LOCAL_TRAINING = "local_training"


@dataclass
class CommonConfig:
    dataset_type: str = 'MNIST'
    model_type: str = 'LR'
    use_cuda: bool = True
    training_structure: str = 'local'

    worker_num: int = 10

    epoch_start: int = 0
    epoch = 0
    global_update_number: int = 10
    test_batch_size: int = 1000

    master_listen_port_base: int = 57000
    client_listen_port_base: int = 47000
    # master_listen_port_for_client<idx> = master_listen_port_base + <client_idx>

    # global_config.writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    writer: SummaryWriter = SummaryWriter()


@dataclass()
class ClientState:
    model: dict = field(default_factory=dict)
    para: dict = field(default_factory=dict)
    resource: dict = field(default_factory=dict)
    acc: int = 0
    loss: int = 1
    running_time: int = 0


@dataclass
class ClientConfig:
    idx: int
    master_ip_addr: str
    action: str
    state: ClientState = field(default_factory=ClientState)


@dataclass
class LocalConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    worker_list: List[Worker] = field(default_factory=list)

    available_gpu: list = field(default_factory=lambda: ['0', '1'])
    pass


@dataclass
class RemoteConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    work_list: List[Worker] = field(default_factory=list)
    pass


@dataclass
class DockerConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    client: List[Worker] = field(default_factory=list)
    pass
