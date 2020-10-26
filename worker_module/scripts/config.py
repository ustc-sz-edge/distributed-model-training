from dataclasses import (
    dataclass,
    field
)

class Action:
    LOCAL_TRAINING = "local_training"

@dataclass()
class ClientState:
    model: dict = field(default_factory=dict)
    para: dict = field(default_factory=dict)
    resource: dict = field(default_factory=dict)
    acc: int = 0
    loss: int = 1
    running_time: int = 0


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

    listen_port: int = 57000
    send_port: int = 47000


@dataclass
class ClientConfig:
    idx: int
    master_ip_addr: str
    action: str
    state: ClientState = field(default_factory=ClientState)
