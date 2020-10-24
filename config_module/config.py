from dataclasses import (
    dataclass,
    field
)

from torch.utils.tensorboard import SummaryWriter


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

    # global_config.writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    writer: SummaryWriter = SummaryWriter()


@dataclass
class ClientConfig:
    pass

@dataclass
class LocalConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    client: ClientConfig = field(default_factory=ClientConfig)

    available_gpu: list = field(default_factory=lambda: ['0', '1'])
    pass


@dataclass
class RemoteConfig:
    common: list = field(default_factory=CommonConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    pass


@dataclass
class DockerConfig:
    common: list = field(default_factory=CommonConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    pass