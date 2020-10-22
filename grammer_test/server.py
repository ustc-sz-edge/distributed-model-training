from multiprocessing import Pool, Queue, Process, Lock
from dataclasses import dataclass, field
from functools import singledispatch

@dataclass
class LocalTrainingConfig:
    available_gpu: list = field(default_factory=lambda: ['0', '1'])
    pass

@dataclass
class RemoteTrainingConfig:
    pass


@dataclass
class DockerTrainingConfig:
    pass

@singledispatch
def train_call(config):
    pass

@train_call.register(LocalTrainingConfig)
def _(config):
        print("This is local")

@train_call.register(RemoteTrainingConfig)
def _(config):
        print("This is remote")


def f(q, num, lock):
    lock.acquire()
    print("Index: " + str(num))
    print(q.get() + num)
    q.put(num)
    lock.release()

if __name__ == '__main__':
    local = LocalTrainingConfig()
    remote = RemoteTrainingConfig()

    train_call(local)
    train_call(remote)

    lock = Lock()

    q = Queue()
    q.put(1)

    for num in range(10):
        Process(target=f, args=(q, num, lock)).start()
    # with Pool(1) as p:
    #     print(p.map(f, [1, 2, 3]))
    

    