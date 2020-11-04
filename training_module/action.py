import asyncio
import functools


class ServerAction:
    LOCAL_TRAINING = "local_training"

    def execute_action(self, action):
        if action == self.LOCAL_TRAINING:
            self.local_training()

    @staticmethod
    def send_states(worker_list):
        loop = asyncio.get_event_loop()
        tasks = []
        for worker in worker_list:
            task = asyncio.ensure_future(worker.send_config())
            tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

    @staticmethod
    def get_states(worker_list):
        loop = asyncio.get_event_loop()
        tasks = []
        for worker in worker_list:
            task = asyncio.ensure_future(worker.get_config())
            tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

    @staticmethod
    def local_training(worker_list):
        loop = asyncio.get_event_loop()
        tasks = []
        for worker in worker_list:
            task = asyncio.ensure_future(worker.local_training())
            tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
