from client_comm_utils import *
from client_config import *

class ActionFunction:
    async def local_update(self, config, local_training_number):
        pass

    def get_action(self, config):
        if config.action == ClientAction.LOCAL_TRAINING:
            pass
