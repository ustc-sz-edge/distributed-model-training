import torch

import fl_models

def main():
    # Init global paramter
    global_config = {
        "dataset_type": "FashionMNIST"
    }

    print(global_config["dataset_type"])
    # Create model instance

    model = fl_models.create_model_instance(global_config["dataset_type"], model_type, device)
    
    # Create dataset instance
    pass

if __name__ == "__main__":
    main()