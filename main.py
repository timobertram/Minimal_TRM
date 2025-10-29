from training import deep_supervision, accuracy
from models import TRM_MLP, TRM_CNN

import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import copy
from torchinfo import summary
import wandb


def main(net, config, train_loader, train_acc_loader, test_loader, epochs):
    opt = torch.optim.AdamW(net.parameters(), lr = config["lr"])

    
    ema_net = copy.deepcopy(net)
    ema_net.eval()
    for name,param in ema_net.named_parameters():
        param.requires_grad = False


    wandb_run = wandb.init(project="TRM_MNIST", config = config)
    global_step = 0

    for e in range(epochs):
        wandb.log({"epoch":e}, step = global_step)
        train_acc = accuracy(
            net = ema_net, 
            loader=train_acc_loader,
            global_step = global_step,
        )
        print(f"New train acc: {train_acc}")
        test_acc = accuracy(
            net = ema_net, 
            loader=test_loader,
            global_step = global_step,
        )
        print(f"New test acc: {test_acc}")

        wandb.log({
            "train/acc": train_acc,
            "test/acc" : test_acc,
        }, step=global_step)

        train_loss, global_step = deep_supervision(
            epoch = e,
            net = net,
            ema_net = ema_net,
            gamma = config["gamma"],
            opt = opt,
            train_loader=train_loader,
            global_step = global_step,
        )



if __name__ == "__main__":
    device = "cuda:2"
    config = {
        "lr": 1e-3,
        "filter_size": [64,128],
        "hidden_size": 128,
        "dropout": 0.0,
        "gamma": 0.99,
        "batch_size": 256,
        "model_type": "TRM_CNN"

    }
    cls =  globals()[config["model_type"]]
    net = cls(
            input_size=28**2, 
            output_size=10,
            device = device,
            **config,
    )


    print(summary(model = net))

    transform = v2.Compose([
        v2.ToTensor(),         
        v2.Normalize((0.1307,), (0.3081,))  # Normalize with mean/std of MNIST
    ])

    train_dataset = datasets.MNIST(
        root='data',     
        train=True,       
        download=True,     
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=True,
                                pin_memory= True,
                                num_workers= 8,
                                prefetch_factor=2)
    train_acc_loader = DataLoader(train_dataset, 
                                batch_size=2048, 
                                shuffle=True,
                                pin_memory= True,
                                num_workers= 8,
                                prefetch_factor=2)
    test_loader = DataLoader(test_dataset, 
                            batch_size=2048, 
                            shuffle=False,
                            pin_memory= True,
                            num_workers= 8,
                            prefetch_factor=2)

    main(
        net = net, 
        config = config,
        train_loader= train_loader,
        train_acc_loader = train_acc_loader,
        test_loader=test_loader,
        epochs = 50
    )
    