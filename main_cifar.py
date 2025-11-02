from training import deep_supervision, accuracy
from models import TRM_MLP, TRM_CNN, TRM_Mixer, TRM_Attn

import torch
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import copy
from torchinfo import summary
import wandb

import os
os.makedirs("checkpoints_cifar", exist_ok=True)


def main(net, config, train_loader, train_acc_loader, test_loader, epochs):
    opt = torch.optim.AdamW(net.parameters(), lr = config["lr"])

    
    ema_net = copy.deepcopy(net)
    ema_net.eval()
    for name,param in ema_net.named_parameters():
        param.requires_grad = False


    wandb_run = wandb.init(project="TRM_CIFAR", config = config)
    global_step = 0
    test_acc = 0

    for e in range(epochs):
        wandb.log({"epoch":e}, step = global_step)
        train_acc = accuracy(
            net = ema_net, 
            loader=train_acc_loader,
            global_step = global_step,
            config = config,
        )
        print(f"New train acc: {train_acc}")
        new_test_acc = accuracy(
            net = ema_net, 
            loader=test_loader,
            global_step = global_step,
            config = config,
        )
        print(f"New test acc: {new_test_acc}")

        if new_test_acc > test_acc:
            ckpt_name = f"epoch_{e:03d}_valacc_{new_test_acc:.3f}.pt"
            ckpt_path = os.path.join("checkpoints_cifar", ckpt_name)

            torch.save({
                "epoch": e,
                "model_state_dict": ema_net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_acc": new_test_acc,
            }, ckpt_path)

            # attach that file to THIS wandb run
            wandb.save(ckpt_path)

            test_acc = new_test_acc


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
            config = config,
        )



if __name__ == "__main__":
    device = "cuda:0"
    config = {
        "lr": 1e-3,
        "input_size": (3,32,32),
        "hidden_size": 256,
        "dropout": 0.1,
        "gamma": 0.99,
        "batch_size": 128,
        "patch_size": 4,
        "model_type": "TRM_Mixer",
        "n_step_supervised" : 8,
        "n" : 2,
        "T" : 2,
    }
    cls =  globals()[config["model_type"]]
    net = cls(
            output_size=10,
            device = device,
            **config,
    )


    print(summary(model = net))

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        T.RandomErasing(
            p=0.25,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(
        root='data',     
        train=True,       
        download=True,     
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=True,
                                pin_memory= True,
                                num_workers= 8,
                                prefetch_factor=2)
    train_acc_loader = DataLoader(train_dataset, 
                                batch_size=10000, 
                                shuffle=True,
                                pin_memory= True,
                                num_workers= 8,
                                prefetch_factor=2)
    test_loader = DataLoader(test_dataset, 
                            batch_size=10000, 
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
        epochs = 100
    )
    