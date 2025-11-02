from training import deep_supervision, accuracy
from models import TRM_MLP, TRM_CNN, TRM_Mixer, TRM_Attn

import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import copy
from torchinfo import summary
import wandb


import os
os.makedirs("checkpoints_mnist", exist_ok=True)


def main(net, config, train_loader, train_acc_loader, test_loader, epochs):
    opt = torch.optim.AdamW(net.parameters(), lr = config["lr"])
    test_acc = 0


    
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
    device = "cuda:2"
    config = {
        "lr": 1e-3,
        "input_size": (1,28,28),
        "hidden_size" : 256,
        "dropout": 0.1,
        "gamma": 0.99,
        "batch_size": 128,
        "patch_size": 4,
        "model_type": "TRM_Mixer",
        "n_step_supervised" : 16,
        "n" : 1,
        "T" : 1,
    }
    cls =  globals()[config["model_type"]]
    net = cls(
            output_size=10,
            device = device,
            **config,
    )


    print(summary(model = net))

    train_transform = v2.Compose([
        v2.RandomAffine(
            degrees=15,        # ±15° rotation
            translate=(0.1, 0.1),  # up to 10% translation
            scale=(0.9, 1.1),      # small zoom in/out
            shear=(-10, 10),       # small shear
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=0
        ),
        v2.RandomApply([
            v2.ElasticTransform(alpha=30.0)  # warping like handwriting
        ], p=0.3),
        v2.RandomErasing(p=0.1, scale=(0.02, 0.15)),  # occlude small region
        v2.ToTensor(),
        v2.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = v2.Compose([
        v2.ToTensor(),         
        v2.Normalize((0.1307,), (0.3081,))  # Normalize with mean/std of MNIST
    ])

    train_dataset = datasets.MNIST(
        root='data',     
        train=True,       
        download=True,     
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=test_transforms
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
    