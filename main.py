from training import deep_supervision, test_accuracy
from models import TRM_MLP, TRM_CNN

import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import copy
from torchinfo import summary
import wandb


def main(net, train_loader, test_loader, epochs):
    opt = torch.optim.AdamW(net.parameters(), lr = 1e-4)

    
    ema_net = copy.deepcopy(net)
    ema_net.eval()
    for name,param in ema_net.named_parameters():
        param.requires_grad = False

    train_losses = []
    test_accuracies = []

    wandb_run = wandb.init(project="TRM_MNIST")
    global_step = 0

    for e in range(epochs):
        test_acc = test_accuracy(
            net = ema_net, 
            test_loader=test_loader,
            global_step = global_step,
        )
        print(f"New test acc: {test_acc}")
        test_accuracies.append(test_acc)

        train_loss, global_step = deep_supervision(
            net = net,
            ema_net = ema_net,
            opt = opt,
            train_loader=train_loader,
            global_step = global_step,
        )
        train_losses.append(np.mean(train_loss))



if __name__ == "__main__":
    device = "cuda:0"
    hidden_size = 128
    net = TRM_CNN(
            input_size=28**2, 
            hidden_size=hidden_size, 
            output_size=10,
            device = device
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

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    main(
        net = net, 
        train_loader= train_loader,
        test_loader=test_loader,
        epochs = 100
    )
    