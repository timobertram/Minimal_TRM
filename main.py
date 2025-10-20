from training import deep_supervision, test_accuracy
from models import TRM

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def collate_fn(batch):
    x,y = zip(*batch)

    x = torch.stack(x)
    x = x.view(x.size(0),-1)
    y = torch.Tensor(y).long()

    return x,y


def main(net, train_loader, test_loader, epochs):
    opt = torch.optim.AdamW(net.parameters(), lr = 1e-4)

    train_losses = []
    test_accuracies = []

    for e in range(epochs):
        test_acc = test_accuracy(
            net = net, 
            test_loader=test_loader,
        )
        print(f"New test acc: {test_acc}")
        test_accuracies.append(test_acc)

        train_loss = deep_supervision(
            net = net,
            opt = opt,
            train_loader=train_loader,
        )
        train_losses.append(np.mean(train_loss))



if __name__ == "__main__":
    hidden_size = 128
    net = TRM(
            input_size=28**2, 
            hidden_size=hidden_size, 
            output_size=10,
            y_init=torch.randn(hidden_size),
            z_init=torch.randn(hidden_size)
    )

    transform = transforms.Compose([
        transforms.ToTensor(),         
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean/std of MNIST
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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn= collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn= collate_fn)

    main(
        net = net, 
        train_loader= train_loader,
        test_loader=test_loader,
        epochs = 10
    )
    