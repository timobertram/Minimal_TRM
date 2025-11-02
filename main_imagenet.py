from training import deep_supervision, accuracy
from models import TRM_MLP, TRM_CNN, TRM_Mixer

import torch
from datasets import load_dataset
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import copy
from torchinfo import summary
import wandb

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        lbl = int(ex["label"])
        if self.transform: img = self.transform(img)
        return img, lbl



def main(net, config, train_loader, train_acc_loader, test_loader, epochs):
    opt = torch.optim.AdamW(net.parameters(), lr = config["lr"])

    
    ema_net = copy.deepcopy(net)
    ema_net.eval()
    for name,param in ema_net.named_parameters():
        param.requires_grad = False


    wandb_run = wandb.init(project="TRM_ImageNet", config = config)
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
        test_acc = accuracy(
            net = ema_net, 
            loader=test_loader,
            global_step = global_step,
            config = config,
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
            config = config,
        )



if __name__ == "__main__":
    device = "cuda:1"
    config = {
        "lr": 3e-4,
        "input_size": (3,160,160),
        "hidden_size": 512,
        "dropout": 0.1,
        "gamma": 0.999,
        "batch_size": 2048,
        "patch_size": 20,
        "model_type": "TRM_Mixer",
        "n_step_supervised" : 16,
        "n" : 3,
        "T" : 3,
    }
    cls =  globals()[config["model_type"]]
    net = cls(
            output_size=200,
            device = device,
            **config,
    )


    print(summary(model = net))

    train_tf = T.Compose([
        T.RandomResizedCrop(160),
        T.RandomHorizontalFlip(),
        T.ToImage(),          # converts PIL→Tensor (v2)
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(180),
        T.CenterCrop(160),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds = load_dataset("zh-plus/tiny-imagenet")
    train_dataset = HFDataset(ds["train"], transform=train_tf)
    val_dataset   = HFDataset(ds["valid"], transform=val_tf)

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
    test_loader = DataLoader(val_dataset, 
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
    