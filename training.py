import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, binary_cross_entropy
from tqdm import tqdm
import wandb
import numpy as np
from torch.optim.lr_scheduler import LinearLR


def latent_recursion(net, x, y, z, n = 6):
    for i in range(n):
        z = net(y+z+x)
    y = net(y+z)
    return y, z

def deep_recursion(net, x, y, z, n = 6, T = 3):
    x = net.get_input_embeddings(x.to(net.device))
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(net, x, y, z, n) 
    y, z = latent_recursion(net, x, y, z, n)
    output, q = net.get_outputs(y)
    return (y.detach(), z.detach()), output, q.squeeze(dim = -1)

@torch.no_grad()
def ema_step(net, ema_net, gamma):
    for param, ema_param in zip(net.parameters(), ema_net.parameters()):
        ema_param.data.mul_(gamma).add_(param.data, alpha=1 - gamma)


def deep_supervision(epoch, net, ema_net, gamma, opt, train_loader, global_step):
    net.train()
    losses = []
    
    with tqdm(train_loader, desc = "Training") as pbar:
        
        if epoch == 0:
            scheduler = LinearLR(
                opt,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=len(pbar),
                last_epoch=-1
            )
        else:
            scheduler = None

        for x, y_true in pbar:
            y_true = y_true.to(net.device)
            y,z = net.init_carries(batch_size = x.size(0))
            bs = x.size(0)
            halted_pct = 0
            halt_at = []
            for step in range(16):

                opt.zero_grad()
                min_halt_steps = ((torch.rand(x.size(0)) < 0.1) * torch.randint(size=(x.size(0),), low=2, high=16 + 1)).to(net.device)
                (y, z), y_hat, q_hat = deep_recursion(net, x, y, z)
                ce_loss = cross_entropy(y_hat, y_true, reduction = "mean")
                bce_loss = binary_cross_entropy(q_hat, (y_hat.argmax(dim=1) == y_true).float(), reduction = "mean")
                loss = ce_loss + bce_loss

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = 1.0)
                opt.step()

                wandb.log({
                    "lr" : opt.param_groups[0]['lr'],
                    "grad_norm": grad_norm
                }, step = global_step)
                ema_step(net, ema_net, gamma = gamma)

                halted = (q_hat > 0.5)
                halted = halted & (step >= min_halt_steps)
                active = (~halted).to(x.device)

                
                halted_pct += halted.float().sum()/bs
                if halted.all():
                    break

                x = x[active]
                y = y[active]
                z = z[active]
                y_true = y_true[active]
                min_halt_steps = min_halt_steps[active]

                halt_at += (([step]*sum(halted)) if step < 15 else ([step]*len(halted)))


                losses.append(loss.item())

                log_dict = {
                    "train/loss": loss.item(),
                    "train/ce_loss_mean": ce_loss.item(),
                    "train/bce_loss_mean": bce_loss.item(),
                    "train/halted_pct": halted_pct,
                    "train/inner_step": step,
                }
                wandb.log(log_dict, step=global_step)
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            wandb.log({
                "halt_at" : np.mean(halt_at)
            }, step = global_step)
            if scheduler is not None:
                scheduler.step()
    return losses, global_step

@torch.no_grad()
def accuracy(net, loader, global_step):
    net.eval()
    correct = total = 0
    for x, y_target in loader:
        y_target = y_target.to(net.device)

        y,z = net.init_carries(batch_size = x.size(0))
        for step in range(16):
            (y, z), y_hat, q_hat = deep_recursion(net, x, y, z)
        pred = y_hat.argmax(dim = 1)
        correct += (pred == y_target).sum().item()
        total += y_target.size(0)

    
    acc = correct / max(1, total)
    return acc