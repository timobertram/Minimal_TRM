import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from tqdm import tqdm
import wandb


def latent_recursion(net, x, y, z, halted, n = 6):
    for i in range(n):
        z = net(z, y, initial_input = x, halted = halted)
    y = net(y, z)
    return y, z

def deep_recursion(net, x, y, z, halted, n = 6, T = 3):
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(net, x, y, z, halted, n) 
    y, z = latent_recursion(net, x, y, z,  halted, n)
    output, q = net.get_outputs(y)
    return (y.detach(), z.detach()), output, q.squeeze()

@torch.no_grad()
def ema_step(net, ema_net, gamma = 0.999):
    for param, ema_param in zip(net.parameters(), ema_net.parameters()):
        ema_param.data.mul_(gamma).add_(param.data, alpha=1 - gamma)


def deep_supervision(net, ema_net, opt, train_loader, global_step):
    net.train()
    losses = []
    with tqdm(train_loader, desc = "Training") as pbar:
        for x, y_true in pbar:
            y_true = y_true.to(net.device)
            y,z = net.init_carries(batch_size = x.size(0))
            #nothing is halted at the start
            halted = torch.zeros(x.size(0)).bool().to(net.device)
            for step in range(16):
                (y, z), y_hat, q_hat = deep_recursion(net, x, y, z, halted)
                ce_loss = cross_entropy(y_hat, y_true, reduction = "none")
                bce_loss = binary_cross_entropy_with_logits(q_hat, (y_hat.argmax(dim=1) == y_true).float(), reduction = "none")
                
                
                active = (~halted).float().to(net.device)
                denom = active.sum().clamp(min=1.0)
                ce_loss = (ce_loss * active).sum()/denom
                bce_loss = (bce_loss * active).sum()/denom
                loss = ce_loss + bce_loss

                loss.backward()
                opt.step()
                opt.zero_grad()
                ema_step(net, ema_net)

                halted = (q_hat > 0) | halted
                halted_pct = halted.float().mean().item()
                if halted.all():
                    break
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
    return losses, global_step

@torch.no_grad()
def test_accuracy(net, test_loader, global_step):
    net.eval()
    correct = total = 0
    for x, y_target in test_loader:
        y_target = y_target.to(net.device)

        halted = torch.zeros(x.size(0)).bool()
        y,z = net.init_carries(batch_size = x.size(0))
        for step in range(16):
            (y, z), y_hat, q_hat = deep_recursion(net, x, y, z, halted)
        pred = y_hat.argmax(dim = 1)
        correct += (pred == y_target).sum().item()
        total += y_target.size(0)

    
    acc = correct / max(1, total)
    log_dict = {
        "test/acc" : acc
    }
    wandb.log(log_dict, step=global_step)
    return acc