import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from tqdm import tqdm


def latent_recursion(net, x, y, z, halted, n = 6):
    for i in range(n):
        z,_,_ = net(y, z, initial_input = x, halted = halted)
    y, output, q = net(y, z)
    return y, z, output, q.squeeze()

def deep_recursion(net, x, y, z, halted, n = 6, T = 3):
    with torch.no_grad():
        for j in range(T-1):
            y, z,_,_ = latent_recursion(net, x, y, z, halted, n) 
    y, z, output, q = latent_recursion(net, x, y, z,  halted, n)
    return (y.detach(), z.detach()), output, q

@torch.no_grad()
def ema_step(net, ema_net, gamma = 0.99):
    for param, ema_param in zip(net.parameters(), ema_net.parameters()):
        ema_param.data.mul_(gamma).add_(param.data, alpha=1 - gamma)


def deep_supervision(net, ema_net, opt, train_loader, y_init = None, z_init = None):
    net.train()
    losses = []
    for x, y_true in tqdm(train_loader, desc = "Training"):
        y_true = y_true.to(net.device)
        net.init_carries()
        y,z = net.y_init.repeat(x.size(0), 1), net.z_init.repeat(x.size(0),1)
        #nothing is halted at the start
        halted = torch.zeros(x.size(0)).bool().to(net.device)
        for step in range(16):
            (y, z), y_hat, q_hat = deep_recursion(net, x, y, z, halted)
            ce_loss = cross_entropy(y_hat, y_true, reduction = "none")
            bce_loss = binary_cross_entropy_with_logits(q_hat, (y_hat.argmax(dim=1) == y_true).float(), reduction = "none")
            
            active = (~halted).float().to(net.device)
            denom = active.sum().clamp(min=1.0)
            loss = ((ce_loss + bce_loss)*active).sum()/denom
            loss.backward()
            opt.step()
            opt.zero_grad()
            ema_step(net, ema_net)

            halted = (q_hat > 0) | halted
            if halted.all():
                break
        losses.append(loss.item())
    return losses

@torch.no_grad()
def test_accuracy(net, test_loader, y_init = None, z_init = None):
    net.eval()
    correct = total = 0
    for x, y_target in test_loader:
        y_target = y_target.to(net.device)

        halted = torch.zeros(x.size(0)).bool()
        net.init_carries()
        y,z = net.y_init.repeat(x.size(0), 1), net.z_init.repeat(x.size(0),1)
        (y, z), y_hat, q_hat = deep_recursion(net, x, y, z, halted)
        pred = y_hat.argmax(dim = 1)
        correct += (pred == y_target).sum().item()
        total += y_target.size(0)
    return correct / max(1, total)