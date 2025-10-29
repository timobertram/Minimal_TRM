import torch.nn as nn
import torch

class SwiGLU(nn.Module):
    def __init__(self,
                inp_size,
                mlp_size,
                proj_mult = 2,
                dropout = 0.0,
            ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(inp_size, mlp_size*proj_mult, bias = False)
        self.w2 = nn.Linear(mlp_size, inp_size, bias= False)
        self.act = nn.SiLU()

    def forward(self, x):
        g,v = self.w1(x).chunk(2, dim = -1)
        x = self.act(g) * v
        x = self.dropout(x)
        return self.w2(x)


class TRM_MLP(nn.Module):
    def __init__(self,
                input_size,
                device,
                hidden_size,
                output_size,
                dropout,
                **kwargs):
        super().__init__()

        self.input_embedding = nn.Linear(input_size, hidden_size)

        self.main_block = nn.ModuleList([SwiGLU(hidden_size, hidden_size, dropout= dropout) for _ in range(2)])
        self.rms_norm = nn.RMSNorm(hidden_size, eps = 1e-5)

        # head to transform latent to final solution
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.q_head[0].weight.zero_()
            self.q_head[0].bias.fill_(-5)

        self.y_init_val = nn.Parameter(torch.randn(hidden_size))
        self.z_init_val = nn.Parameter(torch.randn(hidden_size))

        self.device = device
        self.to(device)

    def init_carries(self, batch_size):
        y_0 = self.y_init_val.to(self.device).repeat(batch_size, 1)
        z_0 = self.z_init_val.to(self.device).repeat(batch_size, 1)
        return y_0, z_0

    def forward(self, hidden_states):
        out = self.main_block[0](hidden_states)
        hidden_states = self.rms_norm(hidden_states + out)
        out = self.main_block[1](hidden_states)
        hidden_states = self.rms_norm(hidden_states + out)
        return hidden_states

    def get_outputs(self, solution):
        return self.output_head(solution), self.q_head(solution)

    def get_input_embeddings(self, x):
        return self.input_embedding(x.flatten(start_dim = 1))

        
class TRM_CNN(nn.Module):
    def __init__(self,
                input_size,
                device,
                dropout,
                hidden_size,
                filter_size,
                output_size,
                **kwargs):
        super().__init__()

        self.input_embedding = nn.Identity()

        size_1, size_2 = filter_size

        self.block_1 = nn.Sequential(
            nn.Conv2d(1,size_1,3, padding = "same"),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),  
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(size_1,size_2,3, padding = "same"),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),  
        )

        self.main_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(7*7*size_2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 28**2)

        )
        self.norm_1 = nn.GroupNorm(4, size_1)
        self.norm_2 = nn.GroupNorm(8, size_2)

        # head to transform latent to final solution
        self.output_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28**2, output_size),
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28**2, 1),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.q_head[1].weight.zero_()
            self.q_head[1].bias.fill_(-5)

        self.y_init_val = nn.Parameter(torch.randn((1,28,28)))
        self.z_init_val = nn.Parameter(torch.randn((1,28,28)))

        self.device = device
        self.to(device)

    def forward(self, hidden_states):
        out = self.block_1(hidden_states)
        out = self.norm_1(out)
        out = self.block_2(out)
        out = self.norm_2(out)
        out = self.main_proj(out).view(-1,1,28,28)
        return out


    
    def init_carries(self, batch_size):
        y_0 = self.y_init_val.to(self.device).repeat(batch_size, 1,1,1)
        z_0 = self.z_init_val.to(self.device).repeat(batch_size, 1,1,1)
        return y_0, z_0

    
    def get_outputs(self, solution):
        return self.output_head(solution), self.q_head(solution)

        
    def get_input_embeddings(self, x):
        return x

        

