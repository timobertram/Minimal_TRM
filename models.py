import torch.nn as nn
import torch


class TRM_MLP(nn.Module):
    def __init__(self,
                input_size,
                device,
                hidden_size,
                output_size,
                dropout = 0.1):
        super().__init__()

        self.input_embedding = nn.Linear(input_size, hidden_size)

        self.main_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # head to transform latent to final solution
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        with torch.no_grad():
            self.q_head[-1].weight.zero_()
            self.q_head[-1].bias.fill_(-5)

        self.y_init_val = torch.randn(hidden_size)
        self.z_init_val = torch.randn(hidden_size)

        self.device = device
        self.to(device)

    def init_carries(self, batch_size):
        y_0 = self.y_init_val.to(self.device).repeat(batch_size, 1)
        z_0 = self.z_init_val.to(self.device).repeat(batch_size, 1)
        return y_0, z_0

    def forward(self, hidden_states, injection, initial_input = None, halted = None):
        if halted is None:
            halted = torch.zeros(hidden_states.size(0)).bool()

        hidden_states = hidden_states.to(self.device)
        injection = injection.to(self.device)
        halted = halted.to(self.device)

        if initial_input is not None:
            initial_input = initial_input.to(self.device)
            initial_input = self.input_embedding(initial_input.flatten(start_dim = 1))
        else:
            initial_input = 0.0


        halted = halted.view(hidden_states.size(0),1)

        candidate = hidden_states + injection + initial_input
        candidate = self.main_block(candidate)
        #candidate = candidate + hidden_states + injection
        hidden_states = torch.where(halted, hidden_states, candidate)
        return hidden_states

    def get_outputs(self, solution):
        return self.output_head(solution), self.q_head(solution)

        
class TRM_CNN(nn.Module):
    def __init__(self,
                input_size,
                device,
                hidden_size = 128,
                output_size = 10,
                dropout = 0.1):
        super().__init__()

        self.input_embedding = nn.Identity()

        self.main_block = nn.Sequential(
            nn.Conv2d(1,32,3, padding = "same"),
            nn.GroupNorm(32,32),
            nn.GELU(),
            nn.Dropout2d(dropout),     
            nn.Conv2d(32,64,3, padding = "same"),
            nn.GroupNorm(64,64),
            nn.GELU(),
            nn.Dropout2d(dropout),   
            nn.Conv2d(64,128,3, padding = "same"),
            nn.GroupNorm(128,128),
            nn.GELU(),
            nn.Dropout2d(dropout),  
            nn.Conv2d(128,1,1, padding = "same"), 
        )

        # head to transform latent to final solution
        self.output_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(28**2),
            nn.GELU(),
            nn.Dropout(dropout),   
            nn.Linear(28**2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, output_size),
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(28**2),
            nn.GELU(),
            nn.Dropout(dropout),   
            nn.Linear(28**2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        with torch.no_grad():
            self.q_head[-1].weight.zero_()
            self.q_head[-1].bias.fill_(-5)

        self.y_init_val = torch.randn((1,28,28))
        self.z_init_val = torch.randn((1,28,28))

        self.device = device
        self.to(device)

    def forward(self, hidden_states, injection, initial_input = torch.zeros(1), halted = None):
        if halted is None:
            halted = torch.zeros(hidden_states.size(0)).bool()

        hidden_states = hidden_states.to(self.device)
        injection = injection.to(self.device)
        initial_input = initial_input.to(self.device)
        halted = halted.to(self.device)

        halted = halted.view(hidden_states.size(0),1)

        candidate = hidden_states + injection + initial_input
        candidate = self.main_block(candidate)

        hidden_states = torch.where(halted.unsqueeze(-1).unsqueeze(-1), hidden_states, candidate)
        return hidden_states

    
    def init_carries(self, batch_size):
        y_0 = self.y_init_val.to(self.device).repeat(batch_size, 1,1,1)
        z_0 = self.z_init_val.to(self.device).repeat(batch_size, 1,1,1)
        return y_0, z_0

    
    def get_outputs(self, solution):
        return self.output_head(solution), self.q_head(solution)

        

