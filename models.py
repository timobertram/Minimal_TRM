import torch.nn as nn
import torch


class TRM(nn.Module):
    def __init__(self,
                input_size,
                y_init, 
                z_init,
                hidden_size = 128,
                output_size = 10):
        super().__init__()

        self.input_embedding = nn.Linear(input_size, hidden_size)

        self.main_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # head to transform latent to final solution
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

        self.y_init_val = y_init
        self.z_init_val = z_init


    def init_carries(self):
        self.y_init = self.y_init_val
        self.z_init = self.z_init_val

    def forward(self, hidden_states, injection, initial_input = torch.zeros(1), halted = None):
        if halted is None:
            halted = torch.zeros(hidden_states.size(0), device = hidden_states.device).bool()
        if (initial_input != 0).any():
            initial_input = self.input_embedding(initial_input)

            
        halted = halted.view(hidden_states.size(0),1)

        candidate = hidden_states + injection + initial_input
        candidate = self.main_block(candidate)
        y_out = self.output_head(candidate)
        q_out = self.q_head(candidate)

        hidden_states = torch.where(halted, hidden_states, candidate)
        return hidden_states, y_out, q_out
        


