import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self,
                inp_size,
                proj_mult = 2,
                dropout = 0.0,
            ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.w1 = nn.Linear(inp_size, inp_size*proj_mult, bias = False)
        self.w2 = nn.Linear(inp_size*(proj_mult//2), inp_size, bias= False)
        self.act = nn.SiLU()

    def forward(self, x):
        g,v = self.w1(x).chunk(2, dim = -1)
        x = self.act(g) * v
        x = self.dropout(x)
        return self.w2(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, hidden_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = self.num_patches_per_dim **2

        self.proj = nn.Linear(in_channels*patch_size*patch_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim)
        )

        self.embed_scale = math.sqrt(hidden_dim)

        
        # init similar to ViT
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, x.shape[1], -1)   

        x = self.proj(x)
        x = torch.cat([self.cls_token.expand(B,1,-1), x], dim = 1)

        x = x + self.pos_embedding
        x = self.embed_scale * x
        return x

class MixerBlock(nn.Module):
    def __init__(self, 
                seq_len,
                hidden_dim,
                dropout,
                eps = 1e-5):
        super().__init__()
        self.mlp_tokens = SwiGLU(inp_size=seq_len, dropout= dropout, proj_mult=4)
        self.norm_tokens = nn.RMSNorm(seq_len, eps = eps)

        self.mlp_channels = SwiGLU(inp_size=hidden_dim, dropout= dropout, proj_mult=4)
        self.norm_channels = nn.RMSNorm(hidden_dim, eps = eps)

    def forward(self, x):
        x_t = x.transpose(1,2)
        out_t = self.mlp_tokens(x_t)
        x_t = x_t + out_t
        x_t = self.norm_tokens(x_t)

        x = x_t.transpose(1,2)

        out_c = self.mlp_channels(x)
        x = x + out_c
        x = self.norm_channels(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, 
                num_heads,
                hidden_dim,
                dropout,
                eps = 1e-5):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim= hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_attn = nn.RMSNorm(hidden_dim, eps = eps)

        self.mlp_channels = SwiGLU(inp_size=hidden_dim, dropout= dropout, proj_mult=4)
        self.norm_channels = nn.RMSNorm(hidden_dim, eps = eps)

    def forward(self, x):
        out_attn,_ = self.attn(x,x,x, need_weights = False)
        x = self.norm_attn(x + out_attn)

        out_c = self.mlp_channels(x)
        x = x + out_c
        x = self.norm_channels(x)

        return x

class TRM_Attn(nn.Module):
    def __init__(self,
                input_size,
                device,
                dropout,
                hidden_size,
                output_size,
                patch_size,
                num_heads,
                **kwargs):
        super().__init__()

        in_channels, img_size, _ = input_size
        self.input_embedding = nn.Identity()
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_dim=hidden_size
        )

        num_patches_per_dim = img_size // patch_size
        seq_len = num_patches_per_dim **2 +1
        self.main_blocks = nn.ModuleList([AttentionBlock(hidden_dim=hidden_size, num_heads = num_heads,dropout=dropout) for _ in range(2)])

        

        # head to transform latent to final solution
        self.output_head = nn.Sequential(                
            nn.Linear(hidden_size, output_size, bias=False)      
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=True),   
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.q_head[-2].weight.zero_()
            self.q_head[-2].bias.fill_(-5)

        self.y_init_val = torch.randn(seq_len, hidden_size)
        self.z_init_val = torch.randn(seq_len, hidden_size)

        self.device = device
        self.input_size = input_size
        self.to(device)

    def forward(self, x):
        for block in self.main_blocks:
            x = block(x)
        return x


    
    def init_carries(self, batch_size):
        y_0 = self.y_init_val.to(self.device).repeat(batch_size, 1, 1)
        z_0 = self.z_init_val.to(self.device).repeat(batch_size, 1, 1)
        return y_0, z_0


    def get_outputs(self, solution):
        return self.output_head(solution[:,0,:]), self.q_head(solution[:,0,:])

        
    def get_input_embeddings(self, inp):
        return self.patch_embedding(inp)

        


class TRM_Mixer(nn.Module):
    def __init__(self,
                input_size,
                device,
                dropout,
                hidden_size,
                output_size,
                patch_size,
                **kwargs):
        super().__init__()

        in_channels, img_size, _ = input_size
        self.input_embedding = nn.Identity()
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_dim=hidden_size
        )

        num_patches_per_dim = img_size // patch_size
        seq_len = num_patches_per_dim **2 +1
        self.main_blocks = nn.ModuleList([MixerBlock(seq_len=seq_len, hidden_dim=hidden_size, dropout=dropout) for _ in range(3)])

        

        # head to transform latent to final solution
        self.output_head = nn.Sequential(                
            nn.Linear(hidden_size, output_size, bias=False)      
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=True),   
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.q_head[-2].weight.zero_()
            self.q_head[-2].bias.fill_(-5)

        self.y_init_val = nn.Parameter(torch.randn(seq_len, hidden_size))
        self.z_init_val = nn.Parameter(torch.randn(seq_len, hidden_size))

        self.device = device
        self.input_size = input_size
        self.to(device)

    def forward(self, x):
        for block in self.main_blocks:
            x = block(x)
        return x


    
    def init_carries(self, batch_size):
        y_0 = self.y_init_val.to(self.device).repeat(batch_size, 1, 1)
        z_0 = self.z_init_val.to(self.device).repeat(batch_size, 1, 1)
        return y_0, z_0


    def get_outputs(self, solution):
        return self.output_head(solution[:,0,:]), self.q_head(solution[:,0,:])

        
    def get_input_embeddings(self, x):
        return self.patch_embedding(x)

        

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

        self.main_block = nn.ModuleList([SwiGLU(hidden_size, dropout= dropout) for _ in range(2)])
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
                filter_size,
                output_size,
                **kwargs):
        super().__init__()

        self.input_embedding = nn.Identity()

        C, H, W = input_size


        self.C = C
        self.H = H
        self.W = W

        i = 0

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
            nn.Conv2d(C,filter_size[0],3, padding = "same"),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),  
            nn.GroupNorm(filter_size[0]//4 ,filter_size[0])
        ))

        for i in range(len(filter_size)-1):
            in_filter = filter_size[i]
            out_filter = filter_size[i+1]

            block = nn.Sequential(
                nn.Conv2d(in_filter,out_filter,3, padding = "same"),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),  
                nn.GroupNorm(out_filter//4 ,out_filter)
            )
            self.blocks.append(block)
            

        pooled_size = (H // (2**len(filter_size)))
        self.mixing = nn.Sequential(
            MixerBlock(pooled_size**2, hidden_dim = filter_size[-1], dropout=dropout),
            MixerBlock(pooled_size**2, hidden_dim = filter_size[-1], dropout=dropout),
            
        )

        
        self.post_proj = nn.Conv2d(filter_size[-1], C, kernel_size=1)
        self.upsample_refine = nn.Sequential(
            nn.Conv2d(C,C, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.GroupNorm(1, C)
        )

        # head to transform latent to final solution
        self.output_head = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),                
            nn.Flatten(start_dim=1),                    
            nn.Linear(C, output_size, bias=True)      
        )

        # head to transform latent to stopping criterium
        self.q_head = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),                
            nn.Flatten(start_dim=1),                    
            nn.Linear(C, 1, bias=True),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.q_head[-2].weight.zero_()
            self.q_head[-2].bias.fill_(-5)

        self.y_init_val = nn.Parameter(torch.randn(input_size))
        self.z_init_val = nn.Parameter(torch.randn(input_size))

        self.device = device
        self.input_size = input_size
        self.to(device)

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        B,C, H,W = hidden_states.shape

        tokens = hidden_states.view(B,C, H*W).transpose(1,2)
        tokens_mixed = self.mixing(tokens).transpose(1,2).view(B,C,H, W)

        
        out = self.post_proj(tokens_mixed)
        out = F.interpolate(
            out,
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False
        )  # (B, C, H, W)

        out = self.upsample_refine(out)


        return out


    
    def init_carries(self, batch_size):
        y_0 = self.y_init_val.to(self.device).repeat(batch_size, 1,1,1)
        z_0 = self.z_init_val.to(self.device).repeat(batch_size, 1,1,1)
        return y_0, z_0


    def get_outputs(self, solution):
        return self.output_head(solution), self.q_head(solution)

        
    def get_input_embeddings(self, x):
        return x

        

