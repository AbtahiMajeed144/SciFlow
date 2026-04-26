import torch
import torch.nn as nn
from einops import rearrange


def modulate(x, shift, scale):
    """Adaptive Layer Normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN conditioning (class-conditioned, no timestep)."""
    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # 6-way modulation: shift/scale/gate for MSA and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        norm1_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm1_x, norm1_x, norm1_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        norm2_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm2_x)
        return x


class TimeAgnosticDiT(nn.Module):
    """
    Time-agnostic Diffusion Transformer (Module M / Blueprint Generator).
    
    Accepts latent input and class labels, outputs spatial blueprint features.
    No timestep embedding — conditioning is purely class-based via AdaLN.
    Output shape: [Batch, num_patches, hidden_size]
    """
    
    SIZE_CONFIGS = {
        'small': {'hidden_size': 384, 'depth': 12, 'heads': 6},
        'base': {'hidden_size': 768, 'depth': 12, 'heads': 12},
        'large': {'hidden_size': 1024, 'depth': 24, 'heads': 16},
        'xl': {'hidden_size': 1152, 'depth': 28, 'heads': 16}
    }
    
    def __init__(self, model_size='small', img_size=32, in_channels=4, num_classes=10):
        super().__init__()
        
        cfg = self.SIZE_CONFIGS.get(model_size, self.SIZE_CONFIGS['small'])
        hidden_dim = cfg['hidden_size']
        depth = cfg['depth']
        heads = cfg['heads']
        
        self.patch_size = 2
        self.num_patches = (img_size // self.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        # +1 for the null token (CFG unconditional path)
        self.class_emb = nn.Embedding(num_classes + 1, hidden_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x0, labels):
        x = self.patch_embed(x0) 
        x = rearrange(x, 'b c h w -> b (h w) c') 
        
        x = x + self.pos_embed
        c = self.class_emb(labels)
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.norm(x)
        return x
