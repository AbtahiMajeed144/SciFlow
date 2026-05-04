import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
# Sin-Cos Positional Embedding (from DiT / MAE)
# ──────────────────────────────────────────────────────────────────────────────
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2D sin-cos positional embedding.
    Args:
        embed_dim: embedding dimension
        grid_size: int, height and width of the grid
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def _get_1d_sincos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)

class FourierKARTLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=2, harmonics=2):
        super().__init__()
        self.D_in = in_features
        self.D_out = out_features
        self.Q = num_nodes
        self.K = harmonics
        self.register_buffer('k_vec', torch.arange(1, self.K + 1, dtype=torch.float32))
        self.W_c = nn.Linear(self.D_in, self.Q)
        self.w = nn.Parameter(torch.randn(self.Q))
        self.A = nn.Parameter(torch.randn(self.D_out, self.Q, self.K) / math.sqrt(self.Q * self.K))
        self.B = nn.Parameter(torch.rand(self.D_out, self.Q, self.K) * 2 * math.pi)

    def forward(self, X0, t):
        B_dim = X0.shape[0]
        if isinstance(t, float):
            t = torch.full((B_dim, 1), t, device=X0.device, dtype=X0.dtype)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
            
        C = self.W_c(X0)
        time_term = (self.w.unsqueeze(0) * t).unsqueeze(1) 
        angle = C + time_term 
        angle_k = angle.unsqueeze(-1) * self.k_vec 
        inner_term = angle_k.unsqueeze(2) + self.B 
        sin_eval = torch.sin(inner_term)
        V = torch.sum(self.A * sin_eval, dim=(3, 4)) 
        return V

    def integrate_1step(self, X0):
        C = self.W_c(X0) 
        angle_k_0 = (C.unsqueeze(-1) * self.k_vec).unsqueeze(2) 
        C_plus_w = C + self.w.view(1, 1, self.Q) 
        angle_k_1 = (C_plus_w.unsqueeze(-1) * self.k_vec).unsqueeze(2) 
        
        cos_1 = torch.cos(angle_k_1 + self.B) 
        cos_0 = torch.cos(angle_k_0 + self.B) 
        
        denominator = (self.w.view(self.Q, 1) * self.k_vec.view(1, self.K)) 
        denominator = denominator.view(1, 1, 1, self.Q, self.K)
        
        # Ensure sign is never exactly 0 to prevent division by zero
        sign = denominator.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        denominator = torch.where(denominator.abs() < 1e-5, sign * 1e-5, denominator)
        
        integral_elements = (-self.A / denominator) * (cos_1 - cos_0)
        h_1 = torch.sum(integral_elements, dim=(3, 4))
        return h_1

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        # Note: adaLN zero-init is handled centrally in initialize_weights()

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        norm1_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm1_x, norm1_x, norm1_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        norm2_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm2_x)
        return x

class TimeAgnosticTinyDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, heads=4, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        # Learnable parameter, initialized with sin-cos values in initialize_weights()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        
        self.class_emb = nn.Embedding(num_classes + 1, hidden_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Apply DiT initialization protocol
        self.initialize_weights()

    def initialize_weights(self):
        # 1. Xavier uniform for all Linear layers, zero bias
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # 2. Sin-cos positional embedding
        grid_size = int(self.num_patches ** 0.5)
        pos_embed_np = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float().unsqueeze(0))
        
        # 3. Patch embed Conv2d: Xavier uniform (treat as Linear)
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.bias, 0)
        
        # 4. Class embedding: normal(std=0.02)
        nn.init.normal_(self.class_emb.weight, std=0.02)
        
        # 5. Zero-init all adaLN modulation outputs (identity mapping at start)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x0, labels):
        x = self.patch_embed(x0) 
        x = rearrange(x, 'b c h w -> b (h w) c') 
        
        x = x + self.pos_embed
        c = self.class_emb(labels)
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.norm(x)
        return x

class KARTFlowModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            # Fallback to defaults
            img_size, patch_size, in_channels, hidden_dim, depth, heads = 32, 4, 3, 128, 4, 4
            q, k = 2, 2
            num_classes = 10
        else:
            m_cfg = config['model']
            img_size, patch_size, in_channels = m_cfg['img_size'], m_cfg['patch_size'], m_cfg['in_channels']
            hidden_dim, depth, heads = m_cfg['hidden_dim'], m_cfg['depth'], m_cfg['heads']
            q, k = m_cfg['kart_nodes'], m_cfg['kart_harmonics']
            num_classes = config.get('cfg', {}).get('num_classes', 10)

        self.M = TimeAgnosticTinyDiT(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            hidden_dim=hidden_dim, depth=depth, heads=heads, num_classes=num_classes
        )
        
        out_features = patch_size * patch_size * in_channels
        self.K = FourierKARTLayer(in_features=hidden_dim, out_features=out_features, num_nodes=q, harmonics=k)
        
        self.img_size = img_size
        self.in_channels = in_channels
        
    def forward(self, x0, t, labels):
        X_features = self.M(x0, labels)
        V_patches = self.K(X_features, t)
        h = w = self.img_size // self.M.patch_size
        p1 = p2 = self.M.patch_size
        c = self.in_channels
        V_pred = rearrange(V_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p1, p2=p2, c=c)
        return V_pred
        
    def integrate_1step(self, x0, labels):
        X_features = self.M(x0, labels)
        h_1_patches = self.K.integrate_1step(X_features)
        h = w = self.img_size // self.M.patch_size
        p1 = p2 = self.M.patch_size
        c = self.in_channels
        h_1 = rearrange(h_1_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p1, p2=p2, c=c)
        return h_1
