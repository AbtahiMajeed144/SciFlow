import torch
import torch.nn as nn
import math
from einops import rearrange

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
        denominator = torch.where(denominator.abs() < 1e-5, denominator.sign() * 1e-5, denominator)
        
        integral_elements = (-self.A / denominator) * (cos_1 - cos_0)
        h_1 = torch.sum(integral_elements, dim=(3, 4))
        return h_1

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class TimeAgnosticTinyDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x0):
        x = self.patch_embed(x0) 
        x = rearrange(x, 'b c h w -> b (h w) c') 
        
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x

class KARTFlowModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            # Fallback to defaults
            img_size, patch_size, in_channels, hidden_dim, depth, heads = 32, 4, 3, 128, 4, 4
            q, k = 2, 2
        else:
            m_cfg = config['model']
            img_size, patch_size, in_channels = m_cfg['img_size'], m_cfg['patch_size'], m_cfg['in_channels']
            hidden_dim, depth, heads = m_cfg['hidden_dim'], m_cfg['depth'], m_cfg['heads']
            q, k = m_cfg['kart_nodes'], m_cfg['kart_harmonics']

        self.M = TimeAgnosticTinyDiT(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            hidden_dim=hidden_dim, depth=depth, heads=heads
        )
        
        out_features = patch_size * patch_size * in_channels
        self.K = FourierKARTLayer(in_features=hidden_dim, out_features=out_features, num_nodes=q, harmonics=k)
        
        self.img_size = img_size
        self.in_channels = in_channels
        
    def forward(self, x0, t):
        X_features = self.M(x0)
        V_patches = self.K(X_features, t)
        h = w = self.img_size // self.M.patch_size
        p1 = p2 = self.M.patch_size
        c = self.in_channels
        V_pred = rearrange(V_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p1, p2=p2, c=c)
        return V_pred
        
    def integrate_1step(self, x0):
        X_features = self.M(x0)
        h_1_patches = self.K.integrate_1step(X_features)
        h = w = self.img_size // self.M.patch_size
        p1 = p2 = self.M.patch_size
        c = self.in_channels
        h_1 = rearrange(h_1_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p1, p2=p2, c=c)
        return h_1
