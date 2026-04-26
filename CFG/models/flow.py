import torch
import torch.nn as nn
from einops import rearrange

from .dit import TimeAgnosticDiT
from .kart import FourierKARTLayer


class KARTFlowModel(nn.Module):
    """
    Complete KART-Flow model combining Module M (TimeAgnosticDiT)
    and Module K (FourierKARTLayer) with Classifier-Free Guidance support.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            # Fallback to defaults matching roadmap spec
            model_size = 'small'
            img_size, in_channels = 32, 4
            q, k = 128, 6
            num_classes = 10
        else:
            m_cfg = config['model']
            model_size = m_cfg.get('model_size', 'small')
            img_size = m_cfg.get('img_size', 32)
            in_channels = m_cfg.get('in_channels', 4)
            q, k = m_cfg.get('kart_nodes', 128), m_cfg.get('kart_harmonics', 6)
            num_classes = config.get('cfg', {}).get('num_classes', 10)

        self.M = TimeAgnosticDiT(
            model_size=model_size, img_size=img_size, in_channels=in_channels, num_classes=num_classes
        )
        
        # Derive hidden_dim from the DiT's SIZE_CONFIGS (DRY — no duplicated dict)
        hidden_dim = TimeAgnosticDiT.SIZE_CONFIGS.get(model_size, TimeAgnosticDiT.SIZE_CONFIGS['small'])['hidden_size']
        
        out_features = self.M.patch_size * self.M.patch_size * in_channels
        self.K = FourierKARTLayer(in_features=hidden_dim, out_features=out_features, num_nodes=q, harmonics=k)
        
        self.img_size = img_size
        self.in_channels = in_channels
    
    def _unpack_patches(self, patches):
        """Rearrange patch tokens [B, S, patch_dim] back to spatial tensor [B, C, H, W]."""
        h = w = self.img_size // self.M.patch_size
        p1 = p2 = self.M.patch_size
        c = self.in_channels
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p1, p2=p2, c=c)
        
    def forward(self, x0, t, labels, return_delta_x=False):
        X_features = self.M(x0, labels)
        V_patches = self.K(X_features, t)
        V_pred = self._unpack_patches(V_patches)
        
        if return_delta_x:
            h_1_patches = self.K.integrate_1step(X_features)
            delta_x = self._unpack_patches(h_1_patches)
            return V_pred, delta_x
            
        return V_pred
        
    def integrate_1step(self, x0, labels):
        X_features = self.M(x0, labels)
        h_1_patches = self.K.integrate_1step(X_features)
        return self._unpack_patches(h_1_patches)
        
    def generate_with_cfg(self, x0, labels, guidance_scale, num_classes):
        """CFG-guided 1-step generation via blueprint extrapolation."""
        # Conditional blueprint
        blueprint_cond = self.M(x0, labels)
        
        # Unconditional blueprint (null class token)
        null_labels = torch.full_like(labels, num_classes)
        blueprint_uncond = self.M(x0, null_labels)
        
        # CFG Extrapolation: blueprint = uncond + w * (cond - uncond)
        blueprint = blueprint_uncond + guidance_scale * (blueprint_cond - blueprint_uncond)
        
        # KART Layer Analytical Integral
        h_1_patches = self.K.integrate_1step(blueprint)
        return self._unpack_patches(h_1_patches)
