import torch
import torch.nn as nn
import math


class FourierKARTLayer(nn.Module):
    """
    Fourier-KART velocity predictor with harmonic basis.
    
    Implements:
        v(t) = sum_{q,k} A_{d,q,k} * sin(k * (C_q(x) + w_q * t) + B_{d,q,k})
    
    The analytical integral from t=0 to t=1 is computed in closed form
    via the standard antiderivative of sin(w*t + theta).
    """
    def __init__(self, in_features, out_features, num_nodes=128, harmonics=6):
        super().__init__()
        self.D_in = in_features
        self.D_out = out_features
        self.Q = num_nodes
        self.K = harmonics
        self.register_buffer('k_vec', torch.arange(1, self.K + 1, dtype=torch.float32))
        
        # Spatial routing
        self.W_c = nn.Linear(self.D_in, self.Q)
        
        # Physics parameter (requires zero weight decay in optimizer)
        self.w = nn.Parameter(torch.randn(self.Q))
        
        # Amplitude and phase
        self.A = nn.Parameter(torch.randn(self.D_out, self.Q, self.K) / math.sqrt(self.Q * self.K))
        self.B = nn.Parameter(torch.rand(self.D_out, self.Q, self.K) * 2 * math.pi)

    def forward(self, X0, t):
        """
        Evaluate instantaneous velocity at time t.
        
        Args:
            X0: Blueprint features [B, S, D_in]
            t:  Time values [B, 1] or float
        Returns:
            V: Velocity field [B, S, D_out]
        """
        B_dim = X0.shape[0]
        if isinstance(t, float):
            t = torch.full((B_dim, 1), t, device=X0.device, dtype=X0.dtype)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
            
        C = self.W_c(X0)                                           # [B, S, Q]
        time_term = (self.w.unsqueeze(0) * t).unsqueeze(1)          # [B, 1, Q]
        angle = C + time_term                                       # [B, S, Q]
        angle_k = angle.unsqueeze(-1) * self.k_vec                  # [B, S, Q, K]
        inner_term = angle_k.unsqueeze(2) + self.B                  # [B, S, D_out, Q, K]
        sin_eval = torch.sin(inner_term)
        
        V = torch.sum(self.A * sin_eval, dim=(3, 4))                # [B, S, D_out]
        return V

    def integrate_to_t(self, X0, t):
        """
        Analytical integral of the Fourier basis from s=0 to s=t.
        
        Uses closed-form: ∫₀ᵗ A * sin(ω·s + θ) ds
            = A * [-cos(ω·s + θ) / ω] evaluated at boundaries s=0 and s=t
            = A * (cos(θ) - cos(ω·t + θ)) / ω
        
        Args:
            X0: Blueprint features [B, S, D_in]
            t:  Time values [B, 1] or float
        Returns:
            h_t: Displacement integral [B, S, D_out]
        """
        B_dim = X0.shape[0]
        if isinstance(t, float):
            t = torch.full((B_dim, 1), t, device=X0.device, dtype=X0.dtype)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        C = self.W_c(X0)                                                    # [B, S, Q]
        angle_k_0 = (C.unsqueeze(-1) * self.k_vec).unsqueeze(2)             # [B, S, 1, Q, K]
        
        # Compute C + w*t for each sample (variable upper bound)
        w_t = self.w.unsqueeze(0) * t                                       # [B, Q]
        C_plus_wt = C + w_t.unsqueeze(1)                                    # [B, S, Q]
        angle_k_t = (C_plus_wt.unsqueeze(-1) * self.k_vec).unsqueeze(2)     # [B, S, 1, Q, K]
        
        theta = angle_k_0 + self.B                                          # [B, S, D_out, Q, K]
        omega_t_plus_theta = angle_k_t + self.B                             # [B, S, D_out, Q, K]
        
        omega = (self.w.view(self.Q, 1) * self.k_vec.view(1, self.K)).view(1, 1, 1, self.Q, self.K)
        
        # ∫₀ᵗ sin(ω·s + θ) ds = (cos(θ) - cos(ω·t + θ)) / ω
        integral_elements = self.A * (torch.cos(theta) - torch.cos(omega_t_plus_theta)) / (omega + 1e-5)
        h_t = torch.sum(integral_elements, dim=(3, 4))                      # [B, S, D_out]
        return h_t

    def integrate_1step(self, X0):
        """
        Analytical integral of the Fourier basis from t=0 to t=1.
        Convenience wrapper for inference-time 1-step generation.
        
        Args:
            X0: Blueprint features [B, S, D_in]
        Returns:
            h_1: Displacement integral [B, S, D_out]
        """
        return self.integrate_to_t(X0, 1.0)
