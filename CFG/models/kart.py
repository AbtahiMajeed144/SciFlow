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

    def integrate_1step(self, X0):
        """
        Analytical integral of the Fourier basis from t=0 to t=1.
        
        Uses closed-form: integral of A * sin(w*t + theta) dt
            = A * [-cos(w*t + theta) / w]
        evaluated at boundaries t=0 and t=1.
        
        Args:
            X0: Blueprint features [B, S, D_in]
        Returns:
            h_1: Displacement integral [B, S, D_out]
        """
        C = self.W_c(X0)                                                    # [B, S, Q]
        angle_k_0 = (C.unsqueeze(-1) * self.k_vec).unsqueeze(2)             # [B, S, 1, Q, K]
        C_plus_w = C + self.w.view(1, 1, self.Q)                            # [B, S, Q]
        angle_k_1 = (C_plus_w.unsqueeze(-1) * self.k_vec).unsqueeze(2)      # [B, S, 1, Q, K]
        
        theta = angle_k_0 + self.B                                          # [B, S, D_out, Q, K]
        omega_plus_theta = angle_k_1 + self.B                               # [B, S, D_out, Q, K]
        
        omega = (self.w.view(self.Q, 1) * self.k_vec.view(1, self.K)).view(1, 1, 1, self.Q, self.K)
        
        # integral of sin(w*t + theta) = -cos(w*t + theta) / w
        # Evaluate: [-cos(omega_plus_theta)/omega] - [-cos(theta)/omega]
        #         = (cos(theta) - cos(omega_plus_theta)) / omega
        integral_elements = self.A * (torch.cos(theta) - torch.cos(omega_plus_theta)) / (omega + 1e-5)
        h_1 = torch.sum(integral_elements, dim=(3, 4))                      # [B, S, D_out]
        return h_1
