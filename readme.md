# KART-Flow MVP: Analytical 1-Step Lagrangian Flow Matching

## 1. Project Objective & Conceptual Context
You are tasked with building a completely novel generative image model called **KART-Flow**. This framework abandons standard Eulerian continuous ODE solvers. Instead, it tracks fluid particles in a Lagrangian coordinate frame (anchored at $x_0$) and predicts the displacement velocity using a Kolmogorov-Arnold Representation Theorem (KART) network equipped with a global Fourier basis. 

Because the temporal basis is strictly analytical (sine/cosine) and time is injected linearly ($w_q t$), the time variable $t$ is analytically integrated out during inference. This yields mathematically exact 1-step image generation without the truncation errors of numerical ODE solvers (like Runge-Kutta or Euler).

**Conceptual Anchor for the Agent:** Do not treat this as standard Flow Matching. Treat this objective as a hybrid between **Rectified MeanFlow** and **Koopman Diffusion**. 
* Like Koopman Diffusion, we linearize the temporal evolution ($w_q t$) to achieve a 1-step mathematical jump, forcing all non-linear complexity into the spatial contextualizer $M(x_0)$. 
* Like MeanFlow, our goal is to bypass the ODE solver entirely for inference.

## 2. Reference Repositories
Before writing the code, review these repositories to anchor your architecture:
1. **`Xinxi-Zhang/Re-MeanFlow`**: Study this repo specifically for the **Minibatch Optimal Transport** (Rectified Couplings) logic and the Continuous Flow Matching MSE loss on image datasets.
2. **`NVlabs/rcm`**: Study this for state-of-the-art templates on structuring a discrete 1-step Transformer (DiT) framework without numerical ODE solvers.

## 3. Architectural Components (MVP on CIFAR-10)
Do not use high-level `diffusers` pipelines. Build this using native PyTorch. We are testing in pure pixel space to validate the mathematical anti-derivative.

* **Dataset:** Unconditional CIFAR-10 ($3 \times 32 \times 32$ pixels, scaled to $[-1, 1]$). Flattened dimensionality $D = 3072$.
* **Module M (Spatial Contextualizer):**
    * Implement a tiny Vision/Diffusion Transformer (DiT).
    * Patch size: 4. Layers: 4. Heads: 4. Hidden dimension: 128.
    * **CRITICAL DEVIATION:** There is NO time ($t$) input and NO text condition. Remove all AdaLN time-conditioning blocks. The `forward` pass takes ONLY the unpatched noise $x_0 \in \mathbb{R}^{B \times 3 \times 32 \times 32}$.
    * Output: A contextualized state $X_0$. Project this back to a flattened tensor $X_0 \in \mathbb{R}^{B \times 3072}$.
* **Module K (Fourier-KART Layer):**
    * Do not build this from scratch. Use the exact PyTorch class provided in **Section 4**.

## 4. The Fourier-KART PyTorch Module
Use this exact code for the velocity predictor. It prevents shape mismatches and handles the analytical integration safely. Hyperparameters for MVP: `num_nodes (Q) = 2`, `harmonics (K) = 2`.

```python
import torch
import torch.nn as nn
import math

class FourierKARTLayer(nn.Module):
    def __init__(self, in_features, num_nodes=2, harmonics=2):
        """
        in_features (D): The flattened spatial dimension (e.g., 3 * 32 * 32 = 3072)
        num_nodes (Q): Number of KART nodes/rotational axes
        harmonics (K): Number of Fourier frequencies
        """
        super().__init__()
        self.D = in_features
        self.Q = num_nodes
        self.K = harmonics
        
        self.register_buffer('k_vec', torch.arange(1, self.K + 1, dtype=torch.float32))
        
        # M(x0) to Initial Phase Mapping: C_q(X_0)
        self.W_c = nn.Linear(self.D, self.Q)
        
        # Learnable Time Frequencies: w_q
        self.w = nn.Parameter(torch.randn(self.Q))
        
        # Outer Amplitudes and Phase Shifts: A_{i,q,k} and B_{i,q,k}
        self.A = nn.Parameter(torch.randn(self.D, self.Q, self.K) / math.sqrt(self.Q * self.K))
        self.B = nn.Parameter(torch.rand(self.D, self.Q, self.K) * 2 * math.pi)

    def forward(self, X0, t):
        """
        Standard forward pass for instantaneous velocity training.
        X0: [Batch, D] - The static contextualized noise from Module M
        t: [Batch, 1] or scalar - The current time
        Returns instantaneous velocity V: [Batch, D]
        """
        B_dim = X0.shape[0]
        if isinstance(t, float):
            t = torch.full((B_dim, 1), t, device=X0.device, dtype=X0.dtype)
            
        C = self.W_c(X0)
        time_term = self.w.unsqueeze(0) * t 
        angle = C + time_term
        angle_k = angle.unsqueeze(-1) * self.k_vec.view(1, 1, self.K)
        inner_term = angle_k.unsqueeze(1) + self.B.unsqueeze(0)
        
        sin_eval = torch.sin(inner_term)
        V = torch.sum(self.A.unsqueeze(0) * sin_eval, dim=(2, 3))
        return V

    def integrate_1step(self, X0):
        """
        Analytical Definite Integral from t=0 to t=1 for 1-step generation.
        Returns total displacement h(1): [Batch, D]
        """
        C = self.W_c(X0)
        angle_k_0 = (C.unsqueeze(-1) * self.k_vec.view(1, 1, self.K)).unsqueeze(1)
        C_plus_w = C + self.w.unsqueeze(0)
        angle_k_1 = (C_plus_w.unsqueeze(-1) * self.k_vec.view(1, 1, self.K)).unsqueeze(1)
        B_unsqueezed = self.B.unsqueeze(0)
        
        cos_1 = torch.cos(angle_k_1 + B_unsqueezed)
        cos_0 = torch.cos(angle_k_0 + B_unsqueezed)
        
        denominator = (self.k_vec.view(1, 1, self.K) * self.w.view(1, self.Q, 1)).unsqueeze(1)
        denominator = torch.where(denominator.abs() < 1e-5, denominator.sign() * 1e-5, denominator)
        
        integral_elements = (-self.A.unsqueeze(0) / denominator) * (cos_1 - cos_0)
        h_1 = torch.sum(integral_elements, dim=(2, 3))
        return h_1
```

## 5. The Training Pipeline (Instantaneous Flow)
We train the model to match the instantaneous vector field of a straightened Optimal Transport path.

1. **Sample:** Batch of data $x_1 \sim$ CIFAR-10. Batch of noise $x_0 \sim \mathcal{N}(0, I)$.
2. **Minibatch OT (Crucial):** Flatten $x_0, x_1$ to `[B, 3072]`. Use `scipy.optimize.linear_sum_assignment` based on Euclidean distance to perfectly pair $x_0$ and $x_1$ across the batch. (See `Re-MeanFlow` for implementation).
3. **Time Sampling:** Sample random times $t \sim \mathcal{U}(0, 1)$.
4. **Target:** Compute target velocity: $v_{target} = x_1 - x_0$.
5. **Forward Pass:**
   * Contextualize the initial noise: $X_0 = M(x_0)$. *(CRITICAL: The input to M is strictly $x_0$, NOT the intermediate state $x_t$. This is a Lagrangian frame).*
   * Predict Velocity: $V_{pred} = \text{FourierKARTLayer}(X_0, t)$.
6. **Loss:** Compute MSE: $|| V_{pred} - v_{target} ||^2_2$. Backpropagate through both KART and M.

## 6. The Inference Pipeline (Analytical 1-Step)
DO NOT write a numerical ODE loop (Euler/RK4). Generate the image in one mathematical jump.

1. Sample pure noise $x_0 \sim \mathcal{N}(0, I)$ and flatten to `[B, 3072]`.
2. Contextualize: $X_0 = M(x_0)$.
3. **Analytical Anti-Derivative:** Calculate total displacement: $h(1) = \text{FourierKARTLayer.integrate\_1step}(X_0)$.
4. **Kinematic Addition:** Calculate the final image: $x_{final} = x_0 + h(1)$.
5. Reshape $x_{final}$ back to `[B, 3, 32, 32]`, clamp to $[-1, 1]$, and save the image grid.

## 7. Execution Checkpoints for Agent
* [ ] Implement Time-Agnostic Tiny-DiT ($M$) and add the `FourierKARTLayer`.
* [ ] Implement the Minibatch OT pairing step in the dataloader/training loop.
* [ ] Execute the continuous training loop on CIFAR-10. Log the MSE loss.
* [ ] Execute the 1-step analytical inference function every 10 epochs and output sample grids.
* [ ] Provide an analysis of the loss curve behavior over the first 50 epochs.