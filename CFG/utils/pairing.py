import torch
import scipy.optimize

@torch.no_grad()
def pair_samples(X0, X1, strategy='sliced_sorting'):
    """
    Pairs samples from X0 and X1 based on the selected strategy.
    
    Args:
        X0: Noise samples [B, ...]
        X1: Data samples [B, ...]
        strategy: 'random', 'sliced_sorting', or 'minibatch_ot'
        
    Returns:
        X0, X1_paired: Tensors on the original device.
    """
    assert X0.shape == X1.shape, f"Shape mismatch: {X0.shape} vs {X1.shape}"
    B = X0.shape[0]
    
    if strategy == 'random':
        # 1. Independent Coupling (O(1) overhead)
        idx = torch.randperm(B, device=X0.device)
        return X0, X1[idx]
        
    elif strategy == 'sliced_sorting':
        # 2. 1D Sliced Optimal Transport (O(N log N))
        X0_flat = X0.view(B, -1)
        X1_flat = X1.view(B, -1)
        D = X0_flat.shape[1]
        
        # Random projection to 1D
        v = torch.randn(D, device=X0.device, dtype=X0.dtype)
        v = v / torch.norm(v)
        
        proj0 = X0_flat @ v
        proj1 = X1_flat @ v
        
        # Sort based on 1D values
        idx0 = torch.argsort(proj0)
        idx1 = torch.argsort(proj1)
        
        # Rearrange X1 so that its sorted indices match X0's sorted indices
        X1_paired = torch.empty_like(X1)
        X1_paired[idx0] = X1[idx1]
        
        return X0, X1_paired
        
    elif strategy == 'minibatch_ot':
        # 3. Mini-Batch Exact Optimal Transport (O(N^3) via Hungarian)
        X0_flat = X0.view(B, -1)
        X1_flat = X1.view(B, -1)
        
        # Efficient L2 pairwise cost matrix calculation
        C = torch.cdist(X0_flat, X1_flat, p=2)
        
        # Detach and move to CPU for scipy Hungarian algorithm
        C_cpu = C.detach().cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(C_cpu)
        
        # row_ind is always [0, 1, ..., B-1]. col_ind provides the optimal permutation.
        idx1 = torch.tensor(col_ind, device=X0.device)
        return X0, X1[idx1]
        
    else:
        raise ValueError(f"Unknown pairing strategy: {strategy}")
