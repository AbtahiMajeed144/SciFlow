import torch


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains shadow copies of all trainable parameters, updated with:
        shadow = beta * shadow + (1 - beta) * param
    
    Use apply_shadow() before inference to swap in EMA weights,
    and restore() to swap back to active training weights.
    """
    def __init__(self, model, beta=0.9999):
        self.beta = beta
        self.shadow = {}
        self.backup = {}
        # Register shadow copies of all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model):
        """Update shadow weights after an optimizer step."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].sub_((1.0 - self.beta) * (self.shadow[name] - param.data))

    def apply_shadow(self, model):
        """Backup active weights and apply EMA weights for inference."""
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore active weights after inference."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Return serializable state for checkpointing."""
        return {
            'beta': self.beta,
            'shadow': {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state):
        """Restore EMA state from a checkpoint."""
        self.beta = state['beta']
        self.shadow = {k: v.clone() for k, v in state['shadow'].items()}
