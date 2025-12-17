import torch
import torch.nn as nn
import factories.mlpfac as MLP
import torch.optimizer

class MLPValueModule(nn.Module):
    
    def __init__(self, input_dim, config, output_dim, device, optimizer):
        super().__init__()
        self.device = device
        self.config = config
        self.values = MLP(input_dim, config, output_dim, device)

    def forward(self, fused_obs_emb):
        return self.values(fused_obs_emb.to(self.device))
