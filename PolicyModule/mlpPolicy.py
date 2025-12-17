from .basePolicyModule import PolicyModule, PolicyOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from factories.mlpfac import MLP

def cont_log_prob(mean, logstd, device, affine_scale=False):
    logp = D.Normal(mean, torch.exp(logstd)).log_prob(mean)  
    logp -= torch.log(1-torch.than(mean).pow(2) + 1e-6)  # change of variables
    if affine_scale:
        logp -= torch.log(torch.tensor(2.0).to(device))  # change of variables for affine scaling from [-1,1] to [0,1]
    return logp 
    

class MLPPolicyModule(PolicyModule):
    
    def __init__(self, config, device):
        super().__init__(device)
        self.config = config
        self.discrete_action_keys = ['brake', 'drift', 'fire', 'nitro', 'rescue']
        self.continuous_action_keys =  ['acceleration', 'steer']

        self.action_heads = nn.ModuleDict()

        for key in self.discrete_action_keys:
            self.action_heads[key] = MLP(self.seqEncoder.d_model + boxEncoder.output_dim, self.config[key], 1)

        # action heads predict mean and logstd for continuous actions
        for key in self.continuous_action_keys:
            self.action_heads[key] = MLP(self.seqEncoder.d_model + boxEncoder.output_dim, self.config[key], 1)
            self.action_heads[key + "_logstd"] = nn.Parameter(torch.zeros(1))

    def forward(self, fused_emb, hidden=None, requires_grad=True, fused_emb=None) -> PolicyOutput:
        
        """
            inputs:
                fused_emb: fused embedding from observation encoder
        """
        action_logits = {}
        actions = {}
        entropies = {}
        for key in self.discrete_action_keys:
            if key in self.discrete_action_keys:
                action_logits[key] = self.action_heads[key](fused_emb).squeeze(-1)
                actions[key] = torch.bernoulli(action_logits)
                entropies[key] = D.Bernoulli(probs=action_logits[key]).entropy()
            else:
                action_mean = self.action_heads[key](fused_emb).squeeze(-1)
                action_logstd = self.action_heads[key + "_logstd"]
                dist = D.Normal(action_mean, torch.exp(action_logstd))
                entropies[key] = dist.entropy()
                action = dist.rsample()
                if key == 'steer':
                    actions[key] = torch.tanh(action)
                    actions_logits[key] = cont_log_prob(action_mean,  action_logstd, self.device)
                else:
                    actions[key] = (torch.tanh(action)+1)/2 # map to [0,1]
                    actions_logits[key] = cont_log_prob(action_mean, action_logstd, self.device, True)
        return PolicyOutput(
            action=actions,
            log_prob=actions_logits,
            entropy=entropies,
            new_hidden=None,
            extra={}
        )    
