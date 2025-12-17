import torch
import torch.nn.functional as F
from algorithms.base_algo import RLAlgorithm
from algorithms.registry import register_algorithm  # Assuming you have this registry setup
from buffers.types import Batch
from Agents.baseAgent import Agent 
from typing import Dict, Any, Tuple

# --- Hyperparameters ---
# You would define these in your config dict
# PPO_EPOCHS = 4
# PPO_CLIP_EPS = 0.2
# PPO_GAMMA = 0.99
# PPO_LAMBDA = 0.95

class PPO(RLAlgorithm):

    def __init__(
        config: Dict[str, Any],
    ):
        super().__init__(config)
        self.config = config

    def select_action(
        self,
        obs,
        actions,
        hidden=None,
        eval_mode=False
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Select action using the policy module.
        """
        fused_emb = self.agent.value_ObsEncoder(obs)
        fused_emb = self.agent.policy_ObsEncoder(obs)
        if self.agent.SRNet is not None:
            fused_SR_emb = self.agent.SRNet(fused_emb, actions)
            fused_emb = torch.cat([fused_emb, fused_SR_emb], dim=-1)
        if self.agent.ValueNet is not None:
            value_estimate = self.agent.ValueNet(fused_emb)
        if self.agent.QNet is not None:
            value_estimate = self.agent.QNet(fused_emb, actions)

        with torch.no_grad():
            policy_output = self.agent.policy(obs, hidden, eval_mode)
        if not eval_mode:
            return (
                policy_output.action,
                policy_output.log_prob,
                value_estimate,
                policy_output.new_hidden
            )
        else:
            return (
                policy_output.action,
                policy_output.new_hidden
            )

    def _compute_gae(deltas, dones, gamma, lam):
        """helper function to compute GAE advantages"""
        factor = torch.tensor([ (lam * gamma)**exponent for exponent in range(deltas.shape[1])], device=deltas.device)
        factors = torch.stack([factor]*(deltas.shape[0]), dim=0)
        GAE = factor * deltas

        return GAE.sum(dim=1)

    def update(
        self,
        batch: Batch
    ) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0

        fused_obs_emb = self.agent.ObsEncoder(batch.obs)
        next_fused_obs_emb = self.agent.ObsEncoder(batch.next_obs)
        for _ in range(self.config['ppo_epochs']):
            # Compute advantages and returns
            with torch.no_grad():
                values = self.agent.ValueNet(fused_obs_emb)
                next_values = self.agent.ValueNet(next_fused_obs_emb)
                deltas = batch.rewards + self.config['gamma'] * next_values * (1 - batch.dones) - values
                advantages = self._compute_gae(deltas, batch.dones, self.config['gamma'], self.config['lambda'])
                returns = advantages + values

            # Get current policy outputs
            policy_output = self.agent.policy(batch.obs)
            log_probs = policy_output.log_prob

            # Compute ratios
            ratios = torch.exp(log_probs - batch.log_probs)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.config['clip_eps'], 1.0 + self.config['clip_eps']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_estimates = self.agent.ValueNet(batch.obs)
            value_loss = F.mse_loss(value_estimates, returns)

            # Total loss
            total_loss = policy_loss + value_loss

            # Backpropagation
            self.agent.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)

            value_loss.backward()
            
            self.agent.policy.optimizer.step()
            self.agent.ValueNet.optimizer.step()

            # SRNet update if applicable
            if self.agent.SRNet is not None:
                sr_loss = self.agent.SRNet.compute_loss(batch)
                self.agent.SRNet.optimizer.zero_grad()
                sr_loss.backward()
                self.agent.SRNet.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            'policy_loss': total_policy_loss / self.config['ppo_epochs'],
            'value_loss': total_value_loss / self.config['ppo_epochs'],
            'returns_mean': returns.item(),
            'values_mean': values.mean().item(),
        }


    
