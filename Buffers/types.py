# buffers/types.py
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
import torch

@dataclass
class Transition:
    states: Any
    action: Any
    reward: Optional[float] = None
    next_states: Optional[Any] = None
    truncated: Optional[bool] = None
    terminated: Optional[bool] = None

    episode_id: Optional[int] = None
    step_id: Optional[int] = None

    log_prob: Optional[Any] = None
    value: Optional[Any] = None
    hidden: Optional[Any] = None
    next_hidden: Optional[Any] = None

@dataclass
class Batch:
    
    # fixed-size observations
    box_obs: Dict[str, torch.Tensor]

    # variable-length observations (padded)
    seq_obs: Dict[str, List] # Batch by Lmax by feature_dim
    seq_mask: Dict[str, torch.Tensor]

    disc_actions: Dict[str, torch.Tensor]
    cont_actions: Dict[str, torch.Tensor]
    rewards: torch.Tensor
    dones: torch.Tensor

    next_box_obs: Dict[str, torch.Tensor] = None
    next_seq_obs: Dict[str, torch.Tensor] = None

    # masks
    next_seq_mask: Dict[str, torch.Tensor] = None
    done_masks: torch.Tensor = None
    reward_masks: torch.Tensor = None
    disc_actions_mask: Dict[str, torch.Tensor] = None
    cont_actions_mask: Dict[str, torch.Tensor] = None

    extras: Dict[str, torch.Tensor] = None
    extras_mask: Dict[str, torch.Tensor] = None

