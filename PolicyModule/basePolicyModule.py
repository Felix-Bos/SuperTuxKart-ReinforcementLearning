# modules/policy.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class PolicyOutput:
    action: Dict                  # torch.Tensor or np.array
    log_prob: Optional[Any]      # log π(a|s)
    entropy: Optional[Any]       # entropy bonus (can be None)
    new_hidden: Optional[Any]    # new recurrent state
    extra: dict                  # anything else (mu, std, logits, etc.)

class PolicyModule(ABC):
    def __init__(self, device):
        self.device = device
