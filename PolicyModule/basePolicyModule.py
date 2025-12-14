# modules/policy.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class PolicyOutput:
    action: Any                  # torch.Tensor or np.array
    log_prob: Optional[Any]      # log π(a|s)
    entropy: Optional[Any]       # entropy bonus (can be None)
    new_hidden: Optional[Any]    # new recurrent state
    extra: dict                  # anything else (mu, std, logits, etc.)

class PolicyModule(ABC):
    @abstractmethod
    def act(
        self,
        obs,
        hidden: Optional[Any] = None,
        deterministic: bool = False,
        requires_grad: bool = False,
    ) -> PolicyOutput:
        """
        - obs: batch of observations (shape arbitrary)
        - hidden: recurrent hidden state or None
        - deterministic: use mean or mode instead of sampling
        - requires_grad: if True, returned action/log_prob must be differentiable
        """
        ...
