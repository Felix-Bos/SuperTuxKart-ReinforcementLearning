# algorithms/base_algo.py
from abc import ABC, abstractmethod
from typing import Literal
import torch
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from .types import Transition, Batch

from typing import Dict, Any, Optional, Literal
from modules.policy import PolicyModule
from modules.value import ValueModule, QModule, SRModule
from buffers.base_buffer import ReplayBuffer
from buffers.types import Transition, Batch

class RLAlgorithm(ABC):
    def __init__(
        self,
        policy: Optional[PolicyModule],
        value_fn: Optional[ValueModule],
        q_fn1: Optional[QModule],
        q_fn2: Optional[QModule],
        sr_fn: Optional[SRModule],
        buffer: ReplayBuffer,
        config: Dict[str, Any],
    ):
        self.policy  = policy
        self.value_fn = value_fn
        self.q_fn1   = q_fn1
        self.q_fn2   = q_fn2
        self.sr_fn   = sr_fn
        self.buffer  = buffer
        self.config  = config

    @abstractmethod
    def select_action(self, obs, hidden=None, eval_mode=False):
        """
        Returns:
            action
            log_prob (or None)
            value_estimate (or None)
            new_hidden (or None)
        """
        ...

    def store_transition(self, transition: Transition):
        self.buffer.add(transition)

    @abstractmethod
    def train_step(self):
        """
        Called repeatedly by Trainer.
        Should:
          - check buffer.can_sample
          - sample batch with appropriate mode
          - compute losses
          - step optimizers
        """
        ...

    def reset_hidden(self, batch_size=1):
        return None
