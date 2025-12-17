# algorithms/base_algo.py
from abc import ABC, abstractmethod
import torch
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from .types import Transition, Batch

from typing import Dict, Any, Optional, Literal
from PolicyModule.basePolicyModule import PolicyModule
from ValueModule.baseValueModule import ValueModule, QModule, SRModule
from buffers.base_buffer import ReplayBuffer
from buffers.types import Transition, Batch
from Agents.baseAgent import Agent
from typing import Type, Dict, Any

## Helper 

# The "Phonebook" dictionary
ALGORITHM_REGISTRY: Dict[str, Type] = {}

def register_algorithm(name: str):
    """
    A decorator to register a class under a string name.
    Usage: @register_algorithm("PPO")
    """
    def _register(cls):
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return _register

def get_algorithm(name: str):
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' is not registered. Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name]


class RLAlgorithm(ABC):
    def __init__(
        self,
        agent: Agent,
        config: Dict[str, Any],
    ):
        self.agent   = agent
        self.config  = config

    @abstractmethod
    def select_action(self, obs, actions:Optional=None, hidden=None, eval_mode=False):
        """ 
        Calls forward pass through the policy to select an action.
        Returns: tuple of
            action
            log_prob (or None)
            value_estimate (or None)
            new_hidden (or None)
        """
        ...

    @abstractmethod
    def update(self, Batch: Batch) -> Dict[str, float]:
        """
        Called repeatedly by Trainer.
        Receives a batch of transitions from the replay buffer.

        Should:
          - preprocess the batch (i.e., obs handling: box and seq etc.) 
          - perform forward passes
          - compute losses
          - step optimizers

        Returns:
          - dict of losses or other info for logging
            - e.g., Critic loss, Actor loss, Entropy, etc.
            - key should be in the format "<Metric>/<Type>"
                e.g., "Loss/Actor", "Loss/Critic", "Stats/Entropy" etc.
        """
        ...

    def reset_hidden(self):
        return None
