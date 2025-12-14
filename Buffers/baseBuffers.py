# buffers/base_buffer.py
from abc import ABC, abstractmethod
from typing import Literal
import torch
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from .types import Transition, Batch

class ReplayBuffer(ABC):
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.storage = [] 
        self._episode_counter = defaultdict(int)
        self.device = device 

    def __len__(self):
        return len(self.storage)

    @abstractmethod
    def add(self, transition: Transition):
        """Add a single transition (or sequence, if you want to overload it)."""
        ...
    @abstractmethod
    def can_sample(self, batch_size: int) -> bool:
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        """
        sample a batch of transitions/sequences/trajectories from the buffer
        for asynchronous vectorized environments, samples are drawn across all environments
        Returns:
        Batch: a Batch object containing the sampled data
        """
        ...
