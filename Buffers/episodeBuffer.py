# episodeReplayBuffer.py
import torch
from dataclasses import asdict, is_dataclass
from collections import defaultdict, deque
from typing import List, Any

from .baseBuffers import ReplayBuffer
from .types import Transition, Batch
from utils.buffer_utils import transitions_to_batch

class episodeReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, Lmax:int, device: torch.device):
        super().__init__(capacity=capacity, device=device)

        '''
        capacity: maximum number of transitions to store in the buffer
        Lmax: maximum length for variable length observations
        device: device to store the buffer on
        '''

        self.storage = []
        self.episodes = []        # (start, length)
        self._current_start = 0
        self.Lmax = Lmax

    def add(self, transition: Transition):
        self.storage.append(transition)

        if transition.terminated or transition.truncated:
            length = len(self.storage) - self._current_start
            self.episodes.append((self._current_start, length))
            self._current_start = len(self.storage)

        if len(self.storage) > self.capacity:
            self.storage.pop(0)
            self.episodes = [
                (s - 1, l) for s, l in self.episodes if s > 0
            ]

    def can_sample(self, batch_size: int) -> bool:
        return len(self.episodes) >= batch_size

    def sample(self, batch_size: int, recent:bool=False):
        assert self.can_sample(batch_size), "Not enough episodes to draw from the buffer."
        if recent:
            idxs = self.episodes[-1]
        else:
            idxs = torch.randint(0, len(self.episodes), (batch_size,))
        batches = []

        for i in idxs:
            start, length = self.episodes[i]
            traj = self.storage[start : start + length]
            batches.append(traj)
        batches = transitions_to_batch(batches, self.Lmax, self.device)
        return batches

