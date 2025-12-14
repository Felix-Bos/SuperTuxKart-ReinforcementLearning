import torch
from dataclasses import is_dataclass

from .baseBuffers import ReplayBuffer
from .types import Transition, Batch
from utils.buffer_utils import transitions_to_batch_sequence


class sequenceReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        Lmax:int,
        device: torch.device,
    ):

        '''
        capacity: maximum number of transitions to store in the buffer
        sequence_length: length of the sequences to sample
        Lmax: maximum length for variable length observations
        device: device to store the buffer on
        '''
        super().__init__(capacity=capacity,  device=device)

        self.sequence_length = sequence_length
        self.storage = []
        self.valid_starts = []
        self._episode_start = 0
        self.Lmax = Lmax

    def add(self, transition: Transition):
        idx = len(self.storage)
        self.storage.append(transition)

        if idx - self._episode_start + 1 >= self.sequence_length:
            self.valid_starts.append(idx - self.sequence_length + 1)

        if transition.terminated or transition.truncated:
            self._episode_start = len(self.storage)

        if len(self.storage) > self.capacity:
            self.storage.pop(0)
            self.valid_starts = [s - 1 for s in self.valid_starts if s > 0]

    def can_sample(self, batch_size: int) -> bool:
        return len(self.valid_starts) >= batch_size

    def sample(self, batch_size: int):
        idxs = torch.randint(0, len(self.valid_starts), (batch_size,))
        batches = []

        for i in idxs:
            start = self.valid_starts[i]
            seq = self.storage[start : start + self.sequence_length]
            batches.append(seq)

        batches = transitions_to_batch_sequence(batches, self.Lmax, self.device)
        return batches
