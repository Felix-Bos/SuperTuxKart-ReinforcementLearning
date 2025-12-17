import torch
from .baseBuffers import ReplayBuffer
from .types import Transition, Batch

class stepReplayBuffer(ReplayBuffer):

    def __init__(self, capacity: int, device: torch.device):
        super().__init__(capacity=capacity, device=device)

    def add(self, transition: Transition):

        if self.__len__() < self.capacity:
            self.storage.append(transition)
            self.size += 1
        else:
            self.ptr = (self.ptr + 1) % self.capacity
            self.storage[self.ptr] = transition

    def can_sample(self, batch_size: int) -> bool:
        return len(self.storage) >= batch_size

    def sample(self, batch_size: int, recent:bool=False) -> Batch:
        
        assert self.can_sample(batch_size), "Not enough samples to draw from the buffer."

        indices = torch.randint(0, len(self.storage), (batch_size,))
        return Batch([self.storage[i] for i in indices])
    


        
        
