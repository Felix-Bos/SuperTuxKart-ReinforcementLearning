
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
from baseAlgorithm import RLAlgorithm

class Agent(ABC):
    """
     wrapper for interacting with agents'modules (policy, valueNet, qNet, SRNet) and RL algrithms
    """
    def __init__(
        self,
        obsEncoder: Optional[Any]=None,
        policy: Optional[PolicyModule]=None,
        valueNet: Optional[ValueModule]=None,
        qNet: Optional[QModule]=None,
        SRNet: Optional[SRModule]=None,
        mcReturn: Optional[ValueModule]=None
    ):

        self.policy  = policy
        self.ValueNet = ValueNet
        self.QNet   = qNet
        self.SRNet   = SRNet
        self.seq_obsEncoder = obsEncoder
        self.mcReturn = mcReturn

