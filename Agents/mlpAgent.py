import torch
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from .types import Transition, Batch
from typing import Dict, Any, Optional, Literal
from PolicyModule.basePolicyModule import PolicyModule
from ValueModule.baseValueModule import ValueModule, QModule, SRModule
from buffers.base_buffer import ReplayBuffer
from buffers.types import Transition, Batch
from algorithms.baseAlgorithm import RLAlgorithm, get_algorithm, 
from obsEncoder.mlpObsEncoder import mlpObsEncoder



class MLPAgent(Agent):
    def __init__(
        algo: str,
        device,
        name,
        obsEncoder: Optional[mlpObsEncoder],
        policy: Optional[PolicyModule],
        valueNet: Optional[ValueModule],
        qNet: Optional[QModule],
        SRNet: Optional[SRModule],
        optimizer_config,
    ):
        super().__init__(obsEncoder, policy, valueNet, qNet, SRNet)
        self.name = name
        self.device = device

        # define algorithm
        AlgorithmClass = get_algorithm(algo)
        self.algo = AlgorithmClass(self, config['algo_config'])




        
