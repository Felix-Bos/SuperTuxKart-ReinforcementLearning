# modules/value.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass

class ValueOutput:
    value: Any                   # V(s) or Q(s,a)
    new_hidden: Optional[Any]    # for recurrent critics
    extra: dict                  # optional diagnostics

class ValueModule(ABC):
    @abstractmethod
    def evaluate(self, obs, hidden=None) -> ValueOutput:
        ...

class QModule(ABC):
    @abstractmethod
    def evaluate(self, obs, action, hidden=None) -> ValueOutput:
        ...

class SRModule(ABC):
    @abstractmethod
    def evaluate(self, obs, action, hidden=None) -> ValueOutput:
        """
        value is ψ(s,a) ∈ R^d (successor features)
        """
        ...
