# MLP.py 
# This file defines a Multi-Layer Perceptron (MLP) class using PyTorch, allowing for customizable architecture based on provided parameters.


import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict


def get_activation(name: str) -> nn.Module:
    if name is None or name.lower() in ("identity", "linear", ""):
        return nn.Identity()
    name = name.lower()
    if name == "relu":       return nn.ReLU()
    if name == "leakyrelu":  return nn.LeakyReLU()
    if name == "elu":        return nn.ELU()
    if name == "tanh":       return nn.Tanh()
    if name == "sigmoid":    return nn.Sigmoid()
    if name == "silu" or name == "swish": return nn.SiLU()
    if name == "gelu":       return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        params: Dict,
        output_dim: int, 
        device: Optional[str] = None
    ):
        """
        A wrapper for building MLP from given parameters.
        args:
        input_dim: dimension of input features
        params: dictionary containing MLP parameters with keys: hidden_size (list of int),hidden_activation (list of str), output_activation (str)
        output_dim: dimension of output features
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        
        layers = []
        prev_dim = input_dim
        for i, h in enumerate(params["hidden_size"]):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(get_activation(params.get("hidden_activation")[i]))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        if "output_activation" in params and params["output_activation"]:
            layers.append(get_activation(params["output_activation"][0]))
        self.net = nn.Sequential(*layers)
        
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.device))
