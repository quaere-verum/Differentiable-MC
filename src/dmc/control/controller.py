import torch
import abc

class Controller(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError