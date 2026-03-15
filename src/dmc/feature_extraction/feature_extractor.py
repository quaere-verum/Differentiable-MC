import torch
import abc
from ..simulation.simulation_state import SimulationState

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def update_hidden_state(self, state: SimulationState) -> torch.Tensor | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_features(self, state: SimulationState) -> torch.Tensor:
        raise NotImplementedError
    
    @abc.abstractmethod
    def hidden_state_dim(self) -> int | None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError