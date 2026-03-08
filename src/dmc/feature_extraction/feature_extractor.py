import torch
import abc
from ..simulation.simulation_state import SimulationState
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureExtractorResult:
    features: torch.Tensor
    hidden_state: torch.Tensor | None

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_features(self, state: SimulationState) -> FeatureExtractorResult:
        raise NotImplementedError
    
    @abc.abstractmethod
    def hidden_state_dim(self) -> int | None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError