import torch
import abc
from ..simulation.simulation_result import SimulationResult
from ..simulation.simulation_grid import SimulationGrid

class Product(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def compute_payoff(self, simulated: SimulationResult) -> torch.Tensor:
        raise NotImplementedError
    
    @abc.abstractmethod
    def compute_smooth_payoff(self, simulated: SimulationResult) -> torch.Tensor:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_timegrid(self) -> SimulationGrid:
        raise NotImplementedError

    @abc.abstractmethod
    def bind(self, sim_grid: SimulationGrid) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def final_maturity(self) -> float:
        raise NotImplementedError