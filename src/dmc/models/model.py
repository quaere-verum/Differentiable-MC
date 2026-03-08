import abc
from ..simulation.simulation_result import SimulationResult
from ..simulation.simulation_grid import SimulationGrid
import torch

class StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def simulate(self, batch_size: int) -> SimulationResult:
        raise NotImplementedError
    
    @abc.abstractmethod
    def bind(self, sim_grid: SimulationGrid) -> None:
        raise NotImplementedError