from .product import Product
from ..simulation.simulation_grid import SimulationGrid
from ..simulation.simulation_result import SimulationResult
import torch

class DownAndOutBarrierOption(Product):
    def __init__(
        self,
        maturity: float,
        barrier: float,
        strike: float,
        observation_grid: torch.Tensor,
        *,
        softplus_beta: float = 10.0,
        barrier_beta: float = 20.0,
    ):
        super().__init__()
        self.register_buffer("maturity", torch.tensor(maturity, dtype=torch.float32))
        self.register_buffer("barrier", torch.tensor(barrier, dtype=torch.float32))
        self.register_buffer("strike", torch.tensor(strike, dtype=torch.float32))
        self.register_buffer("obs_grid", observation_grid)
        self._softplus_beta = softplus_beta
        self._barrier_beta = barrier_beta
        self._indices_monitored: torch.Tensor = None
        self._index_maturity: int = None

        if maturity in observation_grid:
            self.register_buffer("timegrid", observation_grid.clone())
        else:
            self.register_buffer("timegrid", torch.cat((
                observation_grid, 
                torch.tensor([maturity]),
            )))

    def get_timegrid(self):
        return SimulationGrid(self.get_buffer("timegrid"))
    
    def bind(self, sim_grid: SimulationGrid):
        self._indices_monitored = sim_grid.find_times_in_grid(self.get_buffer("timegrid"))
        self._index_maturity = int(sim_grid.find_times_in_grid(self.get_buffer("maturity").view(1)).item())

    def compute_payoff(self, simulated: SimulationResult):
        assert self._indices_monitored is not None
        monitored_spot = simulated.spot[:, self._indices_monitored]
        min_price = torch.min(monitored_spot, dim=1).values
        final_price = simulated.spot[:, self._index_maturity]

        intrinsic = torch.relu(final_price - self.get_buffer("strike"))
        alive = (min_price > self.get_buffer("barrier")).to(torch.float32)
        return intrinsic * alive
    
    def compute_smooth_payoff(self, simulated: SimulationResult):
        assert self._indices_monitored is not None
        monitored_spot = simulated.spot[:, self._indices_monitored]
        min_price = torch.min(monitored_spot, dim=1).values
        final_price = simulated.spot[:, self._index_maturity]

        intrinsic = torch.nn.functional.softplus(final_price - self.get_buffer("strike"), self._softplus_beta)
        alive = torch.sigmoid(self._barrier_beta * (min_price - self.get_buffer("barrier"))).to(torch.float32)
        return intrinsic * alive
    
    def final_maturity(self):
        return float(self.get_buffer("maturity").item())