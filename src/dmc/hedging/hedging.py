from dataclasses import dataclass
import torch
from ..models.model import StochasticModel
from ..products.product import Product
from ..control.controller import Controller
from ..feature_extraction.feature_extractor import FeatureExtractor
from .hedge_result import HedgeResult
from ..simulation.simulation_result import SimulationResult
from ..simulation.simulation_state import SimulationState
from ..simulation.simulation_grid import SimulationGrid

@dataclass(frozen=True)
class ControlIndices:
    start_idx: torch.Tensor
    end_idx: torch.Tensor

@dataclass(frozen=True)
class ControlIntervals:
    start_times: torch.Tensor
    end_times: torch.Tensor

    def n_intervals(self) -> int:
        return int(self.start_times.shape[0])
    
@dataclass
class HedgingEngine:
    product: Product
    controller: Controller
    feature_extractor: FeatureExtractor
    use_smooth_payoff: bool
    initial_cash: float
    transaction_cost_rate: float
    control_intervals: ControlIntervals

    def __post_init__(self):
        self.control_indices: ControlIndices = None
        self.simulation_grid: SimulationGrid = None

    def bind(self, sim_grid: SimulationGrid) -> None:
        self.control_indices = ControlIndices(
            start_idx=sim_grid.find_times_in_grid(self.control_intervals.start_times),
            end_idx=sim_grid.find_times_in_grid(self.control_intervals.end_times),
        )
        self.simulation_grid = sim_grid

    def run(self, simulated: SimulationResult) -> HedgeResult:
        assert self.control_indices is not None, ("Bind to SimulationGrid before calling run.")
        S = simulated.spot
        batch_size = S.size(0)

        payoff = self.product.compute_smooth_payoff(simulated) if self.use_smooth_payoff else self.product.compute_payoff(simulated)
        wealth = torch.full((batch_size,), self.initial_cash, dtype=torch.float32, device=S.device)
        position = torch.zeros((batch_size,), dtype=torch.float, device=S.device)

        state = SimulationState()

        state.t_prev = 0.0
        state.spot_previous = S.select(1, 0)
        state.spot_cumulative_min = S.select(1, 0)
        state.spot_cumulative_max = S.select(1, 0)
        hidden_state_dim = self.feature_extractor.hidden_state_dim()
        if hidden_state_dim:
            state.hidden_state = torch.zeros((batch_size, hidden_state_dim), dtype=torch.float32, device=S.device)

        last_obs_idx = 0
        time_grid = self.simulation_grid.time_grid
        for k in range(self.control_intervals.n_intervals()):
            t0 = self.control_indices.start_idx[k]
            t1 = self.control_indices.end_idx[k]
            state.t_next = self.control_intervals.end_times[k]

            for j in range(last_obs_idx + 1, t0):
                state.spot = S.select(1, j)

                state.spot_cumulative_min = torch.minimum(
                    state.spot_cumulative_min, state.spot
                )
                state.spot_cumulative_max = torch.maximum(
                    state.spot_cumulative_max, state.spot
                )

                if simulated.variance is not None:
                    state.variance = simulated.variance.select(1, j)
                if simulated.short_rate is not None:
                    state.short_rate = simulated.short_rate.select(1, j)

                state.t = time_grid[j]


                state.hidden_state = self.feature_extractor.update_hidden_state(state)
                state.t_prev = state.t
                state.spot_previous = state.spot

            last_obs_idx = t0
            state.spot = S.select(1, t0)
            state.spot_cumulative_min = torch.minimum(state.spot_cumulative_min, state.spot)
            state.spot_cumulative_max = torch.maximum(state.spot_cumulative_max, state.spot)

            if simulated.variance is not None:
                state.variance = simulated.variance.select(1, t0)
            if simulated.short_rate is not None:
                state.short_rate = simulated.short_rate.select(1, t0)

            state.t = time_grid[t0]

            state.hidden_state = self.feature_extractor.update_hidden_state(state)
            features = self.feature_extractor.get_features(state)
            hedge = self.controller.forward(features)
            trade = hedge - position
            cost = self.transaction_cost_rate * state.spot * torch.abs(trade)
            
            state.t_prev = state.t
            state.spot_previous = state.spot

            wealth += hedge * (S.select(1, t1) - state.spot) - cost
            position = hedge

        final_spot = S.select(1, self.control_indices.end_idx[-1])
        wealth -= self.transaction_cost_rate * final_spot * torch.abs(position)

        pnl = wealth - payoff
        return HedgeResult(pnl=pnl)