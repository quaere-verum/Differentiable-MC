from dataclasses import dataclass
from torch import Tensor

@dataclass
class SimulationState:
    spot: Tensor | None = None
    spot_previous: Tensor | None = None
    spot_cumulative_min: Tensor | None = None
    spot_cumulative_max: Tensor | None = None
    hidden_state: Tensor | None = None
    variance: Tensor | None = None
    short_rate: Tensor | None = None
    t_prev: float | None = None
    t: float | None = None
    t_next: float | None = None