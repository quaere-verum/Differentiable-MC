from dataclasses import dataclass
from torch import Tensor

@dataclass(frozen=True)
class SimulationResult:
    spot: Tensor | None
    variance: Tensor | None 
    short_rate: Tensor | None